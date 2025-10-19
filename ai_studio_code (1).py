import os
import sys
import argparse
import threading
import uvicorn
import time
import signal
from typing import Optional, Dict, Any
from rich.pretty import pprint

"""
ai_studio_code.py
- Adds: robust config validation, --mode branching (server/train/eval),
  graceful shutdown hooks, and logger usage notes.
- Note: This file assumes the efca_adapt package structure exists. If not,
  replace the imports with local equivalents or guards.
"""

try:
    from efca_adapt.utils.config_loader import load_config
    from efca_adapt.infra.logger import setup_logger
    from efca_adapt.utils.torch_utils import get_device, set_seed
    from efca_adapt.agent.meta_agent import MetaAgent
    from efca_adapt.adapt_platform.environment import MetaRLToyEnv
    from efca_adapt.infra.monitoring import MLOpsMonitor
    from efca_adapt.api import server
except ImportError as e:
    print(f"ERROR: Failed to import required modules: {e}")
    print("Please ensure all dependencies are installed: pip install -r requirements.txt")
    sys.exit(1)

api_server_instance: Optional[uvicorn.Server] = None
shutdown_event = threading.Event()


def signal_handler(signum, frame):
    """Handle SIGINT/SIGTERM for graceful shutdown.
    - Set shutdown_event so loops can exit cooperatively
    - Signal uvicorn server to exit
    """
    print("\n[INFO] Shutdown signal received. Cleaning up...")
    shutdown_event.set()
    if api_server_instance:
        api_server_instance.should_exit = True


def validate_config(config) -> bool:
    """Validate minimal structure of config.
    Expected top-level sections: operational, system, environment, monitoring, api
    Required fields:
      - system.seed, system.device
      - api.host, api.port
      - operational.log_level, optional: save_interval
    """
    required_sections = ["operational", "system", "environment", "monitoring", "api"]
    for sec in required_sections:
        if not hasattr(config, sec):
            raise ValueError(f"Config missing required section: {sec}")

    if not hasattr(config.system, "seed"):
        raise ValueError("Config missing system.seed")
    if not hasattr(config.system, "device"):
        raise ValueError("Config missing system.device")

    if not hasattr(config.api, "host") or not hasattr(config.api, "port"):
        raise ValueError("Config missing api.host or api.port")

    if not hasattr(config.operational, "log_level"):
        raise ValueError("Config missing operational.log_level")

    # Provide safe defaults
    if not hasattr(config.operational, "save_interval"):
        setattr(config.operational, "save_interval", 10)

    return True


def start_api_server(host: str, port: int, log_level: str = "error") -> threading.Thread:
    """Start FastAPI server in a daemon thread and return the thread."""
    global api_server_instance
    api_config = uvicorn.Config(server.app, host=host, port=port, log_config=None, log_level=log_level)
    api_server_instance = uvicorn.Server(api_config)
    api_thread = threading.Thread(target=api_server_instance.run, daemon=True)
    api_thread.start()
    return api_thread


def run_training_loop(log, env, agent, monitor, save_interval: int):
    log.info("[bold magenta]Starting training loop...[/bold magenta]")
    episode = 0
    while not shutdown_event.is_set():
        try:
            episode += 1
            log.info("\n" + "=" * 60)
            log.info(f"[bold]Episode {episode}[/bold]")
            log.info("=" * 60)

            obs = env.reset()
            done = False
            total_reward = 0
            step = 0

            while not done and not shutdown_event.is_set():
                step += 1
                action = agent.act(obs)
                next_obs, reward, done, info = env.step(action)
                agent.learn(obs, action, reward, next_obs, done)
                total_reward += reward
                obs = next_obs
                if step % 100 == 0:
                    log.debug(f"Step {step}: reward={reward:.4f}")

            log.info(f"Episode {episode} finished: total_reward={total_reward:.4f}, steps={step}")
            monitor.log_metric("episode_reward", total_reward, episode)
            monitor.log_metric("episode_length", step, episode)

            if episode % save_interval == 0:
                log.info("[bold cyan]Saving checkpoint...[/bold cyan]")
                agent.save_weights()
                monitor.save_metrics()

        except KeyboardInterrupt:
            log.info("\n[yellow]Training interrupted by user[/yellow]")
            break
        except Exception as e:
            log.error(f"[bold red]Error in training loop: {e}[/bold red]")
            log.exception(e)
            continue

    log.info("[bold green]Training loop exiting.[/bold green]")


def run_eval_loop(log, env, agent):
    log.info("[bold magenta]Starting evaluation loop...[/bold magenta]")
    try:
        obs = env.reset()
        done = False
        total_reward = 0
        step = 0
        while not done and not shutdown_event.is_set():
            step += 1
            action = agent.act(obs, deterministic=True) if hasattr(agent, "act") else agent.predict(obs)
            obs, reward, done, info = env.step(action)
            total_reward += reward
        log.info(f"[bold green]Eval completed: total_reward={total_reward:.4f}, steps={step}[/bold green]")
    except Exception as e:
        log.error(f"Evaluation error: {e}")
        log.exception(e)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="EFCA-ADAPT-AG Runner")
    parser.add_argument("--mode", choices=["server", "train", "eval"], default="server", help="Run mode")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config file")
    parser.add_argument("--port", type=int, default=None, help="Override API port")
    return parser.parse_args()


def main():
    global api_server_instance

    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        args = parse_args()
        print("[INFO] Loading configuration...")
        config = load_config(args.config) if args.config else load_config()
        validate_config(config)

        # Initialize logger early; ensure consistent formatting and levels
        log = setup_logger(getattr(config.operational, "log_level", "INFO"))
        log.info("[bold green]Initializing EFCA-ADAPT System...[/bold green]")
        try:
            # If config object supports .dict() (pydantic), pretty print it
            cfg_dict: Dict[str, Any] = config.dict() if hasattr(config, "dict") else vars(config)
            pprint(cfg_dict)
        except Exception:
            pass

        # Reproducibility and device
        set_seed(config.system.seed)
        device = get_device(config.system.device)
        log.info(f"Using device: [bold cyan]{device}[/bold cyan]")

        # Construct environment, agent, monitoring
        log.info("Setting up environment and agent...")
        env = MetaRLToyEnv(config.environment)
        agent = MetaAgent(config, env.obs_dim, env.action_dim)
        monitor = MLOpsMonitor(config.monitoring)
        server.set_agent_instance(agent)

        # Load weights if available
        try:
            agent.load_weights()
            log.info("[bold green]Successfully loaded agent weights[/bold green]")
        except FileNotFoundError:
            log.warning("[bold yellow]No saved weights found. Starting from scratch.[/bold yellow]")
        except Exception as e:
            log.warning(f"[bold yellow]Could not load weights: {e}[/bold yellow]")

        # Branch by mode
        mode = args.mode
        if mode == "server":
            # Start API only
            log.info("Starting API server...")
            if args.port is not None:
                config.api.port = args.port
            api_thread = start_api_server(config.api.host, config.api.port)
            time.sleep(1.5)
            log.info(f"[bold green]API Server running at http://{config.api.host}:{config.api.port}[/bold green]")

            # Wait until shutdown
            while not shutdown_event.is_set():
                time.sleep(0.5)

        elif mode == "train":
            # Start API in background for monitoring while training
            log.info("Starting background API server for training mode...")
            api_thread = start_api_server(config.api.host, config.api.port)
            time.sleep(1.5)
            log.info(f"[bold green]API Server running at http://{config.api.host}:{config.api.port}[/bold green]")
            run_training_loop(log, env, agent, monitor, save_interval=config.operational.save_interval)

        elif mode == "eval":
            # Optional: start API for live inspection during eval
            log.info("Starting background API server for eval mode (optional)...")
            api_thread = start_api_server(config.api.host, config.api.port)
            time.sleep(1.5)
            log.info(f"[bold green]API Server running at http://{config.api.host}:{config.api.port}[/bold green]")
            run_eval_loop(log, env, agent)

        # On exit paths, persist state
        log.info("[bold green]Saving final state and shutting down...[/bold green]")
        try:
            agent.save_weights()
            monitor.save_metrics()
        except Exception as e:
            log.warning(f"During shutdown, save failed: {e}")

        log.info("[bold green]Shutdown complete.[/bold green]")

    except Exception as e:
        print(f"[FATAL ERROR] System initialization failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
