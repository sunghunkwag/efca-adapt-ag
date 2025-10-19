import os
import sys
import threading
import uvicorn
import time
import signal
from typing import Optional
from rich.pretty import pprint

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
    print("\n[INFO] Shutdown signal received. Cleaning up...")
    shutdown_event.set()
    if api_server_instance:
        api_server_instance.should_exit = True
    sys.exit(0)

def validate_config(config):
    required_attrs = ['operational', 'system', 'environment', 'monitoring', 'api']
    for attr in required_attrs:
        if not hasattr(config, attr):
            raise ValueError(f"Config missing required attribute: {attr}")
    if not hasattr(config.system, 'seed'):
        raise ValueError("Config missing system.seed")
    if not hasattr(config.api, 'host') or not hasattr(config.api, 'port'):
        raise ValueError("Config missing api.host or api.port")
    return True

def main():
    global api_server_instance
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    try:
        print("[INFO] Loading configuration...")
        config = load_config()
        validate_config(config)
        log = setup_logger(config.operational.log_level)
        log.info("[bold green]Initializing EFCA-ADAPT System...[/bold green]")
        pprint(config.dict())
        set_seed(config.system.seed)
        device = get_device(config.system.device)
        log.info(f"Using device: [bold cyan]{device}[/bold cyan]")
        log.info("Setting up environment and agent...")
        env = MetaRLToyEnv(config.environment)
        agent = MetaAgent(config, env.obs_dim, env.action_dim)
        monitor = MLOpsMonitor(config.monitoring)
        server.set_agent_instance(agent)
        try:
            agent.load_weights()
            log.info("[bold green]Successfully loaded agent weights[/bold green]")
        except FileNotFoundError:
            log.warning("[bold yellow]No saved weights found. Starting from scratch.[/bold yellow]")
        except Exception as e:
            log.warning(f"[bold yellow]Could not load weights: {e}[/bold yellow]")
        log.info("Starting background services (API Server)...")
        api_config = uvicorn.Config(server.app, host=config.api.host, port=config.api.port, log_config=None, log_level="error")
        api_server_instance = uvicorn.Server(api_config)
        api_thread = threading.Thread(target=api_server_instance.run, daemon=True)
        api_thread.start()
        time.sleep(2)
        log.info(f"[bold green]API Server running at http://{config.api.host}:{config.api.port}[/bold green]")
        log.info("[bold magenta]Starting main training loop...[/bold magenta]")
        episode = 0
        while not shutdown_event.is_set():
            try:
                episode += 1
                log.info(f"\n{'='*60}")
                log.info(f"[bold]Episode {episode}[/bold]")
                log.info(f"{'='*60}")
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
                if episode % config.operational.save_interval == 0:
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
        log.info("[bold green]Training completed. Saving final checkpoint...[/bold green]")
        agent.save_weights()
        monitor.save_metrics()
        log.info("[bold green]Shutdown complete.[/bold green]")
    except Exception as e:
        print(f"[FATAL ERROR] System initialization failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
