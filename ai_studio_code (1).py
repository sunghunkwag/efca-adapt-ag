import os
import threading
import uvicorn
import time
from rich.pretty import pprint

from efca_adapt.utils.config_loader import load_config
from efca_adapt.infra.logger import setup_logger
from efca_adapt.utils.torch_utils import get_device, set_seed
from efca_adapt.agent.meta_agent import MetaAgent
from efca_adapt.adapt_platform.environment import MetaRLToyEnv
from efca_adapt.infra.monitoring import MLOpsMonitor
from efca_adapt.api import server

def main():
    """
    Main entry point for the EFCA-ADAPT agent system.
    Initializes all components, starts background services, and runs the main training loop.
    """
    # --- 1. Initialization ---
    config = load_config()
    log = setup_logger(config.operational.log_level)
    log.info("[bold green]Initializing EFCA-ADAPT System...[/bold green]")
    pprint(config.dict())

    set_seed(config.system.seed)
    device = get_device(config.system.device)
    log.info(f"Using device: [bold cyan]{device}[/bold cyan]")
    
    # --- 2. Component Setup ---
    env = MetaRLToyEnv(config.environment)
    agent = MetaAgent(config, env.obs_dim, env.action_dim)
    monitor = MLOpsMonitor(config.monitoring)
    server.set_agent_instance(agent) # Make agent instance available to the API
    
    # --- 3. Load Weights ---
    # agent.load_weights()

    # --- 4. Start Background Services ---
    log.info("Starting background services (API Server)...")
    api_config = uvicorn.Config(server.app, host=config.api.host, port=config.api.port, log_config=None)
    api_server = uvicorn.Server(api_config)
    api_thread = threading.Thread(target=api_server.run, daemon=True)
    api_thread.start()
    log.info(f"API server running at http://{config.api.host}:{config.api.port}")

    # --- 5. Main Training Loop ---
    log.info("[bold green]Starting Meta-Training Loop...[/bold green]")
    start_time = time.time()

    for meta_epoch in range(1, config.meta_learning.meta_epochs + 1):
        epoch_start_time = time.time()
        
        metrics = agent.meta_update(env, monitor)
        
        if meta_epoch % config.operational.log_interval == 0:
            duration = time.time() - epoch_start_time
            log.info(
                f"Meta-Epoch: [bold yellow]{meta_epoch}[/] | "
                f"Meta Loss: {metrics.get('meta_total_loss', 0):.4f} | "
                f"Duration: {duration:.2f}s"
            )

        if meta_epoch % config.operational.save_weights_interval == 0:
            # agent.save_weights()
            pass
            
    total_time = time.time() - start_time
    log.info(f"[bold green]Meta-Training finished in {total_time / 3600:.2f} hours.[/bold green]")


if __name__ == "__main__":
    main()