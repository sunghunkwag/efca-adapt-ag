# EFCA-ADAPT-AG Architecture

This document outlines the high-level architecture of the EFCA-ADAPT-AG system.

- API Layer (FastAPI): Exposes endpoints for inference, training triggers, metrics, and health.
- Agent Layer: MetaAgent implementing policy, adaptation logic, and PPO-like updates.
- Environment Layer: MetaRLToyEnv for simulation/testing; replaceable with real tasks.
- Infra Layer: Logger, monitoring (Prometheus-compatible), config loader, torch utilities.

Data Flow:
1. Requests enter FastAPI. Agent instance is injected.
2. Agent interacts with Environment for actions/observations.
3. Monitoring collects metrics; logger records structured logs.
4. Persistent storage for checkpoints and metrics.

Sequence (Train mode):
- Initialize config, logger, device
- Start API in background
- Loop episodes: act -> step -> learn -> log -> checkpoint

Sequence (Eval/Server modes):
- Initialize components
- Start API
- Serve requests or run single evaluation
