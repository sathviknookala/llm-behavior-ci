# Docker

All three image contexts currently contain only placeholders. There are no Dockerfiles, image overrides, scrape configurations, or root Compose file.

- `gateway/`: image context for this repo's FastAPI gateway and the smolagents runtime it drives.
- `vllm/`: pinned vLLM launch configuration or image overrides, one service per configuration during a canary.
- `prometheus/`: behavior and serving metric scrape configuration.

The future Compose file will live at the repository root. AppWorld runs where `docs/DECISIONS.md` D20 places it; no separate environment image is added otherwise. Grafana is not initialized until `docs/DECISIONS.md` D11 is resolved.
