# Docker

- `gateway/`: image context for this repo's FastAPI gateway.
- `vllm/`: pinned vLLM launch configuration or image overrides.
- `prometheus/`: behavior and serving metric scrape configuration.

The Compose file will live at the repository root. Grafana is not initialized until `docs/DECISIONS.md` D11 is resolved.
