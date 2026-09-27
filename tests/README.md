# Tests

- `unit/`: deterministic component tests.
- `integration/`: boundaries between the gateway, smolagents, vLLM, AppWorld, storage, and statistics, including canary-world isolation.
- `validity/`: simulated-null, coverage, false-alarm, and reference checks.
- `ci/`: offline candidate-versus-production gate checks.
- `e2e/`: service, canary, rollback, and alert paths.
- `fixtures/`: small tracked inputs only, with no AppWorld task content.
