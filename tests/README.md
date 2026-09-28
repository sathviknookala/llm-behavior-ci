# Tests

Current suites:

- `unit/`: 159 deterministic tests for configuration, records, task selection and streams, fake-driven episodes, lazy runtime adapters, storage, export, the statistics formulas, versioned faults, the plan-only offline gate, the canary controller, and the production monitor.
- `ci/`: 7 tests for the plan-only offline gate.
- `integration/`: 26 synthetic tests. They connect a catalog, seeded stream, fake session and agent, episode storage, statistics isolation, and public export, and they exercise `run_pair`, A/A capture, the offline gate, the canary controller, and the production monitor on injected worlds. They do not run the gateway, smolagents, vLLM, or AppWorld, and they are not a noise-floor result.
- `validity/`: 19 CPU tests for the validation runner. They cover catalog eligibility, a constant-null canary with supplied A/A rows, a failed reference, degenerate bootstrap coverage, repeated looks, KL truncation on supplied arrays, harm-study feasibility, capture-row copying, study budgets, and the three evaluation commands. They do not read `nvidia-smi`, start AppWorld or vLLM, or commit a null, A/A, or truncation result. The GitHub workflow does not discover this suite.

Reserved but empty:

- `e2e/`: live service, canary, rollback, and alert paths.
- `fixtures/`: future small tracked inputs only, with nothing from the local-only list in `docs/DATA.md`.

The GitHub workflow sets `PYTHONPATH=src` and runs `tests/unit`, `tests/ci`, and `tests/integration` with `python -m unittest discover -s tests/<suite> -p "test_*.py"`, then checks that a bare `python scripts/run_offline_gate.py` exits 2. `tests/validity/` is run locally with the same discover command and is not in that workflow. pytest is pinned but is not the current test runner.
