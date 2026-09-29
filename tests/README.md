# Tests

Current suites:

- `unit/`: 186 deterministic tests for configuration, records, task selection and streams, fake-driven episodes, lazy runtime adapters, storage, export, the statistics formulas, versioned faults, the plan-only offline gate, the canary controller, the production monitor, and shared-stream detector replay (`test_replay.py`, 8 tests). Service unit tests skip when FastAPI is absent.
- `ci/`: 7 tests for the plan-only offline gate.
- `integration/`: 33 synthetic tests. They connect a catalog, seeded stream, fake session and agent, episode storage, statistics isolation, and public export, and they exercise `run_pair`, A/A capture, the offline gate, the canary controller, the production monitor, and the lifecycle benchmark harness (`test_lifecycle_benchmark.py`, 6 tests) on injected worlds. They do not run smolagents, vLLM, or AppWorld, and they are not a noise-floor result or a `test_normal` benchmark run. Service integration tests skip when FastAPI is absent.
- `validity/`: 19 CPU tests for the validation runner. They cover catalog eligibility, a constant-null canary with supplied A/A rows, a failed reference, degenerate bootstrap coverage, repeated looks, KL truncation on supplied arrays, harm-study feasibility, capture-row copying, study budgets, and the three evaluation commands. They do not read `nvidia-smi`, start AppWorld or vLLM, or commit a null, A/A, or truncation result. The GitHub workflow does not discover this suite.

Reserved but empty:

- `e2e/`: live service, canary, rollback, and alert paths.
- `fixtures/`: future small tracked inputs only, with nothing from the local-only list in `docs/DATA.md`.

The GitHub workflow installs `environments/test-requirements.txt` (FastAPI, Uvicorn, Prometheus, SciPy, River, pytest, httpx — no AppWorld), sets `PYTHONPATH=src`, and runs `tests/unit`, `tests/ci`, `tests/integration`, and `tests/validity` with `python -m unittest discover -s tests/<suite> -p "test_*.py"`, then checks that bare `python scripts/run_offline_gate.py`, `scripts/benchmark/run_lifecycle_benchmark.py`, `scripts/replay/replay_detectors.py`, `scripts/service/serve.py`, and `scripts/evaluation/lock_protocol.py` each exit 2, then runs the synthetic connected lifecycle. pytest is pinned but is not the current test runner.
