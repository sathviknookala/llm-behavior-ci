# Tests

Current suites (counts as of the hosted-lifecycle branch):

- `unit/`: 668 deterministic tests for configuration, records, task selection and streams, fake-driven episodes, lazy runtime adapters, hosted providers and hosted plan mode (`test_hosted_plan_mode.py`), hosted fault payloads (`test_hosted_faults.py`), the hashed arrival schedule (`test_schedule.py`), runtime routing, alert incidents, slice references, usage, empirical power, and the protocol commitment (`test_lifecycle_wiring.py`), storage, export, the statistics formulas, versioned faults, the plan-only offline gate, the canary controller, the production monitor, and shared-stream detector replay. Builder tests that need git-ignored output paths create a temporary git repository, so they do not depend on a local `.git/info/exclude`. Service unit tests raise `ImportError` when FastAPI is absent.
- `ci/`: 28 tests for the plan-only offline gate and the bare-CLI exit-2 contract of every lifecycle command.
- `integration/`: 79 synthetic tests. They connect a catalog, seeded stream, fake session and agent, episode storage, statistics isolation, and public export, and they exercise `run_pair`, A/A capture, the offline gate (including a hosted gate CLI), the canary controller, the production monitor, the service (including a hosted production config, promotion resets, and train-to-dev admission through `serve.py --task-selection-allowance`), and the lifecycle benchmark harness on injected worlds. They do not run smolagents, vLLM, AppWorld, or a hosted provider, and they are not a noise-floor result or a `test_normal` benchmark run.
- `validity/`: 19 CPU tests for the validation runner. They do not read `nvidia-smi`, start AppWorld or vLLM, or commit a null, A/A, or truncation result.

Every test here is software evidence. A passing suite closes no stage gate.

Reserved but empty:

- `e2e/`: live service, canary, rollback, and alert paths.
- `fixtures/`: future small tracked inputs only, with nothing from the local-only list in `docs/DATA.md`.

The GitHub workflow installs `environments/test-requirements.txt` (FastAPI, Uvicorn, Prometheus, SciPy, River, pytest, httpx — no AppWorld), sets `PYTHONPATH=src`, and runs `tests/unit`, `tests/ci`, `tests/integration`, and `tests/validity` with `python -m unittest discover -s tests/<suite> -p "test_*.py"`, then checks that bare `python scripts/run_offline_gate.py`, `scripts/benchmark/run_lifecycle_benchmark.py`, `scripts/replay/replay_detectors.py`, `scripts/service/serve.py`, and `scripts/evaluation/lock_protocol.py` each exit 2, then runs the synthetic connected lifecycle. pytest is pinned but is not the current test runner.
