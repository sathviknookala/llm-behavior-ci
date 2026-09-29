# Results

No measurements exist as of 2026-09-28. Every stage-aligned directory below is an empty placeholder. `scripts/evaluation/capture_aa.py`, `validate_method.py`, `compare_plan_kl.py`, and `assess_harm_study.py` refuse to write here. `scripts/benchmark/run_lifecycle_benchmark.py` refuses checkpoint and export paths under a directory named `results`. `scripts/replay/replay_detectors.py` refuses its input and output paths the same way. `scripts/service/serve.py` refuses a store path the same way. A harness export outside that tree is not a result in `results/`. A local A/A capture, a validation summary, a supplied-sample KL error, and a planning power number are not results in this tree.

Raw artifacts and the scripts that regenerate every figure will live here. A number outside this directory is not a project result. Only the public items in `docs/DATA.md` belong here: versions and configuration hashes, split names, selection rules and seeds, scenario, task, and episode counts, task-set hashes, and aggregate statistics, confidence intervals, detector outputs, cost and latency summaries, and figures. Resolved task IDs, per-task outcomes, plans, trajectories, evaluator reports, and raw logs stay local.

- `noise_floor/`: stage 0 A/A outcome, trajectory, plan-trace KL, and batch-invariance measurements on `train` and `dev`.
- `task_baselines/`: stage 1 production and do-nothing success on `dev`, the power calculation, and the task-mix curve.
- `validity/`: stage 2 null, coverage, reference, and KL-truncation checks.
- `release_gate/`: stage 3 `dev` harm labels, Tier 1 gate power on `train`, false blocks, gate cost, and gate-versus-harm agreement.
- `canary/`: stage 4 Tier 2 development on `dev`: rollback traces and candidate episodes served before rollback.
- `monitors/`: stage 5 Tier 3 development on `dev`: live-stream and de-duplicated alert evidence.
- `benchmark/`: stage 6 frozen `test_normal` benchmark: the shared-stream comparison and the end-to-end tier flow.
- `figures/`: outputs regenerated from the raw artifacts above.
