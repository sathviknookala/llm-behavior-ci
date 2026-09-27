# Results

No measurements exist.

Raw artifacts and the scripts that regenerate every figure will live here. A number outside this directory is not a project result. Artifacts carry task IDs, outcomes, and statistics, never AppWorld task content, plan traces, or trajectories in plain text (`docs/DATA.md`).

- `noise_floor/`: stage 0 A/A outcome, trajectory, plan-trace KL, and batch-invariance measurements.
- `task_baselines/`: stage 1 production and do-nothing success, the power calculation, and the task-mix curve.
- `validity/`: stage 2 null, coverage, reference, and KL-truncation checks.
- `release_gate/`: stage 3 execution-measured harm labels, gate power, false blocks, gate cost, and gate-versus-harm agreement.
- `canary/`: stage 4 rollback traces and candidate episodes served before rollback.
- `monitors/`: stage 5 live-stream and de-duplicated alert evidence.
- `benchmark/`: stage 6 shared-stream comparison artifacts.
- `figures/`: outputs regenerated from the raw artifacts above.
