# Scripts

`run_offline_gate.py` runs the plan-only offline gate. It requires `--reference`, `--candidate`, `--task-set`, and `--settings`. With no arguments it exits 2. A PASS prints a public JSON decision and exits 0; a BLOCK exits 1. It is not a protocol release gate.

`evaluation/capture_aa.py` pairs one configuration with itself on a supplied finite stream. Repetitions, concurrency, modes, and the vLLM base URL are required arguments. The command writes a local capture and refuses a path under `results/`. It rejects `test_normal`. `--observe-hardware` records memory and wall time from `nvidia-smi`; without that flag those fields stay empty. The command is not a noise-floor measurement and applies no protocol threshold.

`evaluation/validate_method.py` validates one method from a JSON spec. `evaluation/compare_plan_kl.py` scores a supplied full-vocabulary sample at an explicit top-k. `evaluation/assess_harm_study.py` reports harm-study feasibility from explicit margin, variance, sample sizes, and optional evaluator outcomes. `evaluation/characterize_harm.py` applies a versioned fault diff and can freeze a dev harm label from execute-mode pairs. Each writes outside `results/` or refuses that path, and does not read `nvidia-smi` or start AppWorld or vLLM. A passing local run is not a committed null check, truncation floor, power result, or frozen harm label. A caller-supplied margin is not a protocol harm margin.

The other subdirectories are still placeholders:

- `data/`: AppWorld install checks, deterministic task-set selection, and manifest entry points.
- `service/`: gateway, vLLM, and configuration lifecycle entry points.
- `replay/`: task-stream launchers.
- `benchmark/`: pre-registered benchmark launchers and analysis.

No service, replay, or benchmark launcher exists yet. The harm-study and validity commands plan from supplied inputs. They do not measure an AppWorld baseline or a live truncation floor.
