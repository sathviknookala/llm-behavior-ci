# Scripts

`run_offline_gate.py` runs the plan-only offline gate. It requires `--reference`, `--candidate`, `--task-set`, and `--settings`. With no arguments it exits 2. A PASS prints a public JSON decision and exits 0; a BLOCK exits 1. It is not a protocol release gate.

`evaluation/capture_aa.py` pairs one configuration with itself on a supplied finite stream. Repetitions, concurrency, modes, and the vLLM base URL are required arguments. The command writes a local capture and refuses a path under `results/`. It rejects `test_normal`. `--observe-hardware` records memory and wall time from `nvidia-smi`; without that flag those fields stay empty. The command is not a noise-floor measurement and applies no protocol threshold.

`evaluation/validate_method.py` validates one method from a JSON spec. `evaluation/compare_plan_kl.py` scores a supplied full-vocabulary sample at an explicit top-k. `evaluation/assess_harm_study.py` reports harm-study feasibility from explicit margin, variance, sample sizes, and optional evaluator outcomes. `evaluation/characterize_harm.py` applies a versioned fault diff and can freeze a dev harm label from execute-mode pairs. Each writes outside `results/` or refuses that path, and does not read `nvidia-smi` or start AppWorld or vLLM. A passing local run is not a committed null check, truncation floor, power result, or frozen harm label. A caller-supplied margin is not a protocol harm margin.

`evaluation/lock_protocol.py` records caller-supplied settings. With no arguments it exits 2. A lock is not pre-registration.

`service/serve.py` starts the FastAPI gateway. With missing required arguments it exits 2 and refuses a store path under a directory named `results`. There is no live AppWorld episode, load run, or Compose stack.

`service/launch_vllm.py` starts one vLLM server for the `model` section of `--configuration` (a model file such as `configs/models/qwen3_4b_production.json` or a full run configuration). Run it with the vLLM environment's interpreter: `PYTHONPATH=src .venv-vllm/bin/python scripts/service/launch_vllm.py --configuration ...`. The argv is `build_vllm_launch_spec`'s, plus `--host` and `--port`; the environment is the spec's, which always includes `VLLM_USE_FLASHINFER_SAMPLER` from `model.serving.sampler_backend`, plus the machine-only `TRITON_PTXAS_BLACKWELL_PATH` (`docs/CONSTRAINTS.md`). `--dry-run` prints the argv and environment as JSON. It execs `vllm` in the foreground; detach it with `setsid`. With no arguments, or from an interpreter without vLLM, it exits 2.

`benchmark/run_lifecycle_benchmark.py` runs `run_lifecycle_benchmark` for caller-supplied faults and a caller-supplied `ProtocolLock`. With no arguments it exits 2. It requires `--protocol`, one or more `--fault`, `--train-tasks`, `--test-normal-tasks`, `--baselines`, `--checkpoint`, optional `--export`, and env `LLM_BEHAVIOR_CI_RUNTIME` as `module:function` returning `RuntimeDependencies`. Checkpoint and export paths under a directory named `results` are refused. It prints `{"status": ...}` and exits 0 only when status is `completed`, else 1 on a handled failure. The GitHub workflow only checks that a bare invocation exits 2; it does not run this script with real arguments. A harness run on injected tasks is not the frozen `test_normal` benchmark. `BenchmarkResult.gpu_memory_mib` and `gpu_hours` stay None.

`replay/replay_detectors.py` runs `replay_detectors` on one frozen observation sequence. With no arguments it exits 2. It requires `--observations`, `--schedule`, `--factories`, and `--output`. Any of those paths under a directory named `results` is refused. It writes a public JSON document and does not start an agent. The GitHub workflow only checks that a bare invocation exits 2; it does not run this script with real arguments.

`data/` is still a placeholder for AppWorld install checks, deterministic task-set selection, and manifest entry points.

The harm-study and validity commands plan from supplied inputs. They do not measure an AppWorld baseline or a live truncation floor.
