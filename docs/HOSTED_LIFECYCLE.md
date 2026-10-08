# Hosted GLM lifecycle: contract and runbook

**Status: DRAFT software path. Nothing in this file is a measurement.** Every command below is paid Z.AI API use or local GPU use and needs Sathvik's approval before it runs (`CONSTRAINTS.md`). No command here touches `test_normal` except the last section, which stays closed until `EVAL_PROTOCOL.md` is pre-registered.

## What is settled and what is not

- **GLM-5.3 is a provisional hosted reference.** It is a development reference for building and calibrating the three tiers on `train` and `dev`. It is not the Qwen3-4B production path, not a qualified Stage 1 baseline, and not a model this repo controls: the provider can change it underneath a fixed configuration hash.
- **Sonnet's 14/20 is not a GLM baseline.** `results/spotify_capability_20_sonnet_5_5.json` is a different provider, model, and configuration hash. It says nothing about GLM capability.
- **Tier 1 on a hosted model is plan-quality bootstrap plus MMD.** Teacher-forced plan KL needs prompt-token logprobs and a tokenizer, which only a self-hosted vLLM configuration has. A hosted gate whose `required_statistics` includes `kl` fails in preflight (`require_gate_capabilities`); no zero, empty, or independently generated KL is substituted. KL is self-hosted only.
- **Tier 2 stops on paired binary success.** The canary's stopping rule reads the paired evaluator `success` difference. Behavior signals are logged beside it and do not stop the canary.
- **Horizon-without-harm promotion is an exposure policy.** `promotion_policy="horizon_reached_without_harm"` promotes when the canary horizon passes without a harm stop. It bounds how many episodes a candidate serves; it is not evidence that the candidate is harmless.
- **Thresholds stay DRAFT.** No margin, α, horizon, window, canary fraction, or detector threshold below fills a slot in `EVAL_PROTOCOL.md`. Values in the example settings files are execution inputs.
- **Synthetic tests are software evidence only.** The CPU suites and `scripts/synthetic/run_connected_lifecycle.py` show that the modules compose. They are not a null check, an A/A floor, a harm label, power, or a detection result.

## Known limits of the hosted path

- **Mixed-provider pairs are refused.** `run_pair` requires one execution seed; a Qwen configuration carries `sampling.seed` and the GLM configuration sends none, so a Qwen-versus-GLM gate or canary pair is rejected before a world opens. Hosted lifecycle pairs are GLM against GLM (reference against a GLM fault). A paired episode's two worlds come from the reference runtime's session factory.
- **Hosted sampling is not seeded.** `run_seed` orders the stream only. Two GLM episodes on the same task are independent draws, so the A/A floor is the measured disagreement rate, not zero.
- **The 192-token execute cap is untested on GLM.** Reasoning is forced on and `max_tokens` is the whole generation budget. If the smoke repeatedly ends `finish_reason` `length` before visible workflow JSON, stop; a larger cap is a separately locked configuration.
- **Hosted faults not in the catalog.** Weight, quantization, LoRA, template, and batch-invariance faults are vLLM relaunch faults; `provider_fault_support` reports them unsupported for a hosted base and the benchmark lists them as not live-executable. There is no GLM `model.model_id` fault because no alternative GLM id has been chosen.
- **Production-only arrivals in the scheduled canary are counted, not executed.** With a canary fraction below 1, arrivals not assigned to the canary increment `production_only_arrivals`; the monitor stream after promotion is where production traffic is executed.
- **Final-split task metadata is closed.** `annotate_task_metadata.py` refuses `test_normal`, so the benchmark's `--task-metadata` slice labels for `test_normal` are an open gate.

## Configuration inputs

| File | Role |
|---|---|
| `configs/models/glm_5_3_spotify_capability.json` | Unchanged Spotify capability template (same agent stack as Sonnet). |
| `configs/models/glm_5_3_general_experimental.json` | Experimental general template: `prompt-general-auth-v1`, `plan-v1`, `plan_progress_v2`, no tool-access profile, 1024 plan tokens, 192 execute tokens. |
| `configs/faults/hosted_zai/*.v1.json` | Hosted fault catalog: reasoning disabled, token truncation (96/96), greedy sampling, step limit 8, prompt without API guidance, API docs for one app, and two declared no-op benign controls. The 14-entry `configs/faults/` catalog is unchanged. |

A full run configuration is never hand-edited. `build_run_configuration.py` combines a template with a local manifest, binds HEAD, refuses a dirty tree, runs the tool-access preflight against the manifest's `required_apps_by_task`, and prints the hashes.

## Runbook: first `train`/`dev` calibration batch

All commands run from the repo root with `PYTHONPATH=src` and the service interpreter `.venv-service/bin/python` (written `py` below). `APPWORLD_ROOT=data/raw/appworld` must be set. `ZAI_API_KEY` is read from the environment of the process that builds the runtime, including spawned A/A workers; it never goes on a command line or into a file. Use a fresh SQLite store per run. Every path under `data/processed/` is local and git-ignored.

### 0. Smoke (one task, one episode)

```bash
py scripts/evaluation/smoke_live_episode.py \
  --configuration configs/models/glm_5_3_spotify_capability.json \
  --task-set data/processed/spotify_capability_20.json \
  --task-index 0 \
  --store data/processed/glm_smoke_0.sqlite
```

Stop here if the episode ends at `finish_reason` `length` without visible workflow JSON. Only after a usable smoke, the unchanged 20-task pilot is `scripts/evaluation/run_capability_pilot.py` with the same configuration and task set, a fresh `--store`, and an `--output` under `data/processed/`.

### 1. Annotate the local manifests and build configurations

```bash
py scripts/data/annotate_task_metadata.py --manifest data/processed/train_gate.json
py scripts/data/annotate_task_metadata.py --manifest data/processed/dev_calibration.json

py scripts/evaluation/build_run_configuration.py \
  --template configs/models/glm_5_3_general_experimental.json \
  --task-set data/processed/train_gate.json --run-seed 17 \
  --output data/processed/configs/glm_general_train.json
py scripts/evaluation/build_run_configuration.py \
  --template configs/models/glm_5_3_general_experimental.json \
  --task-set data/processed/dev_calibration.json --run-seed 17 \
  --output data/processed/configs/glm_general_dev.json

py scripts/evaluation/build_run_configuration.py \
  --template configs/models/glm_5_3_general_experimental.json \
  --task-set data/processed/train_gate.json --run-seed 17 \
  --fault configs/faults/hosted_zai/glm_reasoning_disabled.v1.json \
  --output data/processed/configs/glm_general_train_reasoning_disabled.json
py scripts/evaluation/build_run_configuration.py \
  --template configs/models/glm_5_3_general_experimental.json \
  --task-set data/processed/dev_calibration.json --run-seed 17 \
  --fault configs/faults/hosted_zai/glm_reasoning_disabled.v1.json \
  --output data/processed/configs/glm_general_dev_reasoning_disabled.json
```

Build all four at one clean HEAD with one `--run-seed`. The train and dev files then differ only in their task-selection leaves (split, selection rule, seed, count, task-set hash), which is what step 8's train-to-dev admission requires; a rebuild at another commit or seed changes the hash and admission refuses it.

### 2. Plan specs for the Tier 1 plan-quality features

```bash
py scripts/data/plan_specs.py skeleton --task-set data/processed/train_gate.json \
  --output data/processed/plan_specs/train_gate.json
py scripts/data/plan_specs.py validate --specs data/processed/plan_specs/train_gate.json \
  --task-set data/processed/train_gate.json \
  --features requirement_coverage_fraction entity_coverage_fraction dependency_consistency_fraction
```

The skeleton fills `available_tools` only. Subgoals, entities, and dependency pairs are written by hand from split-available task metadata, never from evaluator output.

### 3. Hosted A/A (plan and execute; no GPU probe)

```bash
py scripts/evaluation/capture_aa.py \
  --configuration data/processed/configs/glm_general_dev.json \
  --task-set data/processed/dev_calibration.json \
  --stream-settings data/processed/streams/dev_aa_stream.json \
  --repetitions 3 --concurrency 2 --modes plan,execute \
  --output data/processed/aa/glm_general_dev_aa.json
```

Hardware is `not_applicable` and `nvidia-smi` is never called; elapsed wall time is recorded. `--observe-hardware` is refused for a hosted configuration.

### 3a. Resumable calibration

`scripts/evaluation/calibrate_hosted.py` is the bounded path for steps 3 and 4. `--prepare` inventories stores and captures, fills empty slots only from the same configuration hash, and writes a checkpoint plus a public report. Episodes from another hash stay inventory. A same-hash execute pair fills one A/A slot. Its reference is a healthy production draw and can support a baseline success estimate; the report counts how many baseline slots those references would add and does not assign the episode to both. `--run` resumes the checkpoint and stops after `--max-model-episodes` model episodes. The repetition count in the examples below is a provisional execution budget, not a power-based sample target: that target needs repeated production draws on at least two scenarios, aligned do-nothing outcomes, and a caller-supplied harm margin and alpha. `test_normal` is refused.

### 4. Production and do-nothing baseline

```bash
py scripts/evaluation/collect_baseline.py \
  --configuration data/processed/configs/glm_general_dev.json \
  --task-set data/processed/dev_calibration.json --repetitions 3 \
  --output data/processed/baselines/glm_general_dev.json
```

### 5. Dev fault characterization (one command per hosted fault)

```bash
py scripts/evaluation/characterize_harm.py --live-runtime \
  --base data/processed/configs/glm_general_dev.json \
  --task-set data/processed/dev_calibration.json \
  --fault configs/faults/hosted_zai/glm_reasoning_disabled.v1.json \
  --margin <DRAFT> --confidence-level <DRAFT> --resamples 10000 --seed 1 \
  --output data/processed/harm/glm_reasoning_disabled.json
```

`--candidate` defaults to `apply_fault(base, fault)`. A no-op fault is refused unless it declares `benign_control` `declared_noop`.

### 6. Method validation and power

```bash
py scripts/evaluation/assess_harm_study.py --spec data/processed/harm_spec.json \
  --baselines data/processed/baselines/glm_general_dev.json \
  --output data/processed/harm_study.json
py scripts/evaluation/simulate_power.py \
  --baselines data/processed/baselines/glm_general_dev.json \
  --harm-margin <DRAFT> --alpha <DRAFT> --sample-size 20 --sample-size 40 \
  --simulations 2000 --seed 7
py scripts/evaluation/validate_method.py --spec data/processed/methods/cusum.json \
  --aa-rows data/processed/aa/glm_general_dev_aa_rows.json \
  --output data/processed/validation/cusum.json
```

The empirical simulation resamples scenario clusters and draws reference and candidate from different repetitions of the same task, so its null rejection rate includes hosted nondeterminism. The normal-approximation power in `assess_harm_study.py` stays advisory.

### 7. Train gate with SQLite artifact

```bash
py scripts/run_offline_gate.py --live-runtime \
  --reference data/processed/configs/glm_general_train.json \
  --candidate data/processed/configs/glm_general_train_reasoning_disabled.json \
  --task-set data/processed/train_gate.json \
  --settings data/processed/gate_settings.json \
  --plan-evidence data/processed/plan_evidence_hosted.json \
  --episode-store data/processed/gate.sqlite
```

`plan_evidence_hosted.json` sets `required_statistics` to `["plan_quality", "mmd"]`. A provider failure is an execution error (exit 2), not a BLOCK.

### 8. Hosted service and train-to-dev admission

The gate ran on train configurations and the service serves dev ones, so the configuration hashes differ. `--task-selection-allowance` lets admission restore the train task-selection leaves on the dev configurations, rebuild both gate hashes, and compare them with the stored artifact. Write the allowance from the train reference the gate ran, never by hand:

```bash
py scripts/evaluation/build_task_selection_allowance.py \
  --train-config data/processed/configs/glm_general_train.json \
  --output data/processed/task_selection_allowance.json

py scripts/service/serve.py \
  --production-config data/processed/configs/glm_general_dev.json \
  --candidate-config data/processed/configs/glm_general_dev_reasoning_disabled.json \
  --task-selection-allowance data/processed/task_selection_allowance.json \
  --store data/processed/gate.sqlite --max-in-flight 2 --shutdown-timeout-seconds 30 \
  --canary-settings data/processed/canary_settings.json \
  --monitor-settings data/processed/monitor_settings.json \
  --frozen-reference data/processed/frozen_reference.json \
  --distributional-monitors data/processed/distributional_monitors.json \
  --task-metadata data/processed/dev_calibration.json \
  --dedup-seconds 0 --canary-assignment-seed 3
curl -s -X POST localhost:8000/candidates -H 'content-type: application/json' \
  -d '{"artifact_id": "<artifact_id printed by step 7>"}'
```

Required inputs:

| Input | Requirement |
|---|---|
| `--store` | The same SQLite file step 7 wrote with `--episode-store`; admission looks the artifact up there. |
| `--production-config`, `--candidate-config` | The dev builds from step 1. Both must have split `dev` and the same task binding. |
| `--task-selection-allowance` | A `task-selection-allowance-v1` document: exactly `version`, `allowed_leaves`, `train_task_set_hash`, `train_values`. It must name `task.split` (value `train`) and `task.task_set_hash`, and may name only `task.split`, `task.selection_rule`, `task.selection_seed`, `task.task_count`, and `task.task_set_hash`. Startup refuses unknown keys, repeated or unknown leaves, wrong value types, a missing candidate, and a non-dev serving split (exit 2). |
| `--frozen-reference`, `--monitor-settings` | `configuration_hash` and `reference_configuration_hash` equal the dev production hash printed by step 1. |
| `--canary-settings`, `--distributional-monitors` | Local DRAFT settings; not protocol values. |
| `--task-metadata` | The dev manifest from step 1, annotated; it supplies difficulty slices. |
| `ZAI_API_KEY` | In the server's environment. Hosted roles take no `--*-base-url`. |

Admission (`POST /candidates`, HTTP 409 on refusal) accepts only a stored `gate_run` PASS; `serve.py` admits in release mode, so `synthetic_fixture` evidence is refused with or without an allowance. After the train leaves are restored, every other hashed leaf must match the gate: model, agent, prompt, sampling, `run_seed`, `git_commit`, and `protocol_hash`. An allowance whose train values differ from the gate's task set fails the same hash check. Without `--task-selection-allowance` admission is strict, and a train gate cannot admit dev configurations. The service never builds a PASS.

`--dedup-seconds` now applies only to the alert sink, for alerts that have no period. The production monitor opens one aggregate incident per signal per monitoring period, and slices only attribute that incident. On its first feed, the service restores the open incidents from `--store`, so a restart does not raise an incident again.

Hosted roles take no `--*-base-url`. Endpoints route by configuration hash, so a promoted candidate keeps serving from its own route while the monitor compares it with the previous production reference.

### 9. Dev stream rehearsal

```bash
py scripts/evaluation/rehearse_dev_stream.py \
  --base data/processed/configs/glm_general_dev.json \
  --fault configs/faults/hosted_zai/glm_reasoning_disabled.v1.json \
  --task-set data/processed/dev_calibration.json \
  --schedule data/processed/schedules/dev_rehearsal.json \
  --monitor-settings data/processed/monitor_settings.json \
  --frozen-reference data/processed/frozen_reference.json \
  --distributional-monitors data/processed/distributional_monitors.json \
  --state data/processed/rehearsal/glm_reasoning_disabled.json
```

The rehearsal and the benchmark share `experiments/schedule.py`: the same hashed schedule, simulated clock, onset, and per-arrival persistence. Re-running the same command resumes and reaches the same decisions as an uninterrupted run. Healthy-prefix alarms are reported apart from post-onset detection. The connected lifecycle (gate, canary, monitor in order) and the detector comparison (`scripts/replay/replay_detectors.py` on one frozen observation stream) remain separate analyses.

### 10. Usage export

```bash
py scripts/evaluation/export_usage.py --store data/processed/gate.sqlite \
  --pricing data/processed/pricing/zai_v1.json --output data/processed/usage.json
```

Without `--pricing`, or with an unknown count in a priced field, cost is null. Z.AI cost charges `prompt_tokens − cached_tokens` at the input rate, `cached_tokens` at the cache-read rate, and `completion_tokens` once; a pricing file with a reasoning rate is refused (`docs/EVAL_PROTOCOL.md`, Method contracts).

### 11. Protocol commitment (after pre-registration only)

```bash
py scripts/evaluation/lock_protocol.py --settings data/processed/protocol_settings.json \
  --output data/processed/protocol.lock.json
py scripts/evaluation/lock_protocol.py --require data/processed/protocol.lock.json \
  --commitment data/processed/protocol_commitment.json
```

The lock stays local (plan evidence and task selections can name tasks). The commitment holds the digest, per-section digests, configuration and task-set hashes, fault versions, and validation methods.

### 12. Final benchmark and replay (closed)

Not run until `EVAL_PROTOCOL.md` is pre-registered and every method's null check has passed.

```bash
py scripts/benchmark/run_lifecycle_benchmark.py --live-runtime \
  --protocol data/processed/protocol.lock.json \
  --fault configs/faults/hosted_zai/glm_reasoning_disabled.v1.json \
  --train-tasks data/processed/train_gate.json \
  --test-normal-tasks data/processed/test_normal.json \
  --baselines data/processed/benchmark_baselines.json \
  --plan-evidence data/processed/plan_evidence_hosted.json \
  --schedule data/processed/schedules/benchmark.json \
  --checkpoint data/processed/benchmark/checkpoint.json \
  --export data/processed/benchmark/public.json
py scripts/replay/replay_detectors.py --observations <frozen observations> \
  --schedule <replay schedule> --factories <detector factories> --output <public output>
```
