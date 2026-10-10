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

### 3b. Ledgered qualification batches

`scripts/evaluation/run_qualification_batch.py` (`experiments/qualification_batch.py`) runs two fixed designs under the same attempt ledger as §12:

- **`dev_four_arm`**: per dev task, in a seeded per-task order (`seeded_arm_orders`), one execute episode in a fresh world for each of four arms:
  - H1, the healthy reference;
  - H2, an independent healthy repeat;
  - C0, the declared no-op;
  - C1, the regression.

  Each candidate must equal its fault applied to the healthy configuration. Each `(task, arm)` pair is its own ledger scope, so H1, H2 and C0 stay distinct even when they share a hash.
- **`plan_aa`**: two healthy plan generations per train task, run as one plan pair the way the gate runs them.

**Caps and resumption.**
- `--prepare` binds the design and the caps to a new checkpoint and calls no provider.
- A cap may not exceed the design's attempt count, and the design's other mode gets 0.
- `--run` reconciles first, and never dispatches a scope already in the ledger, whatever its state.
- A raised runtime error marks its started attempt `failed` and stops the batch.
- `reconcile_attempts.py` works on this checkpoint.

**`--evidence` output.** It reads only the checkpoint:

| Output | Source |
|---|---|
| Per-arm outcomes | Missing outcomes stay `None` |
| Execution A/A | H1 as repetition 0, H2 as repetition 1, as `task_success` observations; C0 never enters |
| CUSUM scale rule | H1 and H2 |
| No-op label (H1, C0) and regression label (H1, C1) | `harm_label_from_outcomes`, the math `measure_harm` uses. Both share the H1 column; a label with any missing outcome is `unmeasurable` |
| Plan gate replay (`replay_plan_gate`) | The gate's own score, MMD vectors, clusters and seed, applied to the stored pairs |
| Plan A/A series | See below |
| `criteria` | The approved qualification criteria (`qualification-criteria-v1`, `experiments/qualification_criteria.py`; rules in `EVAL_PROTOCOL.md`) and the failure rule |

**Plan A/A series.**
- The bootstrap reads the gate's weighted score, mapped affinely from its weight-implied range onto [0, 1].
- MMD reads one series per coordinate of its representation, so it gets one A/A result per coordinate.
- Both reach `validate_method` as `plan_quality_score` observations with task, scenario and repetition labels.
- Plan A/A requires every quality and MMD feature to be a fraction.

**Provenance.** Every result records whether its runtime was live (`is_live_runtime`). An A/A series is `local_runtime` only when every episode in it was live; otherwise it is `synthetic`, which `validate_method` never accepts. Fake-runtime outcomes therefore cannot qualify a configuration.

**Lock reports.** Repeated `--lock-inputs` files (`{"spec", "null_seed_blocks", "reference_cases"}`) add one assembled `ValidationReport` per method and its status (`validated`, `failed`, `pending_live_aa`, `unavailable`):

| Batch | Method | A/A components |
|---|---|---|
| `dev_four_arm` | `sequential_canary`, `cusum` | `task_success` (H1, H2) |
| `plan_aa` | `clustered_paired_bootstrap` | `plan_quality` |
| `plan_aa` | `mmd_permutation_test` | one `mmd:<feature>` per coordinate |

- `assemble_validation_report` simulates each null seed block under the study budget, then runs the checks once over every replicate. The result equals one `validate_method` call on all the seeds.
- The A/A check passes only when every declared series has a component and every component passed. A single series binds its component unchanged.
- Components must share one provenance, split and configuration hash, and that hash must be the healthy arm's.
- MMD's coordinate results describe the joint test's inputs. They do not replace the joint test, which stays in the gate replay, and they make no overall α statement.
- Once the batch estimates the CUSUM scale, the CUSUM inputs must carry exactly that target, slack and threshold.
- `lock_protocol.py` reloads a report with its seed blocks and components.

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

On restart against the same `--store`, the service reads its last deployment decision back from SQLite before serving. A stored promotion must name the registered candidate as promoted over the registered production. The promoted candidate then serves again, compared against that production in the same monitoring period, so an open incident does not alert twice. A stored rollback keeps production and refuses the rolled-back candidate at `POST /candidates`. An admission with no promotion or rollback after it is closed with a persisted `restart_with_unfinished_canary` rollback, and candidate traffic does not resume. A stored decision whose hashes differ from the registered configurations stops startup. `GET /deployment` reports the recovered state with `recovered_from_store`.

### 8a. Connected three-tier dev run (documented, not run)

`scripts/demo/run_three_tier_dev.py live` replaces step 7 and the manual `curl` in step 8 with one command. Start `serve.py` exactly as in step 8, on a fresh `--store`. Then run:

```bash
py scripts/demo/run_three_tier_dev.py live \
  --gate-reference data/processed/configs/glm_general_train.json \
  --gate-candidate data/processed/configs/glm_general_train_reasoning_disabled.json \
  --task-set data/processed/train_gate.json \
  --gate-settings data/processed/gate_settings.json \
  --plan-evidence data/processed/plan_evidence_hosted.json \
  --production-config data/processed/configs/glm_general_dev.json \
  --candidate-config data/processed/configs/glm_general_dev_reasoning_disabled.json \
  --dev-task-set data/processed/dev_calibration.json \
  --canary-arrivals 10 --production-arrivals 10 \
  --service-url http://127.0.0.1:8000 \
  --store data/processed/three_tier_dev.sqlite \
  --release-summary data/processed/three_tier_dev_release.json
```

For this dev run, set `fraction` to `1.0` in the local canary settings file passed to `serve.py --canary-settings`. Set `--canary-arrivals` to at least that file's `stopping_rule.horizon_episodes`. Each arrival is one request that the service routes. At a fraction below `1.0`, some canary arrivals go to production, so 10 arrivals at a fraction of `0.1` do not guarantee 10 candidate pairs. A canary that runs out of arrivals before its horizon ends as `canary_incomplete`. This applies only to the local dev settings; it changes no production default or threshold.

The command has no model-specific logic. A vLLM gate takes `--reference-endpoint` and `--candidate-endpoint`, a hosted gate takes neither, and `serve.py` takes the matching `--*-base-url`. Before the gate runs, it refuses a dirty tree or a configuration not built at HEAD. It also refuses a `--store` that does not exist yet (it must be the store the running service opened) or that already holds a validation artifact, decision, or episode, an existing `--release-summary` file, a service whose registered hashes differ from the dev files, and a `kl` requirement on a hosted configuration. The gate artifact goes into `--store`. A BLOCK stops before admission. A PASS is admitted only by the service's release admission and allowance. The canary arrivals run until the controller promotes or rolls back. After a promotion, the production arrivals are Tier 3: the service sends them to the promoted configuration, and its monitor compares them against the previous production configuration. After a rollback, the same arrivals are reported as `fallback_verification` and must all go to the known-good configuration. If the canary arrivals run out before the controller decides, the command calls `POST /deployment/rollback`, which records a `manual_rollback` decision, not a stopping-rule decision. It then checks that the candidate is closed and the known-good configuration serves, and reports `canary_incomplete` with a `cleanup` entry. If that check fails, the command reports an execution failure. The JSON summary holds the gate result, the admission status, receipts by role and hash, the deployment state, and the stored decisions, alerts, and episode counts.

Exit codes: `0` promoted; `1` blocked, rolled back, or admission refused; `2` invalid invocation or input, including a broken release lineage, before any model call; `3` `canary_incomplete` after a confirmed cleanup rollback; `4` execution or infrastructure failure, or a release summary that could not be written, with no summary. The summary is printed for every completed status, and the public `release-summary-v1` record is written to `--release-summary` first.

Every episode costs provider spend or GPU time. The canary, monitor, and frozen-reference files are still local DRAFT settings. `synthetic --scenario blocked|healthy|regression` runs the same function on injected runtimes, with `serve.py`'s own dependency builder and test admission. It is an engineering test: it exits `0` when the scenario reaches its expected outcome (blocked, promoted, rolled back), and `1` when it does not.

### 8b. Consecutive releases and the release workflow (wired, not run)

**One store per release.** Each release runs against its own new SQLite store, and the stores of earlier releases stay on the trusted machine as audit history. Restart recovery (step 8) is unchanged: it serves one production/candidate pair per store, so a release never reuses an earlier store.

**Release summary.** `--release-summary` gets a `release-summary-v1` document with hashes, outcomes, and counts only:

- `configurations`: `production`, `candidate`, `serving` (what the service serves after the run: the candidate after a promotion, production otherwise), `known_good` (the production this release started from), and `frozen_reference` (the reference hash on the store's last lifecycle decision, null when nothing was admitted).
- `gate`: outcome, reason codes, artifact id, evidence source, and the train hashes.
- `deployment`: the last `admit`/`promote`/`rollback` decision and its method, the decision sequence, state, admission, and candidate episodes served.
- `monitoring`: period id, Tier 3 and fallback-verification arrival counts, and alert counts by signal.
- `lineage`: null for a first release.

It holds no task id, trace, or store path.

**Lineage.** `--previous-release` names release N's summary. Release N+1 is refused with exit `2`, before any model call or admission, unless its `--production-config` hashes to release N's `serving` hash. The service for N+1 is therefore provisioned with that configuration as production and a frozen reference for it (`serve.py` already requires the frozen reference hash to match production). Release N+1's `lineage` records the SHA-256 of release N's summary, its status, and its serving, known-good, and frozen-reference hashes, so the chain of known-good configurations survives across stores.

**One pinned commit.** `git_commit` is part of the configuration hash, and live preflight refuses a configuration not built at HEAD. Consecutive releases therefore run at one commit: release N+1's production file is release N's candidate file, unchanged. A release at a different commit is refused by name (`cross-commit rollout is not supported`). Rehashing a deployed configuration at a new commit would be a new configuration with no stored evidence behind it, so cross-commit rollout is an open limitation, not a supported path.

**Workflow.** `.github/workflows/release-lifecycle.yml` starts only from `workflow_dispatch`. Mode `synthetic` runs the `blocked`, `healthy`, and `regression` scenarios on a GitHub-hosted CPU runner; each job fails unless its scenario reaches its expected outcome. Mode `live` runs only when dispatched from `main` of this repository. It runs on a self-hosted runner labelled `llm-behavior-ci-release` in the `release-live` environment, and only when `confirm_live` repeats `release_id`. It calls `scripts/ci/run_live_release.sh`, which runs step 8a's command from a provisioned release directory and exits with the command's own code. Only `0` (promoted) passes the job. `1` (not promoted), `3` (canary incomplete), `2`, and `4` all fail it. The job log shows only the exit code and its meaning; the lifecycle summary and stderr stay in the release directory. No artifact is uploaded.

**Prerequisites before any live dispatch, none of them done:**

- Approve and register the self-hosted runner with label `llm-behavior-ci-release` (`CONSTRAINTS.md`; the repo's fork-PR exposure applies even though no PR trigger exists).
- Create the `release-live` environment with required reviewers and a `main`-only deployment branch rule.
- Set the environment variables `LIFECYCLE_RELEASE_ROOT`, `LIFECYCLE_SERVICE_URL`, `LIFECYCLE_PYTHON` (the `.venv-service` interpreter), and, for vLLM gates, `LIFECYCLE_REFERENCE_ENDPOINT` and `LIFECYCLE_CANDIDATE_ENDPOINT`.
- Provider keys and `APPWORLD_ROOT` live in the runner's own environment. The workflow reads no secrets.
- For each release, provision `$LIFECYCLE_RELEASE_ROOT/<release_id>/` with `gate_reference.json`, `gate_candidate.json`, `train_task_set.json`, `gate_settings.json`, `plan_evidence.json`, `production.json`, `candidate.json`, and `dev_task_set.json`, all built at the `main` commit being dispatched.
- Start `serve.py` (step 8) with that release's production and candidate configurations, canary, monitor, and frozen-reference settings, allowance, and `--store $LIFECYCLE_RELEASE_ROOT/<release_id>/episodes.sqlite`. The runner and the service must see the same file.
- Authorize the paid or GPU spend for the release.

### 8c. Test-only Tier 2 canary on real agents (documented, not run)

`scripts/demo/run_tier2_canary_test.py` runs the real canary, with real hosted agents and AppWorld dev worlds, without running Tier 1. It is an integration test of the downstream software, not a release. Its admission evidence is a PASS artifact labelled `synthetic_fixture`, which the harness builds itself; no gate produced it. Every summary says `test_only_admission: true` and `gate_executed: false`. A Tier 1 result from another run, such as a BLOCK, is not read, changed, or reinterpreted.

- **`preflight`** makes no provider call and starts no service.
  - It loads every input and checks provenance at HEAD.
  - It builds the service dependencies exactly as `serve` does, against a scratch store. That also checks that the provider key is set.
  - It checks that their identity equals the one computed from the intended files.
  - It checks that test admission accepts the fixture and that release admission (`authorize_gated_candidate`) refuses it.
- **`serve`** builds `serve.py`'s dependencies with `_build_dependencies` and switches only `admission_mode` to `test`.
  - It runs uvicorn on 127.0.0.1, on a free unprivileged port, with a `--store` that must not exist yet. The store is used on the main thread, as under `serve.py`.
  - Two routes exist on this app only. `GET /test-only/identity` reports the settings the running service loaded: both configuration hashes, the canary settings, the assignment seed, the task-selection allowance, the monitor reference, the admission mode, and their SHA-256. `GET /test-only/canary` reports how many paired outcomes the stopping rule received.
  - `serve.py` has no test mode or test routes, and the release workflow does not use this script.
- **`drive`** compares `GET /test-only/identity` with the identity computed from the intended files. On any difference it exits `2` before writing the fixture, calling admission, or running a model.
  - It then admits the fixture through `POST /candidates` and sends at most `--canary-arrivals` canary arrivals through `POST /episodes`, using `lifecycle.connected.admit_and_run_canary`, the same code step 8a uses. It sends no Tier 3 arrivals.
  - Pair ids stay in process for the stored-pair checks and never enter a summary.
  - The summary keeps three verdicts apart:
    - `admission_verification`: the fixture checks and whether it was admitted.
    - `canary_execution`: pairs executed and persisted, pairs with both evaluator outcomes, pairs eligible for the stopping rule, stopping-rule observations, horizon reached, the controller's decision and its method, the final state, and `statistical_integration`.
    - `integrity_checks`: registered hashes per role, distinct and persisted episodes, the controller's served count, every eligible pair reaching the stopping rule, the fixture-backed admit decision, and the final deployment state.
- **`statistical_integration`** is `complete` only when the stopping rule itself promoted or rolled back (`stopping_rule` or `stopping_rule_alarm`). A cleanup rollback (`manual_rollback`) leaves it `incomplete`, even when the integrity checks pass.
- **Missing outcomes:** a pair missing either evaluator outcome is counted but never reaches the stopping rule.
- **Separate worlds:** `run_pair` opens two worlds per pair and rejects a pair whose worlds are the same object or do not share an initial state. The world identity itself is not stored.
- **`drive` exit codes:**
  - `0`: integrity holds, pairs ran, and the stopping rule decided.
  - `3`: integrity holds and pairs ran, but statistical integration is incomplete.
  - `1`: an integrity check failed, admission was refused, or no pair ran.
  - `2`: invalid input or a service identity mismatch.
  - `4`: execution failure.
- A promoted test service keeps running with its store for Tier 3; stop it with Ctrl-C or SIGINT.

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

Every plan generation and execute episode is an attempt in the checkpoint's ledger (`experiments/attempts.py`). Its identity (scope, role, mode, configuration hash, task) is persisted as `reserved`, then `started`, before the call that can reach the provider; `completed` or `failed` (a recorded `runtime_error`) is written by the same atomic checkpoint replace that stores its result. The caps are recorded on first use and a resume must repeat them or omit both. After a crash, `reconcile_attempts.py` (or the resume itself) marks every open attempt `interrupted`: the slot stays consumed, `known_starts` counts only attempts whose start was persisted, and the scope is never dispatched again. An interrupted canary pair counts against the pair horizon; an interrupted monitor arrival is recorded and skipped; a gate with spent attempts and no decision ends `execution_failed`. A reservation that would exceed either cap is refused before anything is written. Resume with the original command and the same `--checkpoint`.

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
  --export data/processed/benchmark/public.json \
  --max-plan-generations 120 --max-executions 98
py scripts/benchmark/reconcile_attempts.py \
  --checkpoint data/processed/benchmark/checkpoint.json
py scripts/replay/replay_detectors.py --observations <frozen observations> \
  --schedule <replay schedule> --factories <detector factories> --output <public output>
```
