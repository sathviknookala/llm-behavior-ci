# Stages

Read before starting or closing a stage. No stage is complete. Implementation has begun across stages 0–6, but a stage is complete only when its gate's evidence is committed. The current task stream, fake-driven episode path, paired runner, A/A capture command, statistics library, validation runner, versioned fault diffs, plan-only offline gate, canary controller, production monitor, SQLite store, public export, lifecycle benchmark harness, and shared-stream detector replay are reusable components; they are not stage-gate evidence.

The hosted GLM path (`docs/HOSTED_LIFECYCLE.md`) adds code for stages 1–6: a shared runtime factory, hosted plan mode, a provider-aware gate, service, harm, baseline, and benchmark CLIs, a hosted fault catalog, a hashed arrival schedule with a simulated clock, per-slice references, restart-durable alert incidents, empirical power, usage accounting, and a public protocol commitment. None of it has run against Z.AI, and none of it closes a gate. CPU tests of that code are software evidence only.

Suggested numeric defaults belong in `EVAL_PROTOCOL.md` and stay open until that file is pre-registered. Do not invent a threshold in a script.

Keep the order. Stages 0–2 produce the noise floor and the null checks. Stages 3, 4, and 5 build the three evaluation tiers in `PROJECT_SPEC.md`: the Tier 1 offline CI regression gate, the Tier 2 canary, and Tier 3 continuous production monitoring. The benchmark is stage 6 and does not start before `docs/EVAL_PROTOCOL.md` is pre-registered and committed.

Splits follow the canonical policy in `PROJECT_SPEC.md`. Stages 0–5 use only `train` and `dev`. `test_normal` is first run in stage 6, after every methodology choice is frozen. `test_challenge` is unused.

| Stage | Splits | Implementation status | Gate status |
|---|---|---|---|
| 0 Service skeleton, AppWorld integration, task-stream harness, noise floor | `train`, `dev` | paired CPU path, A/A capture, gateway module, and one live execute episode with a persisted evaluator outcome; no noise-floor result | unmet |
| 1 Split selection, baseline agent, task-mix drift | `train`, `dev` | selection, episode abstractions, and a harm-study planner; one smoke evaluator outcome is not a baseline | unmet |
| 2 Statistics library and validity checks | `train`, `dev` | library and validation runner; no committed null, reference, A/A, or truncation result | unmet |
| 3 Tier 1: offline CI regression gate | gate on `train`; harm labels on `dev` | plan-only gate and versioned fault diffs; no committed harm labels or protocol thresholds | unmet |
| 4 Tier 2: canary evaluation and automatic rollback | `dev` | controller on injected execute pairs; gateway module exists; no live rollback | unmet |
| 5 Tier 3: continuous production monitoring and alerts | `dev` | monitor against caller-supplied baselines; no webhook or live stream | unmet |
| 6 Fault-injection benchmark | gate on `train`; canary and monitors on `test_normal` | lifecycle harness and shared-stream replay on caller-supplied inputs; benchmark not started | unmet |
| 7 Write-up and resume facts | — | not started | unmet |

## Stage 0 — service skeleton, AppWorld integration, task-stream harness, and noise floor

Current code provides configuration and record schemas, deterministic task streams, a fake-driven episode path, paired execution on injected worlds (`run_pair` in `runtime/episode.py`), an A/A capture command (`scripts/evaluation/capture_aa.py`), lazy AppWorld and vLLM adapters, SQLite episode storage, aggregate export, a FastAPI gateway module (`service.py` `create_app`) and serve script (`scripts/service/serve.py`), and a synthetic CPU integration contract. CPU tests show a shared pair id, separate worlds, independent mutations, a preserved candidate failure, distinct run identities, and a schedule that does not change with concurrency. One live execute episode has been run against the existing Qwen3-4B server and persisted an AppWorld evaluator outcome (`docs/RUNTIME_CONTEXT.md`). That closes the live runtime integration. Model task success is not required for that close. Baseline AppWorld capability is stage 1. A later failure to solve a task is an experimental outcome unless it exposes a runtime defect. There is no load run or committed noise-floor result. The capture rejects `test_normal`. Memory and wall time stay unset unless the command is asked to read `nvidia-smi`. A probe or a synthetic test is not GPU evidence, and no GPU capture has been written under `results/`.

Gateway (FastAPI):

- accepts a task request (AppWorld task ID and seed) and resolves the production or candidate configuration from the configuration registry;
- runs one smolagents episode against the configuration's vLLM server in a fresh AppWorld world for that task;
- logs, per step, the model input and output, top-k log-probabilities, the action, AppWorld's execution output and any error, latency, and timestamp; and, per episode, the task ID, seed, configuration hash, step count, and AppWorld evaluator result;
- can run two episodes of one task in isolated worlds from the same initial state (for canary pairing), and supports a one-command rollback.

Task-stream harness: emits AppWorld task IDs in seeded order at a pre-registered arrival rate, open-loop, deterministic in its schedule given a seed. Model outputs are not deterministic; the schedule is.

**The A/A noise floor.** On `train` and `dev` streams, run the same stream twice against the same configuration under different concurrency. Measure task-success disagreement, the difference in the fraction of evaluator requirements passed, trajectory divergence (first divergent step, length difference, tool-selection distribution), and the plan-trace KL floor. Measure with and without `VLLM_BATCH_INVARIANT=1`, and record the throughput cost of batch-invariant mode.

Gate: no dropped logs under the target load; the stream schedule is reproducible; every environment mutation goes through AppWorld, and the evaluator result of a logged episode is reproduced from AppWorld's stored outputs; the two worlds of a paired episode are shown to be isolated; the noise floor is measured and committed.

## Stage 1 — split selection, baseline agent, and task-mix drift

`plan_progress_v1` (`runtime/workflow.py`) completed a 20-task Spotify capability pilot at 0/20. The public aggregate is `results/spotify_capability_20_workflow.json` (configuration hash `221418b6294249094a4a0e77961782d046d66c0e168d6e96a7b6c68dce5898c9`, commit `a5e5c08`): 20/20 episodes hit the execute turn cap, and `complete_task_termination_count` is 0. Replaying the local store `data/processed/spotify_capability_20_14b_plan_progress_v1.sqlite` through that controller counts 53 inconsistent-ledger rejections, 34 repeat-gate blocks, and a stall in all 20 episodes. No `complete_task` call executed. v1 remains an implemented but failed Stage-1 baseline candidate. The primary failure was workflow-state management: the model had to reproduce redundant ledger fields. `plan_progress_v2` (`configs/models/qwen3_14b_awq_spotify_capability_v2.json`) moves those derived fields into the controller. Its 20-task pilot is `results/spotify_capability_20_plan_progress_v2.json` (configuration hash `4f229b2bf2d1600c79acbdc2286a8f643fe188f94e442a3d71945161b486bc0a`, commit `1b9aa3b`): evaluator success 0/20, 11 `complete_task` terminations, and 9 execute-turn-cap stops. A trace audit found no material controller or runtime defect. The rendered API docs omitted stored argument constraints, and action tasks did not say the empty `complete_task` answer is `answer=None`. Five episodes also repeated an invalid action form the parser already rejected. `prompt-runtime-auth-v2` on `configs/models/qwen3_14b_awq_spotify_capability_v2_interface.json` is an interface correction on the frozen controller, not a new policy. Its 20-task pilot is `results/spotify_capability_20_plan_progress_v2_interface.json` (configuration hash `50324f4089cbf48c5cfd33790bf0e10cf44a2b9972b49b26ea668239fce5d06f`, commit `bab2676`): evaluator success 0/20, 17 `complete_task` terminations, 3 execute-turn-cap stops, and 0 parser errors. It is not a qualified baseline. Stage 1 has not succeeded.

`spotify_capability_20` remains the long-horizon stress set: the committed train tasks, the 20-turn execute cap, and the existing public aggregates. `spotify_capability_short_v1` still refuses a 20-task short benchmark. Requirement count is not a horizon: 15 easy Spotify train tasks have 2 evaluator tests, and 12 of those still paginate. The shape limit is at most one paged Spotify call and at most 10 non-auth Spotify calls. That native pool is 6 tasks in 2 scenarios (3 easy, 3 medium). `spotify_short_horizon_diagnostic_v1` keeps those 6 in task-id order at seed 17. It is a capability diagnostic, not a representative benchmark. The 32B run uses `configs/models/qwen3_32b_awq_spotify_short_horizon_diagnostic.json` (the prepared 32,768-token serving settings, `plan_progress_v2`, `prompt-runtime-auth-v2`, 20 execute turns). Do not widen the shape limits to force 20 tasks, and do not replace tasks from model outcomes.

Current code can load a catalog manifest, deterministically select and hash a task set, generate a seeded stream, and run a plan or execute episode through injected interfaces. `assess_harm_study` in `experiments/validation.py` compares supplied production and do-nothing evaluator outcomes with scenario-clustered bootstrap intervals, reports success by app and difficulty, and computes `harm_detection_power` at supplied sample sizes. The power formula is a one-sided normal approximation: the null paired-mean difference is 0, the alternative is `-harm_margin`, and the variance is an argument. Feasibility stays unset until both evaluator arms exist, and it is false when the power target fails or the production-minus-do-nothing interval does not lie above zero. The command rejects `test_normal` and `test_challenge`. No AppWorld version, resolved task set, model revision, prompt, action interface, baseline, task-mix curve, or power calculation is fixed. One smoke evaluator outcome is recorded in `docs/RUNTIME_CONTEXT.md`. It has not been supplied to this study, and it is not a baseline.

- **Environment.** AppWorld at a pinned version. The split roles are the canonical policy in `PROJECT_SPEC.md`. The Tier 1 gate's fixed `train` subset, and the `dev` and `test_normal` stream rules, are each defined by a deterministic selection rule and seed (`EVAL_PROTOCOL.md`). `test_normal` is selected by rule, not run.
- **Default agent.** A Qwen3-4B chat checkpoint (Apache 2.0), thinking mode off, served by vLLM and driven by smolagents with the action interface in `DECISIONS.md` D18. Record the exact revisions. The prompt and step limit are fixed.
- **Plan mode.** The same prompt, task, and tool context as the executing agent, instructed to emit an explicit plan trace in a fixed format without executing any tool.
- **Floors.** On `dev`: a do-nothing agent that completes the task immediately, scored by the evaluator. Production success, and the fraction of requirements passed, with bootstrap confidence intervals clustered by scenario, overall and by app and difficulty. Beating the do-nothing agent is a sanity check, not a result.
- **Task-mix curve.** On `dev`: success by app and difficulty with scenario-clustered bootstrap confidence intervals. It shows which task-mix shifts change aggregate success and which do not. Part 3 of the question depends on it. Any shift used later on `test_normal` is defined over the difficulty indicators only, since that split releases no app labels.

Gate: production success beats the do-nothing agent, and a committed power calculation shows the pre-registered harm margin is detectable at the planned sample sizes on `dev`. If it is not, stop and ask Sathvik; changing the model size or the task selection is his call. The selection rules, seeds, and hashes of the resolved local task sets are committed; the resolved task IDs stay local (`DATA.md`).

## Stage 2 — statistics library and its validity checks

The Python implementations and formula-level unit tests exist under `src/llm_behavior_ci/stats/` and `tests/unit/`. `experiments/validation.py` and `tests/validity/` run simulated-null, coverage, stopping, repeated-look, reference, A/A-dependence, and KL-truncation checks on caller-supplied inputs. `implemented_methods()` lists every catalog method as not validated. `benchmark_eligible` is true only when that run requested every minimum check and each one passed. CUSUM and ADWIN are recorded without a nominal false-alarm bound, matching their implementations. Level and at-most methods are judged against the spec's α and tolerance. The local suite is not in the GitHub workflow. No simulated-null rate, reference match, real A/A false-alarm rate, or truncation error is committed under `results/`.

Implemented in this repo:

- the paired bootstrap, resampling by AppWorld scenario when several task instances or repeated episodes share one scenario;
- teacher-forced plan-trace KL: full-vocabulary (Transformers) and top-k (vLLM prompt log-probabilities);
- a chi-square test on the tool-selection distribution;
- KS on trajectory length;
- MMD with a permutation null, on plan-trace and trajectory representations;
- a classifier two-sample test on the same representations;
- CUSUM;
- ADWIN;
- Howard et al. confidence sequences for Bernoulli task success and for bounded paired differences;
- e-detectors (Shin et al.);
- the Podkopaev–Ramdas harmful-shift test;
- a sequential canary test in the style of Lindon et al.

The stage gate still requires these CPU checks before a method is benchmark-eligible:

- simulated nulls confirm each method's false-alarm rate or coverage at its nominal level (fixed-window tests are expected to inflate over repeated looks; show that too);
- reference checks against River, confseq, and SciPy on shared inputs;
- on the real `train` and `dev` A/A streams from stage 0, every detector's false-alarm rate is measured, including the effect of scenario-level dependence (shared scenarios and repeated tasks) on the independence assumptions.

Truncation check: top-k plan-trace KL against full-vocabulary KL on a sample, at k = 20 (vLLM's default `max_logprobs`) and larger.

Gate: every method matches its reference, and its measured null behavior is committed. A method that fails its own null check is fixed or dropped before stage 3.

## Stage 3 — Tier 1: the offline CI regression gate

`lifecycle/offline_gate.py` runs `run_pair` in plan mode on a caller-supplied `train` task set. It applies clustered paired bootstrap, truncated plan KL, and plan MMD using `GateSettings`. A runtime failure raises `GateExecutionError` and is not a BLOCK. The script exits 2 when its arguments are missing or the run cannot decide. CPU CI checks that bare invocation. Thresholds come from the caller. There is no committed harm label, protocol threshold, or GPU path. It does not satisfy this stage.

**Fault catalog.** Each entry is a configuration diff. Each is run by execution on `dev`, scored by AppWorld's evaluator, to measure its true drop in task success with a scenario-clustered confidence interval, and classified as harmful (the drop is at least the pre-registered margin) or benign before any gate or monitor sees it. The labels are frozen before `test_normal` is run. `experiments/faults.py` can apply a diff from `configs/faults/` and record that label from execute-mode pairs on a caller-supplied `dev` set. The API-documentation and LoRA entries name schema leaves the configuration hash does not have yet, so they are not representable. Nothing under `results/` is a frozen label.

1. FP8 weights (vLLM FP8 quantization).
2. NVFP4 weights (made with llm-compressor, or a published checkpoint; log which).
3. Model downgrade to Qwen3-1.7B.
4. A prompt edit that removes the API-usage guidance.
5. A chat-template bug (thinking mode on, or the system prompt dropped).
6. Sampling temperature raised to 1.0.
7. `max_tokens` too small, which truncates actions.
8. The agent step limit reduced.
9. An API-documentation bug confined to one app (a fault confined to one slice).
10. A LoRA fine-tune on off-distribution data.
11. Benign controls: a no-op redeploy, toggling batch-invariant mode, a logging refactor, and a candidate identical to production.

**The gate.** On every PR that changes the model, prompt, agent runtime, or serving configuration, run the permanent fixed `train` task set through production and candidate in plan mode. Both receive the same task and tool context; neither executes a tool. Compute the paired bootstrap on plan-quality metrics, teacher-forced plan-trace KL (same tokenizer only), and MMD on plan representations. The gate fails when a confidence interval crosses the margin or a divergence test rejects. A candidate that passes may enter canary.

Report power against task-set size for each fault, the false-block rate on the benign controls, GPU-minutes per gate run, and how often the gate's decision on `train` agrees with each fault's harm label from `dev`.

**Anchor.** Run the Gao et al. MMD test on one model modification from their released benchmark and match their reported detection behavior within a tolerance written in `EVAL_PROTOCOL.md`, or document why it does not transfer. Their paper reports a median power of 77.4% across modifications. That figure is theirs (`PRIOR_WORK.md`), not a result of this repo.

Gate: end to end in GitHub Actions, or the local-runner path in `CONSTRAINTS.md`, a planted harmful change is blocked from canary and a no-op passes.

## Stage 4 — Tier 2: canary evaluation and automatic rollback

`stats/canary.py` contains the sequential statistical primitive and unit tests. `run_pair` runs the reference episode and then the candidate in separate worlds and keeps a failed candidate. `scripts/evaluation/capture_aa.py` replays a supplied stream of one configuration and records disagreement, requirement fractions, trajectory divergence, and plan-scoring inputs. It does not apply the canary stopping rule. `lifecycle/canary.py` starts after a PASS gate, observes execute-mode pairs, and can continue, roll back, or promote from `CanarySettings`. Candidate episodes are counted as served. `service.py` `create_app` is a FastAPI gateway; `scripts/service/serve.py` with missing required arguments exits 2 and refuses a store path under a directory named `results`. HTTP `POST /deployment/rollback` sets admission only; it does not roll `CanaryController` back. `run_pair` still runs the reference episode and then the candidate on one runtime, so a paired episode cannot use two base URLs. That is not the served-result rule in `DECISIONS.md` D21. There is no traffic split, alert channel, or chaos test.

Developed, tuned, and acceptance-tested on `dev`. The gateway sends a pre-registered fraction of tasks to the canary. Each canary task is initialized twice from the same AppWorld task state, in two isolated worlds. The candidate executes in one and its episode is the served result; production executes in the other as the shadow reference. Their tool trajectories diverge naturally. AppWorld's evaluator scores both. Two vLLM processes share the GPU with split `gpu_memory_utilization`. The fallback, only after that path has been tried, is LoRA adapters on one base model (`PROJECT_SPEC.md`).

The canary test compares the paired outcomes as episodes finish: task success and the fraction of requirements passed from the evaluator, with the outcome delay pre-registered; behavior signals (tool-call errors, trajectory length, tool selection) as each episode ends. Methods: the sequential canary test, a confidence sequence on the paired success difference, and a fixed-window baseline.

Rollback is automatic when a test shows sufficient evidence of harmful degradation, and it is logged with the evidence.

Chaos tests: kill the candidate mid-episode; return malformed actions; slow the candidate down. Production episodes must keep running and serving, the candidate failure must be recorded, and neither world may be touched by the other configuration.

Measure the candidate episodes served before rollback, how many of those failed, and the time to rollback.

Gate: on `dev`, for a planted harmful candidate, the whole sequence works end to end: gate pass, canary, reject, roll back, alert.

## Stage 5 — Tier 3: continuous production monitoring and alerts

`lifecycle/monitoring.py` normalizes episodes and runs the configured detectors against caller-supplied frozen baselines. Alerts for one signal are de-duplicated in process. `tool_selection` and `task_mix` are normalized and are not monitored series. There is no webhook, dashboard, or live production stream. Offline shared-stream detector replay lives in stage 6; it is not a live monitor.

Developed, tuned, and acceptance-tested on `dev`. After promotion, the previous known-good production configuration is the reference. Monitors on the production stream:

- task success from the evaluator (a confidence sequence, the Podkopaev–Ramdas harmful-shift test, CUSUM, ADWIN, e-detectors);
- final-state correctness (the fraction of evaluator requirements passed);
- the rate of invalid or erroring tool calls;
- trajectory length;
- tool-selection behavior (the distribution over apps and APIs; chi-square and MMD, windowed and sequential);
- the task mix itself, so a mix shift is attributed as input drift rather than as a regression.

Alerts go to a webhook that Sathvik configures. Each alert carries which monitor fired, the evidence (statistic, threshold, window), the slice that moved most (on `test_normal`, only the difficulty indicators or the apps the agent itself called, never ground-truth app labels), the deploy and configuration hash, and a link to the runbook section. Alerts are de-duplicated and rate-limited. Do not connect an alert to an account until he names the channel (`DECISIONS.md`).

A dashboard (Grafana or a static page) shows vLLM latency metrics beside the behavior metrics.

Gate: every monitor runs live on a simulated `dev` stream, and a planted fault produces exactly one de-duplicated alert that names the right slice.

## Stage 6 — the fault-injection benchmark

`experiments/benchmark.py` `run_lifecycle_benchmark` is a local harness on caller-supplied inputs. For each caller-supplied fault and each seed in a caller-supplied `ProtocolLock`, it runs the real plan-mode offline gate on a `train` task set. A gate outcome other than PASS leaves canary and monitor `not_reached` with reason `gate_block`. After PASS it runs `CanaryController` on a `test_normal` stream in execute mode. Rollback leaves the monitor `not_reached` with reason `canary_rollback` (canary status `completed`). Promotion (canary status `promoted`) is required before `ProductionMonitor`. A non-representable fault records `not_representable` on all three tiers and does not run them. A canary stream that ends without rollback or promote raises `BenchmarkError`. `admit_test_normal` rejects a configuration that is not a lock template, so the harness binds and admits the locked `test_normal` template and then `apply_fault`; the faulted candidate is not admitted. `CanaryController.start` receives a PASS view whose hashes are the admitted test_normal configuration hashes, because the train gate decision carries train hashes. Checkpoints are caller-supplied and refused when a parent directory is named `results`. Resume reloads stored canary pairs through `restore_pair_execution` in `runtime/episode.py` and skips stored stream indexes. Interrupted runs do not call `export_public_results`. A completed run exports only when `export_path` is set, and that path is also refused under a `results` directory. `BenchmarkResult.gpu_memory_mib` and `gpu_hours` stay None; `compute_seconds` is elapsed wall time from `time.perf_counter`, not GPU-hours. Harmful versus benign comes from the lock's harm label for that fault version. `scripts/benchmark/run_lifecycle_benchmark.py` with no arguments exits 2; the GitHub workflow does not check it. `experiments/replay.py` `replay_detectors` replays one frozen `MonitorObservation` sequence through fresh detectors. It does not run an agent and uses no RNG. `agent_execution_seconds` is None. It hashes the stream before any detector runs. Outcome delay applies to `task_success` and `requirement_fraction`; other signals apply immediately. Observations still held at the end stay withheld. `monitoring_detector_factories` builds cusum, adwin, e_detector, and bounded_mean_cs by duplicating the parameter mapping in `lifecycle/monitoring.py`, plus hourly KS and chi-square with and without Bonferroni correction on the caller's `window_episodes`, plus `harmful_shift`. `scripts/replay/replay_detectors.py` with no arguments exits 2 and refuses paths under a directory named `results`; the GitHub workflow does not check it. CPU tests on injected inputs are not a `test_normal` benchmark run. This is not a stage-6 result. The gate below stays unmet.

This is the headline: the frozen final benchmark on `test_normal`. It runs only after `EVAL_PROTOCOL.md` is pre-registered and committed, with every item in its "Frozen before `test_normal`" section fixed. Nothing is changed after `test_normal` is observed.

Pre-register: the fault catalog with its frozen `dev` harm labels; each fault's onset, abrupt or ramped (the faulty configuration's share of traffic grows over time); the selection rules and seeds; the task-mix shift schedule; the healthy horizon; replicates (seeds and onsets); α and thresholds, set on `dev`.

Two parts, both on the same frozen protocol:

1. **Shared-stream method comparison.** Every method runs on the same `test_normal` streams.
2. **End-to-end lifecycle flow.** Each candidate goes through the Tier 1 gate on the fixed `train` tasks; a candidate that escapes the gate enters the Tier 2 canary on `test_normal`; a candidate the canary promotes enters Tier 3 production monitoring on `test_normal`.

Methods compared on the same streams:

- hourly KS and chi-square, with and without correction;
- CUSUM;
- ADWIN;
- confidence sequences;
- e-detectors;
- the Podkopaev–Ramdas test.

Metrics per method: detection delay in episodes and simulated hours; the miss rate within the horizon; false alarms per healthy horizon, including across task-mix shifts and benign controls; the share of harmful faults caught against the share of benign changes flagged; compute cost. For the gate and the canary, the metrics in stages 3 and 4. For the flow, which faults each tier stopped and what escaped to the next. Each fault's `test_normal` effect size is reported beside its frozen `dev` label and never relabels it.

Report with confidence intervals over replicates, clustered by scenario where episodes share one, and state which method you would ship at which traffic level.

## Stage 7 — write-up and resume facts

- A README with the headline figure (detection delay against false alarms per method) and exact commands.
- A workshop-style draft. Ask before submitting.
- Headline numbers, taken only from measured results under `results/`.
- Upstream PRs are optional and require asking first. Candidates named in the handoff: an anytime-valid detector for River, or a sequential drift test for Evidently, if either lacks one. Its documentation, as checked 2026-09-26, shows none.
