# Stages

Read before starting or closing a stage. Status of every stage: **not started**. A hardcoded CPU demo of the three offline-gate statistics exists; it has no plan traces, AppWorld integration, null validation, reference checks, protocol thresholds, or measurements and does not complete a stage. A stage is complete when its gate's evidence is committed, not when a process starts.

Suggested numeric defaults belong in `EVAL_PROTOCOL.md` and stay open until that file is pre-registered. Do not invent a threshold in a script.

Keep the order. Stages 0–2 produce the noise floor and the null checks. Stages 3, 4, and 5 build the three evaluation tiers in `PROJECT_SPEC.md`: the Tier 1 offline CI regression gate, the Tier 2 canary, and Tier 3 continuous production monitoring. The benchmark is stage 6 and does not start before the protocol is committed.

Splits follow the canonical policy in `PROJECT_SPEC.md`. Stages 0–5 use only `train` and `dev`. `test_normal` is first run in stage 6, after every methodology choice is frozen. `test_challenge` is unused.

| Stage | Splits | Status |
|---|---|---|
| 0 Service skeleton, AppWorld integration, task-stream harness, noise floor | `train`, `dev` | not started |
| 1 Split selection, baseline agent, task-mix drift | `train`, `dev` | not started |
| 2 Statistics library and validity checks | `train`, `dev` | not started |
| 3 Tier 1: offline CI regression gate | gate on `train`; harm labels on `dev` | not started |
| 4 Tier 2: canary evaluation and automatic rollback | `dev` | not started |
| 5 Tier 3: continuous production monitoring and alerts | `dev` | not started |
| 6 Fault-injection benchmark | gate on `train`; canary and monitors on `test_normal` | not started |
| 7 Write-up and resume facts | — | not started |

## Stage 0 — service skeleton, AppWorld integration, task-stream harness, and noise floor

Gateway (FastAPI):

- accepts a task request (AppWorld task ID and seed) and resolves the production or candidate configuration from the configuration registry;
- runs one smolagents episode against the configuration's vLLM server in a fresh AppWorld world for that task;
- logs, per step, the model input and output, top-k log-probabilities, the action, AppWorld's execution output and any error, latency, and timestamp; and, per episode, the task ID, seed, configuration hash, step count, and AppWorld evaluator result;
- can run two episodes of one task in isolated worlds from the same initial state (for canary pairing), and supports a one-command rollback.

Task-stream harness: emits AppWorld task IDs in seeded order at a pre-registered arrival rate, open-loop, deterministic in its schedule given a seed. Model outputs are not deterministic; the schedule is.

**The A/A noise floor.** On `train` and `dev` streams, run the same stream twice against the same configuration under different concurrency. Measure task-success disagreement, the difference in the fraction of evaluator requirements passed, trajectory divergence (first divergent step, length difference, tool-selection distribution), and the plan-trace KL floor. Measure with and without `VLLM_BATCH_INVARIANT=1`, and record the throughput cost of batch-invariant mode.

Gate: no dropped logs under the target load; the stream schedule is reproducible; every environment mutation goes through AppWorld, and the evaluator result of a logged episode is reproduced from AppWorld's stored outputs; the two worlds of a paired episode are shown to be isolated; the noise floor is measured and committed.

## Stage 1 — split selection, baseline agent, and task-mix drift

- **Environment.** AppWorld at a pinned version. The split roles are the canonical policy in `PROJECT_SPEC.md`. The Tier 1 gate's fixed `train` subset, and the `dev` and `test_normal` stream rules, are each defined by a deterministic selection rule and seed (`EVAL_PROTOCOL.md`). `test_normal` is selected by rule, not run.
- **Default agent.** A Qwen3-4B chat checkpoint (Apache 2.0), thinking mode off, served by vLLM and driven by smolagents with the action interface in `DECISIONS.md` D18. Record the exact revisions. The prompt and step limit are fixed.
- **Plan mode.** The same prompt, task, and tool context as the executing agent, instructed to emit an explicit plan trace in a fixed format without executing any tool.
- **Floors.** On `dev`: a do-nothing agent that completes the task immediately, scored by the evaluator. Production success, and the fraction of requirements passed, with bootstrap confidence intervals clustered by scenario, overall and by app and difficulty. Beating the do-nothing agent is a sanity check, not a result.
- **Task-mix curve.** On `dev`: success by app and difficulty with scenario-clustered bootstrap confidence intervals. It shows which task-mix shifts change aggregate success and which do not. Part 3 of the question depends on it. Any shift used later on `test_normal` is defined over the difficulty indicators only, since that split releases no app labels.

Gate: production success beats the do-nothing agent, and a committed power calculation shows the pre-registered harm margin is detectable at the planned sample sizes on `dev`. If it is not, stop and ask Sathvik; changing the model size or the task selection is his call. The selection rules, seeds, and hashes of the resolved local task sets are committed; the resolved task IDs stay local (`DATA.md`).

## Stage 2 — statistics library and its validity checks

Implement in this repo:

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

Validity checks, run in CI on CPU:

- simulated nulls confirm each method's false-alarm rate or coverage at its nominal level (fixed-window tests are expected to inflate over repeated looks; show that too);
- reference checks against River, confseq, and SciPy on shared inputs;
- on the real `train` and `dev` A/A streams from stage 0, every detector's false-alarm rate is measured, including the effect of scenario-level dependence (shared scenarios and repeated tasks) on the independence assumptions.

Truncation check: top-k plan-trace KL against full-vocabulary KL on a sample, at k = 20 (vLLM's default `max_logprobs`) and larger.

Gate: every method matches its reference, and its measured null behavior is committed. A method that fails its own null check is fixed or dropped before stage 3.

## Stage 3 — Tier 1: the offline CI regression gate

**Fault catalog.** Each entry is a configuration diff. Each is run by execution on `dev`, scored by AppWorld's evaluator, to measure its true drop in task success with a scenario-clustered confidence interval, and classified as harmful (the drop is at least the pre-registered margin) or benign before any gate or monitor sees it. The labels are frozen before `test_normal` is run.

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

Developed, tuned, and acceptance-tested on `dev`. The gateway sends a pre-registered fraction of tasks to the canary. Each canary task is initialized twice from the same AppWorld task state, in two isolated worlds. The candidate executes in one and its episode is the served result; production executes in the other as the shadow reference. Their tool trajectories diverge naturally. AppWorld's evaluator scores both. Two vLLM processes share the GPU with split `gpu_memory_utilization`. The fallback, only after that path has been tried, is LoRA adapters on one base model (`PROJECT_SPEC.md`).

The canary test compares the paired outcomes as episodes finish: task success and the fraction of requirements passed from the evaluator, with the outcome delay pre-registered; behavior signals (tool-call errors, trajectory length, tool selection) as each episode ends. Methods: the sequential canary test, a confidence sequence on the paired success difference, and a fixed-window baseline.

Rollback is automatic when a test shows sufficient evidence of harmful degradation, and it is logged with the evidence.

Chaos tests: kill the candidate mid-episode; return malformed actions; slow the candidate down. Production episodes must keep running and serving, the candidate failure must be recorded, and neither world may be touched by the other configuration.

Measure the candidate episodes served before rollback, how many of those failed, and the time to rollback.

Gate: on `dev`, for a planted harmful candidate, the whole sequence works end to end: gate pass, canary, reject, roll back, alert.

## Stage 5 — Tier 3: continuous production monitoring and alerts

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
- `RESUME_FACTS.md`, filled only from measured results.
- Upstream PRs are optional and require asking first. Candidates named in the handoff: an anytime-valid detector for River, or a sequential drift test for Evidently, if either lacks one. Its documentation, as checked 2026-09-26, shows none.
