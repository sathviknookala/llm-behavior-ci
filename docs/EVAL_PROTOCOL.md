# Evaluation protocol

**Status: DRAFT. Not pre-registered.** Written 2026-09-26 so the slots exist before any data does; re-slotted 2026-09-27 for the tool-using agent design (`DECISIONS.md` D1, D8). `test_normal` is not run until every slot marked OPEN is filled, fault harm labels are measured by execution on `dev` and frozen, thresholds are tuned on `dev`, and this file is committed in that state.

A number in this file is a planned slot or a cited prior. It is not a result of this repo. Results, when they exist, live under `results/` with the script that regenerates them.

## Split-use contract

The canonical split policy is in `PROJECT_SPEC.md`. This table is how the protocol applies it.

| Split | Used for | Tuning allowed | Individual tasks inspected by humans |
|---|---|---|---|
| `train` | Development-visible tasks; the permanent fixed task set of the Tier 1 CI gate | Yes | Yes |
| `dev` | Calibration, execution-based harm labels, power analysis, canary and monitor development, threshold tuning | Yes | Yes |
| `test_normal` | The frozen final benchmark: Tier 2 canary and Tier 3 production monitoring | No; everything is frozen before it is first run | No |
| `test_challenge` | Unused unless separately pre-registered later | — | No |

## Frozen before `test_normal`

Every item below is fixed on `train` and `dev` and committed here before `test_normal` is first run, and none changes after: prompts; thresholds and α; fault harm labels; detectors and their parameters; model and serving configurations; stopping and rollback rules; task-stream rules, including selection, seeds, and the task-mix shift schedule; and the analysis procedure, including slices, resampling, and figures. Effect sizes observed on `test_normal` are reported, never used to relabel a fault or revise a choice.

## Slots

| Slot | State | Value |
|---|---|---|
| Environment | planned | AppWorld. Exact package and data version OPEN. |
| Split roles | locked | As in the split-use contract above (`DECISIONS.md` D8). |
| Default model | planned | Qwen3-4B chat, Apache 2.0, thinking mode off. Exact revision OPEN. |
| Agent runtime | planned | smolagents. Version, action interface (`DECISIONS.md` D18), prompt, and step limit OPEN. |
| Plan-trace format | OPEN | The explicit plan emitted in plan mode, without tool execution. Fixed before the gate runs. |
| Plan-quality metrics | OPEN | Computed programmatically against the task metadata AppWorld releases for `train`, for example the apps and APIs the reference solution requires. Not ground truth for harm. |
| Plan representation for MMD | OPEN | Text embeddings of the plan, a kernel on the planned API sequence, or both. |
| Plan-trace KL | planned | A frozen production plan trace, teacher-forced under both configurations with identical context; per-position `D_KL(production ‖ candidate)` in nats. Same tokenizer only. Whether the candidate's trace is also scored is OPEN. |
| Outcome metrics | planned | Task success is the evaluator's per-task `success`. Final-state correctness is the fraction of evaluator requirements passed. |
| Behavior signals | planned | Invalid or erroring tool calls, trajectory length, tool-selection distribution. Exact definitions OPEN. |
| Harm margin | OPEN | The drop in evaluator task success that classifies a fault as harmful. No number has been chosen. |
| Harm measurement | rule | By execution on `dev`, scored by AppWorld's evaluator; labels frozen before `test_normal`. Tasks and repetitions within `dev` OPEN. |
| Fault catalog | planned | The eleven configuration diffs in `STAGES.md` stage 3. Harm labels are not measured. |
| Fault onsets | OPEN | Abrupt or ramped, and the start times. |
| Methods in the benchmark | planned | The stage 6 set in `STAGES.md`. Parameters below are OPEN. |
| Thresholds | OPEN | Tuned only on `dev`, then frozen here. Never tuned on `test_normal`. The `DEMO_*` thresholds in `src/llm_behavior_ci/lifecycle/offline_gate.py` do not fill this slot. |
| α | OPEN | Nominal level for each test. The demo's value is an execution fixture, not a protocol choice. |
| Bootstrap draws | proposed | 10,000, carried as the rig style from the quantization study (`PRIOR_WORK.md`). Not locked. The CPU demo uses fewer draws only to keep its CI check small. |
| Resampling unit | rule | The AppWorld scenario, the group of task variants over which AppWorld defines scenario-level completion, whenever several task instances or repeated episodes share one scenario; the task only when it is its scenario's sole member in the stream. Intervals and null checks cluster at that unit. |
| Replication floor | rule, unmeasured | Repeated baseline runs, reported beside every difference. The floor is not subtracted from a difference. The floor's value does not exist yet. |
| Gate task set | rule; size OPEN | A fixed subset of `train`, chosen once by a committed deterministic selection rule and seed, and used permanently. Power is reported against its size. |
| Task stream | OPEN | Task sampling (with replacement across episodes), arrival rate, concurrency, and seeds. Developed on `dev`; `test_normal` streams come from the same frozen rule. The harness is open-loop at the set rate and its schedule is deterministic given a seed. |
| Task-mix shift schedule | OPEN | Which mixes shift, when, and how fast. Constructed, not natural. Defined only over metadata available for the split it runs on; on `test_normal`, that is the difficulty indicators. |
| Healthy horizon | OPEN | The handoff's "30 simulated days" was an example for replayed traffic, not a lock. |
| Replicates | OPEN | Independent stream seeds and fault start times. |
| Outcome delay | OPEN | Evaluator outcomes exist when an episode ends; any additional simulated delay before monitors see them. |
| Canary fraction | OPEN | Share of tasks sent to paired canary execution. |
| Canary served-result rule | locked | In a canary pair, the candidate's episode is the served result, so a bad candidate's cost is counted; production's episode is the shadow reference (`DECISIONS.md` D21). |
| Monitoring reference | OPEN | How the previous known-good configuration's reference behavior is estimated. The estimation rule is fixed on `dev`. |
| Calibration vs benchmark split | rule | Calibration on `dev`; the benchmark on `test_normal`. The benchmark is run once per pre-registered configuration. |
| Top-k for KL truncation check | planned | k = 20, vLLM's default `max_logprobs`, and larger values. The truncation error is a measurement still to run. |
| Batch-invariant mode | OPEN | Stage 0 measures the A/A floor with and without `VLLM_BATCH_INVARIANT=1`, including the throughput cost. The benchmark records which mode was on. |
| Gao et al. match tolerance | OPEN | The anchor is their reported median power of 77.4% across modifications (arxiv.org/abs/2410.20247). The tolerance for "matched" is not chosen. |

## Current offline-gate demo

`src/llm_behavior_ci/stats/` currently provides a deterministic, standard-library execution path over hardcoded healthy inputs. The inputs are generic score vectors, next-token distributions, and embeddings. They are not plan traces, and no AppWorld task is involved:

- a paired percentile bootstrap of `candidate - production`;
- per-position `D_KL(production || candidate)` in nats from normalized log-probabilities, with no epsilon flooring when candidate support is missing;
- a biased squared MMD with an RBF kernel and a Monte Carlo p-value from within-pair production/candidate swaps, preserving the paired design.

`src/llm_behavior_ci/lifecycle/offline_gate.py` combines those three checks and `scripts/run_offline_gate.py` prints their evidence. This path proves that the modules compose and that GitHub-hosted CPU CI can execute them. It does not establish calibration, false-alarm control, coverage, reference agreement, an A/A floor, or a release threshold. It therefore satisfies neither the stage 2 nor stage 3 gate.

## What each tier is allowed to use

The library implements the full set in `STAGES.md` stage 2. Each consumer uses a subset:

- **Tier 1, offline CI gate:** plan mode only, on the fixed `train` task set, with identical task and tool context for both configurations. Paired bootstrap on plan-quality metrics; teacher-forced plan-trace KL when the tokenizer matches; MMD on plan representations. Fail when a confidence interval crosses the margin or a divergence test rejects. Passing allows canary entry.
- **Tier 2, canary:** paired execution in isolated AppWorld worlds from the same initial state; the candidate's episode is served and production's is the shadow reference. The sequential canary test; a confidence sequence on the paired success difference; a fixed-window baseline. Behavior signals as each episode ends; evaluator outcomes after the pre-registered outcome delay. Roll back on sufficient evidence of harmful degradation. Developed and tuned on `dev`; benchmarked on `test_normal`.
- **Tier 3, production monitors:** the previous known-good configuration as reference. Task success and final-state correctness via a confidence sequence, the Podkopaev–Ramdas test, CUSUM, ADWIN, and e-detectors; tool-call error rate; trajectory length; tool selection via chi-square and MMD; task mix as input drift. Developed and tuned on `dev`; benchmarked on `test_normal`.
- **Benchmark comparison, same `test_normal` streams:** hourly KS and chi-square, with and without correction; CUSUM; ADWIN; confidence sequences; e-detectors; Podkopaev–Ramdas.

## Metrics the benchmark reports

Per method, with confidence intervals over replicates:

- detection delay in episodes and in simulated hours;
- miss rate within the horizon;
- false alarms over the healthy horizon, including task-mix shifts and benign controls, and including the A/A streams;
- share of harmful faults caught against share of benign changes flagged;
- compute cost.

Also, for the gate: power against task-set size per fault, false-block rate on the benign controls, GPU-minutes per gate run, and agreement between gate decisions on `train` and the harm labels measured on `dev`. For the canary: candidate episodes served before rollback, failed candidate episodes among them, and time to rollback. For each fault, its `test_normal` effect size beside its frozen `dev` label.

## Reporting that travels with every headline

The split name, the hash of the resolved local task set, scenario, task, and episode counts, the task-stream rule, seeds, and arrival rate, the AppWorld version, the smolagents version, the model revision and configuration hash, whether batch-invariant mode was on, the commit, GPU-hours, and peak memory.

Task-success numbers carry bootstrap confidence intervals clustered by scenario. Benchmark numbers carry confidence intervals over replicates.

Scope line that travels with the claim: seeded AppWorld task streams with simulated users, operator-side monitoring, one GPU.

## What is not ground truth

An LLM judge. The agent's own claim that it completed a task. A plan-quality metric by itself. A divergence with no evaluator-measured drop in task success. A threshold chosen on `test_normal`. A fault label revised after `test_normal`. A method that has not passed its null check. A synthetic-null false-alarm rate reported without the A/A and task-mix-shift streams.

## Run log

`results/` contains no result artifacts, only its README and empty stage-aligned directories. When runs start, every benchmark configuration is logged there with a script that regenerates every figure. Committed artifacts are limited to the public list in `DATA.md`: versions and configuration hashes, split names, selection rules and seeds, counts, task-set hashes, and aggregate statistics, intervals, detector outputs, cost and latency summaries, and figures. Resolved task IDs, per-task outcomes, and everything derived from AppWorld content stay local. A run that is not in that tree is not a result.
