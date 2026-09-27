# Evaluation protocol

**Status: DRAFT. Not pre-registered.** Written 2026-09-26 so the slots exist before any data does; re-slotted 2026-09-27 for the tool-using agent design (`DECISIONS.md` D1, D8). No benchmark stream runs against this file until every slot marked OPEN is filled, fault harm labels are measured by execution with AppWorld's evaluator, thresholds are set on a calibration pool that is not a benchmark pool, and this file is committed in that state.

A number in this file is a planned slot or a cited prior. It is not a result of this repo. Results, when they exist, live under `results/` with the script that regenerates them.

## Slots

| Slot | State | Value |
|---|---|---|
| Environment | planned | AppWorld. Exact package and data version OPEN. |
| Split roles | proposed | `train` for the gate's fixed task set (its ground-truth metadata is released); `dev` for the calibration pool; `test_normal` for the harm-measurement and benchmark pools, aggregate use only; `test_challenge` unused unless pre-registered. Not locked. Must respect AppWorld's restrictions (`DATA.md`). |
| Default model | planned | Qwen3-4B chat, Apache 2.0, thinking mode off. Exact revision OPEN. |
| Agent runtime | planned | smolagents. Version, action interface (`DECISIONS.md` D18), prompt, and step limit OPEN. |
| Plan-trace format | OPEN | The explicit plan emitted in plan mode, without tool execution. Fixed before the gate runs. |
| Plan-quality metrics | OPEN | Computed programmatically against AppWorld's released task metadata, for example the apps and APIs the reference solution requires. Not ground truth for harm. |
| Plan representation for MMD | OPEN | Text embeddings of the plan, a kernel on the planned API sequence, or both. |
| Plan-trace KL | planned | A frozen production plan trace, teacher-forced under both configurations with identical context; per-position `D_KL(production ‖ candidate)` in nats. Same tokenizer only. Whether the candidate's trace is also scored is OPEN. |
| Outcome metrics | planned | Task success is the evaluator's per-task `success`. Final-state correctness is the fraction of evaluator requirements passed. |
| Behavior signals | planned | Invalid or erroring tool calls, trajectory length, tool-selection distribution. Exact definitions OPEN. |
| Harm margin | OPEN | The drop in evaluator task success that classifies a fault as harmful. No number has been chosen. |
| Harm-measurement pool | OPEN | Tasks and repetitions used to measure each fault's harm by execution. |
| Fault catalog | planned | The eleven configuration diffs in `STAGES.md` stage 3. Harm labels are not measured. |
| Fault onsets | OPEN | Abrupt or ramped, and the start times. |
| Methods in the benchmark | planned | The stage 6 set in `STAGES.md`. Parameters below are OPEN. |
| Thresholds | OPEN | Tuned only on the calibration pool, then frozen here. Not tuned on benchmark streams. The `DEMO_*` thresholds in `src/offline_gate/offline.py` do not fill this slot. |
| α | OPEN | Nominal level for each test. The demo's value is an execution fixture, not a protocol choice. |
| Bootstrap draws | proposed | 10,000, carried as the rig style from the quantization study (`PRIOR_WORK.md`). Not locked. The CPU demo uses fewer draws only to keep its CI check small. |
| Resampling unit | proposed | The task, when a stream repeats tasks, so intervals do not treat repeated episodes of one task as independent. Not locked. |
| Replication floor | rule, unmeasured | Repeated baseline runs, reported beside every difference. The floor is not subtracted from a difference. The floor's value does not exist yet. |
| Gate task-set size | OPEN | Fixed tasks per gate run; power is reported against it. |
| Task stream | OPEN | Task sampling (with replacement across episodes), arrival rate, concurrency, and seeds. The harness is open-loop at the set rate and its schedule is deterministic given a seed. |
| Task-mix shift schedule | OPEN | Which app or difficulty mixes shift, when, and how fast. Constructed, not natural. |
| Healthy horizon | OPEN | The handoff's "30 simulated days" was an example for replayed traffic, not a lock. |
| Replicates | OPEN | Independent stream seeds and fault start times. |
| Outcome delay | OPEN | Evaluator outcomes exist when an episode ends; any additional simulated delay before monitors see them. |
| Canary fraction | OPEN | Share of tasks sent to paired canary execution. |
| Canary served-result rule | proposed | In a canary pair, the candidate's episode is the served result, so a bad candidate's cost is counted; production's episode is the comparison. Not locked. |
| Monitoring reference | OPEN | How the previous known-good configuration's reference behavior is estimated, for example from its calibration-pool or pre-promotion episodes. |
| Calibration vs benchmark split | rule | Disjoint task pools. Benchmark streams are run once per pre-registered configuration. |
| Top-k for KL truncation check | planned | k = 20, vLLM's default `max_logprobs`, and larger values. The truncation error is a measurement still to run. |
| Batch-invariant mode | OPEN | Stage 0 measures the A/A floor with and without `VLLM_BATCH_INVARIANT=1`, including the throughput cost. The benchmark records which mode was on. |
| Gao et al. match tolerance | OPEN | The anchor is their reported median power of 77.4% across modifications (arxiv.org/abs/2410.20247). The tolerance for "matched" is not chosen. |

## Current offline-gate demo

`src/offline_gate/` currently provides a deterministic, standard-library execution path over hardcoded healthy inputs. The inputs are generic score vectors, next-token distributions, and embeddings. They are not plan traces, and no AppWorld task is involved:

- a paired percentile bootstrap of `candidate - production`;
- per-position `D_KL(production || candidate)` in nats from normalized log-probabilities, with no epsilon flooring when candidate support is missing;
- a biased squared MMD with an RBF kernel and a Monte Carlo p-value from within-pair production/candidate swaps, preserving the paired design.

`src/offline_gate/offline.py` combines those three checks and `scripts/run_offline_gate.py` prints their evidence. This path proves that the modules compose and that GitHub-hosted CPU CI can execute them. It does not establish calibration, false-alarm control, coverage, reference agreement, an A/A floor, or a release threshold. It therefore satisfies neither the stage 2 nor stage 3 gate.

## What each lifecycle phase is allowed to use

The library implements the full set in `STAGES.md` stage 2. Each consumer uses a subset:

- **Phase 1, offline CI gate:** plan mode only, on the fixed task set, with identical task and tool context for both configurations. Paired bootstrap on plan-quality metrics; teacher-forced plan-trace KL when the tokenizer matches; MMD on plan representations. Fail when a confidence interval crosses the margin or a divergence test rejects. Passing allows canary entry.
- **Phase 2, canary:** paired execution in isolated AppWorld worlds from the same initial state. The sequential canary test; a confidence sequence on the paired success difference; a fixed-window baseline. Behavior signals as each episode ends; evaluator outcomes after the pre-registered outcome delay. Roll back on sufficient evidence of harmful degradation.
- **Phase 3, production monitors:** the previous known-good configuration as reference. Task success and final-state correctness via a confidence sequence, the Podkopaev–Ramdas test, CUSUM, ADWIN, and e-detectors; tool-call error rate; trajectory length; tool selection via chi-square and MMD; task mix as input drift.
- **Benchmark comparison, same streams:** hourly KS and chi-square, with and without correction; CUSUM; ADWIN; confidence sequences; e-detectors; Podkopaev–Ramdas.

## Metrics the benchmark reports

Per method, with confidence intervals over replicates:

- detection delay in episodes and in simulated hours;
- miss rate within the horizon;
- false alarms over the healthy horizon, including task-mix shifts and benign controls, and including the A/A streams;
- share of harmful faults caught against share of benign changes flagged;
- compute cost.

Also, for the gate: power against task-set size per fault, false-block rate on the benign controls, GPU-minutes per gate run, and agreement between gate decisions and execution-measured harm labels. For the canary: candidate episodes served before rollback, failed candidate episodes among them, and time to rollback.

## Reporting that travels with every headline

The task-stream schedule and arrival rate, the AppWorld version, the smolagents version, the model revision, whether batch-invariant mode was on, the commit, GPU-hours, and peak memory.

Task-success numbers carry bootstrap confidence intervals over tasks. Benchmark numbers carry confidence intervals over replicates.

Scope line that travels with the claim: seeded AppWorld task streams with simulated users, operator-side monitoring, one GPU.

## What is not ground truth

An LLM judge. The agent's own claim that it completed a task. A plan-quality metric by itself. A divergence with no evaluator-measured drop in task success. A threshold chosen on a benchmark stream. A method that has not passed its null check. A synthetic-null false-alarm rate reported without the A/A and task-mix-shift streams.

## Run log

`results/` contains no result artifacts, only its README and empty stage-aligned directories. When runs start, every benchmark configuration is logged there as raw CSV or JSON, with a script that regenerates every figure. Committed artifacts carry task IDs, outcomes, and statistics, never AppWorld's protected content in plain text (`DATA.md`). A run that is not in that tree is not a result.
