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

## Current plan-only offline gate

`src/llm_behavior_ci/stats/` implements the stage 2 method set with standard-library execution paths and formula-level unit tests. `src/llm_behavior_ci/lifecycle/offline_gate.py` runs `run_pair` in plan mode on a caller-supplied `train` task set and applies three checks whose thresholds come only from `GateSettings`:

- a scenario-clustered paired bootstrap of plan-quality scores;
- truncated top-k plan KL when the tokenizers match and the top-k tables align;
- plan-level MMD with a permutation p-value.

`scripts/run_offline_gate.py` prints a public decision. Missing arguments and execution failures exit 2 and are not BLOCK decisions. The GitHub-hosted CPU workflow runs the unit, CI, and synthetic integration suites, then checks that a bare invocation exits 2. This path proves that the modules compose on CPU. It does not establish calibration, false-alarm control, coverage, reference agreement, an A/A floor, or a release threshold. It therefore satisfies neither the stage 2 nor stage 3 gate. The validation runner below does not fill those gaps.

## A/A capture command

`src/llm_behavior_ci/runtime/aa_capture.py` and `scripts/evaluation/capture_aa.py` pair one configuration with itself on a caller-supplied finite stream. Repetitions, concurrency, and modes are explicit arguments. Concurrency does not change task order, membership, or the recorded offsets. The command stores evaluator disagreement, requirement fractions, trajectory divergence, and plan-scoring inputs. Tool-count homogeneity reuses `chi_square_homogeneity` only when the counts meet that function's contract, and the p-value is not a decision. Plan-scoring inputs are the plan texts and top-k log probabilities; no KL limit is applied. A missing evaluator outcome stays missing. A zero requirement total leaves the fraction undefined, so the fraction difference stays unset. `test_normal` is rejected. `--observe-hardware` records MiB and wall time from `nvidia-smi` only; otherwise those fields are unset. A caller-supplied probe is not GPU evidence. This command fills none of the open slots above, including the batch-invariant A/A floor.

## Validation runner

`src/llm_behavior_ci/experiments/validation.py` and the `scripts/evaluation/` commands `validate_method.py`, `compare_plan_kl.py`, and `assess_harm_study.py` score supplied inputs. They fill none of the open slots above. They do not read `GateSettings`, do not choose α, a harm margin, a horizon, or a tolerance, and do not write under `results/`. `test_normal` and `test_challenge` are rejected.

`validate_method` takes explicit seeds, reference cases, and monitor observations. The report stores the seeds, method parameters, input hash, sample counts, and Wilson intervals. A catalog method is implemented before any report exists. It is validated only when a report's `benchmark_eligible` flag is true, which requires every minimum check for that method to have been requested and to have passed. CUSUM and ADWIN reports record the alarm rate and do not claim a nominal false-alarm bound. Fixed-window methods also record the single-look rate and the repeated-look rate. Reference agreement compares this package's statistic with the number in the case. SciPy, River, and confseq are reported as importable and are not called.

A/A dependence is accepted for provenance `local_runtime`, or for `gpu` when the caller also sets a hardware reading and a memory value. Synthetic provenance, an unread GPU, and missing labels leave the effect fields empty. Repeated-task effect is within-task variance over total variance. Scenario clustering is a one-way intraclass correlation. Inference variation is the evaluator disagreement rate on pair rows, or the within-task mean absolute deviation on a monitor series. `MonitorObservation` does not carry scenario, task, or repetition; those labels are a parallel `AAContext` or rows copied from a capture. Task and scenario ids stay out of the public summary.

`compare_plan_kl` keeps, at each position, the top-k production log-probabilities, restricts the candidate to those indexes, and records full mean KL minus truncated mean KL. The approximation name, k, and vocabulary size are stored. `gpu_floor_measured` stays false. A passed status on a supplied sample is not a vLLM truncation floor. `score_top_k` does not accept a vocabulary size; the comparison calls `truncated_next_token_kl`.

`assess_harm_study` is the stage 1 planner in `STAGES.md`. Its power calculation and its feasibility flag are not entries in the slots above. The harm margin, α, and sample sizes remain open until they are written in this file.

`STUDY_BUDGETS` caps how large a `cpu_fast`, `simulation`, or `gpu` call may be (`CONSTRAINTS.md`). The caps are not protocol parameters.

## Hosted GLM reference path (DRAFT)

`docs/HOSTED_LIFECYCLE.md` is the contract and runbook for developing the three tiers on Z.AI GLM-5.3. GLM-5.3 is a provisional hosted reference, not the Qwen3-4B production path and not a qualified baseline. Sonnet's 14/20 (`results/spotify_capability_20_sonnet_5_5.json`) is not a GLM baseline. On a hosted configuration the Tier 1 default is the plan-quality paired bootstrap plus MMD; teacher-forced plan KL is self-hosted only, and a hosted gate that requires `kl` fails in preflight. Tier 2 stops on paired binary evaluator success. `horizon_reached_without_harm` promotion is an exposure policy, not evidence of harmlessness. Every threshold stays DRAFT. Synthetic tests are software evidence only.

The benchmark and the dev rehearsal share one hashed arrival schedule (`experiments/schedule.py`, `benchmark-schedule-v1`): stream settings, healthy prefix, abrupt or ramped onset, analysis horizon, canary fraction and assignment seed, and simulated clock start. Monitors run on the simulated clock and apply `outcome_delay_seconds` on it. Healthy-prefix alarms are reported separately from post-onset delay. The schedule's values are not protocol slots until they are written in the table above.

Hosted usage is per-request sanitized accounting (`ProviderCall`): attempts, status, latency, and the token counts the provider reported, with unknown counts left unknown. Cost is computed only from a versioned pricing file. The full protocol lock stays local; its public commitment is a digest and per-section digests (`lock_protocol.py --require LOCK --commitment PATH`).

## Method contracts (DRAFT)

These are code contracts. None of them fills a slot above.

- **Tier 2 canary.** `paired_difference_cs` is the proposed primary rule. It is an anytime-valid confidence sequence on paired success differences, with an α-level null check (`validate_method`, claim `at_most`). `fixed_window` is a heuristic comparator only. It records α but never reads it and reports no p-value. Its exact rollback probability for an identical candidate is 11/32 at horizon 3 with success 0.5 (`tests/validity/test_fixed_window_null.py`). Reaching the horizon without harm promotes the candidate. That is an exposure policy, not evidence of non-inferiority.
- **Tier 3 CUSUM.** `StoppingRule.slack` is optional and applies only to CUSUM. It is recorded in the settings when set; when unset, it defaults to 0 and the recorded payload is unchanged. CUSUM's null claim is `record`. No slack value supplies an α guarantee. Report the null alarm rate at the monitoring horizon you intend to use.
- **Tier 1 plan features (`plan-features-v3`).** A plan line that begins `App: api` counts as a tool reference when that app and API are available. Required tools are the tools named in a spec's `dependency_pairs`. `required_tool_coverage_fraction` measures how many of them the plan names. A plan with no references has no invalid references, but it has not covered any required tools either. A quality contract that weights tool-reference terms is refused unless it also gives required-tool coverage a positive weight.
- **Tier 1 MMD preflight.** `run_offline_gate` refuses a design before any world opens when the expected smallest paired-permutation p-value, `(1 + B·2^(1−k)) / (B+1)` for k independent scenario clusters and B permutations, exceeds `mmd_alpha`.
- **Validation reports.** Null false-alarm control, interval coverage and alternative power are reported separately. Conservative discrete tests have the claim `at_most`, so they are not expected to reject exactly α of null samples. Power, when an `alternative_shift` is supplied, is `measured` and does not affect `benchmark_eligible`. Tolerances are never widened to pass an undercovering bootstrap.
- **Alerts.** Each signal opens one aggregate incident per explicit monitoring period. Slices only attribute an incident (`attributed_slices`); they never open their own. The incident key is the period, the configuration hash, the signal and the slice. The store deduplicates on that key across restarts, and the service restores the open incidents from the store. `LocalAlertSink.deliver` returns only newly inserted alerts. A promotion starts a new period. Under this policy, an alarm in the healthy prefix suppresses post-onset alerts for the same signal in the same period.
- **Usage cost.** For Z.AI, `prompt_tokens` already includes `cached_tokens`. Uncached input (prompt tokens minus cached tokens) is charged at the input rate, cached tokens once at the cache-read rate, and completion tokens once at the output rate. Reasoning tokens are part of the completion tokens and have no rate of their own. Anthropic input excludes cache reads and writes, so each of those is priced separately. A provider without declared usage semantics is refused.

## What each tier is allowed to use

The library implements the full set in `STAGES.md` stage 2. The plan-only gate, the canary controller, and the production monitor are CPU decision paths. Their thresholds come from the caller and are not the open slots above. The A/A capture records paired outcomes and plan-scoring inputs and does not apply the decision rules below. Each consumer uses a subset:

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

An LLM judge. The agent's own claim that it completed a task. A plan-quality metric by itself. A divergence with no evaluator-measured drop in task success. A threshold chosen on `test_normal`. A fault label revised after `test_normal`. A method that has not passed its null check. A synthetic-null false-alarm rate reported without the A/A and task-mix-shift streams. A `benchmark_eligible` flag on caller-supplied provenance. A normal-approximation power number (`simulate_power.py`'s empirical clustered simulation is the planning number; neither is a result until committed under `results/`). A hosted model's plan KL of any kind. A top-k versus full KL error on supplied arrays reported as a vLLM truncation floor.

## Run log

`results/` contains no result artifacts, only its README and empty stage-aligned directories. When runs start, every benchmark configuration is logged there with a script that regenerates every figure. Committed artifacts are limited to the public list in `DATA.md`: versions and configuration hashes, split names, selection rules and seeds, counts, task-set hashes, and aggregate statistics, intervals, detector outputs, cost and latency summaries, and figures. Resolved task IDs, per-task outcomes, and everything derived from AppWorld content stay local. A run that is not in that tree is not a result.
