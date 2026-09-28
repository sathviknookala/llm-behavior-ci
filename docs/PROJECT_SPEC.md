# Project specification

Living contract for what this repo is. Seeded 2026-09-26 from Sathvik's planning handoff; re-scoped by Sathvik on 2026-09-27 from an LLM classification service to a tool-using agent service (`DECISIONS.md` D1, D8). Record a change that passes the test below in `DECISIONS.md`. A change that fails the test is a scope change: stop and ask Sathvik, naming which non-negotiable it touches.

## What it is

A small tool-using agent service on one 24 GB GPU, with a release lifecycle around it: episode logging, an offline CI regression gate, canary evaluation with automatic rollback, continuous production monitors, and alerts. The project then measures whether that lifecycle does its job.

The agent is a Qwen model served by vLLM and driven by Hugging Face smolagents. The environment is AppWorld: it supplies the tasks, each task's simulated database state, the app APIs the agent calls, the state mutations those calls cause, and the programmatic evaluator that decides whether a task succeeded. This repo does not build its own simulated world or tool set. It owns the serving and evaluation lifecycle around AppWorld.

The traffic is a seeded stream of AppWorld tasks. The users are simulated, and any drift in the task mix is constructed and pre-registered, not natural. Say "seeded AppWorld task streams", "canary on simulated traffic", and "operator-side monitoring". The service owner has log-probabilities, configuration, and full trajectories. This is not a claim about auditing third-party APIs or agents, and it is not a claim about real users.

## Architecture and component responsibilities

```
seeded AppWorld task stream (simulated users)
 -> gateway (FastAPI): config resolution, canary pairing, rollback, logging
 -> agent runtime (smolagents) -> vLLM (production and, during a canary, candidate)
 -> AppWorld world per episode: task state, APIs, mutations, evaluator outcome
 -> episode logs -> offline gate / canary tests / production monitors -> one de-duplicated alert
 -> fault-injection benchmark (delay, miss, false alarms, cost)
```

| Component | Owner | Responsibility |
|---|---|---|
| Qwen model | upstream | The policy under test. The default is `DECISIONS.md` D7 |
| vLLM | upstream | Serves each configuration's model with an OpenAI-compatible API, top-k log-probabilities, teacher-forced prompt log-probabilities, and `/metrics` |
| smolagents | upstream | The agent loop: builds model calls, parses actions, and hands them to AppWorld for execution. It does not execute actions itself |
| AppWorld | upstream | Task instructions, per-task initial database state, the app APIs, state mutations, and the evaluator. It is the only source of task outcomes |
| Gateway | this repo | Accepts task requests, resolves the production or candidate configuration, starts one AppWorld world per episode, runs the agent, pairs canary episodes, executes rollback, and logs every step |
| Configuration registry | this repo | Versioned production and candidate configurations: model revision, quantization, serving flags, prompt, plan-trace format, agent-runtime settings. Each has a hash that travels in every log row |
| Task-stream harness | this repo | Emits AppWorld task IDs in seeded order at a pre-registered arrival rate, open-loop, with the pre-registered task-mix schedule |
| Episode log | this repo | Per step: model input and output, top-k log-probabilities, the action, AppWorld's execution output and any error, latency, timestamp. Per episode: configuration hash, task ID, seed, evaluator result |
| `stats` | this repo | Paired bootstrap, next-token KL, and MMD |
| `lifecycle.offline_gate` | this repo | Offline CI regression gate over plan traces |
| `lifecycle.canary` | this repo | Paired canary tests and the rollback decision |
| `lifecycle.monitoring` | this repo | Sequential and change-detection monitors on the production stream, and alert de-duplication |
| Benchmark and reporting | this repo | Fault injection, shared-stream method comparison, the end-to-end tier flow, and every artifact under `results/` |

Rule for the integration boundary: every environment mutation goes through AppWorld, so its evaluator sees the true final state. smolagents' local Python executor is never the place an action runs.

## Evaluation lifecycle

Three tiers. A candidate passes each one before it reaches the next.

1. **Tier 1: offline CI regression gate.** The permanent fixed task set drawn from `train`. Production and candidate receive the same task and tool context and each produces an explicit, observable plan trace without executing any tool. The gate compares them with paired plan-quality metrics, teacher-forced plan-trace KL, and plan-level two-sample tests such as MMD. It decides whether the candidate may enter canary.
2. **Tier 2: canary evaluation.** Production and candidate execute AppWorld tools in separate worlds initialized from the same task state, and their trajectories diverge naturally. The candidate's episode is the served result; production's episode is the shadow reference. AppWorld's evaluator scores both. Sequential tests on the paired outcomes roll the candidate back when the evidence of harmful degradation is sufficient.
3. **Tier 3: continuous production monitoring.** After promotion, the previous known-good production configuration is the reference. Monitors track the current production configuration's task success, final-state correctness, invalid or erroring tool calls, trajectory length, and tool selection over time, using the sequential and change-detection methods in `STAGES.md`.

In the final benchmark the tiers run in one flow: the Tier 1 gate on the fixed `train` tasks; a candidate that escapes the gate enters the Tier 2 canary on `test_normal`; a candidate the canary promotes enters Tier 3 production monitoring on `test_normal`.

A plan trace is the text the model emits when asked to plan. It is observable behavior, not a claim about the model's internal reasoning. Whether plan-only evidence predicts executed harm is one of the things the project measures.

### AppWorld split policy

Canonical. Every other doc defers to this list (`DECISIONS.md` D8).

- **`train`**: development-visible tasks, and the permanent fixed task set for Tier 1 offline CI regression testing.
- **`dev`**: calibration, execution-based fault harm characterization, power analysis, canary development, production-monitor development, and threshold tuning.
- **`test_normal`**: the frozen held-out final benchmark for the Tier 2 canary and Tier 3 production monitoring. No prompt, threshold, fault label, detector, model configuration, stopping rule, task-stream rule, or analysis procedure may change after observing it.
- **`test_challenge`**: unused unless separately pre-registered later.

Harmful and benign fault labels are measured by execution on `dev` and frozen before `test_normal` is run. Effect sizes observed on `test_normal` are reported, never used to relabel a fault. Slices of `test_normal` results use only the metadata AppWorld releases for the test split (the difficulty indicators) and the agent's own observed behavior, never ground-truth required apps or APIs (`DATA.md`).

## The question

> Can statistically valid CI gates, canaries, and production monitors catch real behavior regressions in a tool-using agent service quickly, without false alarms on healthy traffic, and which methods do it best at what cost?

Three parts, all of which stay answerable:

1. **Before deploy.** Which plan-only offline gates (paired bootstrap on plan-quality metrics, teacher-forced plan-trace KL, plan-level two-sample tests) block which harmful changes, at what fixed-task-set size, while passing no-op changes? How often does a gate decision agree with the harm measured by executing the tasks?
2. **After deploy.** For canaries and production monitors on AppWorld outcomes and agent behavior, how do fixed-window tests, classical change detectors (CUSUM, ADWIN), and anytime-valid methods (confidence sequences, e-detectors) trade detection delay against false alarms over long healthy traffic?
3. **Harmful versus benign.** Can the system tell harmful changes from benign ones (no-op redeploys, inference nondeterminism, trajectories that diverge without changing the outcome, pre-registered task-mix shifts), so that it alerts on lost task success rather than on any change?

"Cost" means three quantities, none of them measured yet: the monitor's compute, the GPU-hours a gate or canary needs, and, for canaries, how many episodes a bad candidate served before it was rolled back. Quote one only after it is written under `results/` and the file is named.

**Ground truth.** Task outcomes come from AppWorld's evaluator: its database-state unit tests decide task success and which requirements passed. Every fault is injected with a known start time, and its real harm is its measured drop in evaluator task success, measured by execution on `dev` and frozen before `test_normal` is run. Plan-quality metrics are computed programmatically against the task metadata AppWorld releases for `train`; they are gate signals, not ground truth for harm. An LLM judge is never ground truth, and neither is the agent's own claim that it completed a task.

## Five kinds of evidence

All five survive any allowed change:

1. **A running service.** The gateway of this repo in front of smolagents, vLLM, and AppWorld. It logs every episode step, versions configurations, pairs canary episodes, and rolls back.
2. **An offline regression gate in CI.** Compares production and candidate plan traces on the fixed `train` tasks and blocks the candidate from canary when it fails.
3. **Canary analysis.** Paired execution in separate AppWorld worlds, a sequential test on evaluator outcomes, and automatic rollback.
4. **Production monitors with alerts.** Evaluator outcomes (task success, final-state correctness) and behavior signals (tool-call errors, trajectory length, tool selection), against the previous known-good configuration.
5. **The fault-injection benchmark.** Detection delay, miss rate, and false alarms over long healthy traffic, per method.

## Non-negotiables

1. The central question stays answerable, all three parts.
2. All five kinds of evidence survive.
3. The statistics are implemented in this repo and validated. That includes the bootstrap, KL, MMD, CUSUM, ADWIN, confidence sequences, e-detectors, and the canary test. Every test's false-alarm rate or coverage is checked by simulation under its null, and on real A/A traffic (the same configuration deployed as both versions). River, confseq, and SciPy are reference checks only. A method whose false-alarm rate has not been checked cannot appear in the benchmark.
4. Harm is defined by each fault's measured drop in AppWorld evaluator task success against a margin fixed in advance. Outcomes come from AppWorld's evaluator.
5. `docs/EVAL_PROTOCOL.md` is committed in pre-registered form before any benchmark run. Thresholds, harm labels, and every other methodology choice are fixed on `train` and `dev` before `test_normal` is observed, under the split policy above. Every headline number has replicates and confidence intervals. Null and negative results are reported plainly, including a finding that plan-only gates miss executed harm, or that anytime-valid methods were slower and not worth it.
6. It runs on Sathvik's single GPU. No cloud compute, paid APIs, or paid CI runners unless he approves.
7. The headline is the measured statistical behavior of the lifecycle. Prometheus, Grafana, Docker, and CI are supporting pieces. SRE/DevOps platform work, agent-framework building, and raising the agent's AppWorld score are out of scope. Do not grow the infrastructure or the agent at the expense of the three parts of the question.
8. Sathvik must be able to explain every component. Python throughout: FastAPI, vLLM, Transformers, smolagents, AppWorld, Postgres or DuckDB over Parquet, Docker Compose, GitHub Actions. No custom CUDA or Triton kernels. No Kubernetes in the core. No custom simulated environment unless integration with AppWorld requires it.
9. Honest wording, as in the traffic paragraph above. Never "first". `PRIOR_WORK.md` lists the close prior work. Never report a number that was not measured.
10. No code, data, or results from outside employment. Prior deploy-gate experience informs the design and never appears as project data or numbers.

## Change test

Run this before adopting any change, whether it is an inferred shortcut, a new idea, a swapped tool, or a decision from Sathvik:

1. Is the central question still answerable, all three parts?
2. Do all five kinds of evidence survive?
3. Is every method in the benchmark still validated, and is harm still defined by AppWorld evaluator outcomes?
4. Does it still fit 24 GB?
5. Can Sathvik still explain it, and does the headline stay statistical ML rather than infrastructure or agent building?

If all five answers are yes, log the change in `DECISIONS.md` with the reason. If any answer is no, stop and ask.

**Additions** are welcome only after the core stages are done, and must not replace a core piece. Examples that stay allowed later: a second model family (the default service model stays primary); Project A's recommender as a second monitored service (the agent service stays primary); Kubernetes, for example a local kind cluster, for deployment (never as the headline); more detectors (the core set in `STAGES.md` stays).

**Fallbacks that weaken the evidence** are allowed only after the preferred path has been tried. Tell Sathvik in the next update. Examples: LoRA adapters on one base model for canaries, if two vLLM processes do not fit; a smaller stream horizon, with every method run on the same streams; KL from vLLM's top-k log-probabilities instead of full-vocabulary KL, with the truncation error measured against full KL on a sample.

## Ask first

Spending money; accepting a dataset or environment license, or gated-access terms (AppWorld included); connecting alerts to any real account; registering a self-hosted runner or making the repo public; opening upstream PRs; contacting anyone; submitting a paper; dropping or reordering a core stage; changing the research question. The full list with the CI notes is in `CONSTRAINTS.md`.

## Why it exists

What this project is for:

- One public system that runs the whole release lifecycle end to end, for a tool-using agent.
- Evidence of operating a service under traffic. The traffic here is simulated from AppWorld tasks.
- A public codebase that an ML-platform or backend reviewer would read as software engineering.

## Definition of done

- Stages 0–7 in `STAGES.md` are complete, and the non-negotiables above hold.
- The A/A noise floor, and the null checks for every method, are committed before the benchmark.
- Every fault's harm label was measured by execution on `dev` with AppWorld's evaluator, and frozen, before any gate or monitor saw the fault and before `test_normal` was run.
- The Gao et al. anchor matched, or its gap is documented (`PRIOR_WORK.md`, `EVAL_PROTOCOL.md`).
- `EVAL_PROTOCOL.md` was committed in pre-registered form, with all methodology frozen, before `test_normal` was first run.
- The final benchmark ran on `test_normal` under that frozen protocol: the shared-stream method comparison, and the end-to-end flow of the Tier 1 gate on `train`, the Tier 2 canary on `test_normal`, and Tier 3 monitoring on `test_normal`.
- Every headline number has confidence intervals, a baseline, and a commit in `RESUME_FACTS.md`.
- The README reproduces the headline figure from committed scripts, and `docker compose up` plus one command runs the service, one `train` or `dev` task through all three tiers, and a demo fault end to end.
- The public repo holds only the artifacts `DATA.md` allows; everything else stays local.
- A workshop-style draft exists. Submitting it requires asking first.
