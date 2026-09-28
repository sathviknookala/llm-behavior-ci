# Decisions

Current design choices and open gates. Update a decision in place when evidence resolves it. Git history is the previous state. A change is logged here only after it passes the five-question test in `PROJECT_SPEC.md`, or after Sathvik makes it as a scope change.

Nothing below was settled by a measurement in this repo. Items marked "handoff 2026-09-26" were fixed by the seed handoff. Items marked "Sathvik 2026-09-27" come from his re-scope to a tool-using agent service, which replaced the classification tasks; that change touched non-negotiables 3 and 4 and the change test, so it was his to make. Open items are Sathvik's, or they wait on a stage that has not started.

No fallback in `PROJECT_SPEC.md` has been taken.

## D1 — Research question

**Status:** LOCKED (handoff 2026-09-26; re-scoped by Sathvik 2026-09-27)

The three-part question in `PROJECT_SPEC.md`, for a tool-using agent service: which plan-only CI gates block which harmful changes, and how well they agree with executed harm; how fixed-window, CUSUM, ADWIN, and anytime-valid methods trade detection delay against false alarms in canaries and production monitors; and whether harmful changes separate from benign ones. Cost is monitor compute, gate and canary GPU-hours, and candidate episodes served before rollback.

**Why:** A result that names the best method at a stated cost is the headline. Dropping a part makes that headline unanswerable. The re-scope keeps the statistical question and moves it to agents, where a regression can change actions and final state, not only labels.

## D2 — Evidence set

**Status:** LOCKED (handoff 2026-09-26; reworded by Sathvik 2026-09-27)

The five kinds in `PROJECT_SPEC.md`: the gateway service over smolagents, vLLM, and AppWorld; the offline CI regression gate; canary analysis with paired execution and rollback; production monitors with alerts; and the fault-injection benchmark.

## D3 — Ground truth

**Status:** LOCKED (Sathvik 2026-09-27; replaces dataset labels)

Task outcomes come from AppWorld's evaluator: database-state unit tests that give per-task success and per-requirement passes and fails. Harm is a drop in evaluator task success against the margin in `EVAL_PROTOCOL.md`. Each fault's harmful or benign label is measured by execution on `dev` before any gate or monitor sees the fault, and frozen before `test_normal` is run; `test_normal` effect sizes are reported and never relabel a fault. An LLM judge is not ground truth, and neither is the agent's own completion claim. Plan-quality metrics are gate signals, not ground truth.

The margin itself is OPEN. See `EVAL_PROTOCOL.md`.

## D4 — Where the statistics live

**Status:** LOCKED (handoff 2026-09-26)

Bootstrap, KL (here, teacher-forced on plan traces), MMD, CUSUM, ADWIN, confidence sequences, e-detectors, and the canary test are implemented in this repo. River, confseq, and SciPy are reference checks on shared inputs. A method that fails its null check is fixed or dropped before it enters the benchmark.

**Why:** confseq's last release noted in the handoff is v0.0.11 (January 2023). The implementation has to be explainable in an interview and valid under this project's null.

**Current implementation note (2026-09-28):** `experiments/validation.py` compares a method with a caller-supplied reference number. It records whether SciPy, River, and confseq can be imported and does not call them. That is not yet the shared-input reference check in this decision. confseq remains uninstalled until system Boost is approved (`CONSTRAINTS.md`). No method's null behavior is committed.

## D5 — Hardware and money

**Status:** LOCKED (handoff 2026-09-26)

One GPU, planned for 24 GB, including a canary that holds two versions. No cloud compute, paid API, paid judge, or paid CI runner unless Sathvik approves. Log GPU-hours per run anyway.

## D6 — Stack boundary

**Status:** LOCKED at the boundary (handoff 2026-09-26; smolagents and AppWorld added by Sathvik 2026-09-27)

Python, FastAPI, vLLM, Transformers, smolagents, AppWorld, Postgres or DuckDB over Parquet, Prometheus, Grafana optional, Docker Compose, GitHub Actions, pytest. No custom CUDA or Triton. No Kubernetes in the core. No custom simulated environment or tool set unless AppWorld integration requires it.

**Still open inside the boundary:** Postgres versus DuckDB (D10); Grafana versus a static page (D11).

**Current implementation note (2026-09-27):** This is the target boundary, not an installed-stack inventory. The CPU suite currently uses `unittest`; pytest is pinned but unused. `requirements.txt` does not yet include smolagents or Transformers, and no gateway, Compose stack, or vLLM environment exists.

## D7 — Default model

**Status:** LOCKED as the family and size (handoff 2026-09-26; kept by Sathvik 2026-09-27). Revision OPEN.

Qwen3-4B chat, Apache 2.0, thinking mode off, served by vLLM. Plan traces are emitted as visible output, not through thinking mode. The exact revision is recorded when the checkpoint is pinned. A second model family may be added only after the core stages, and this model stays primary. If stage 1 shows its AppWorld success is too low for the harm margin to be detectable, the size is Sathvik's call (`RISKS.md`).

## D8 — Task environment

**Status:** LOCKED as AppWorld, with split roles (Sathvik 2026-09-27; replaces the arXiv and HuffPost classification tasks). Version OPEN.

AppWorld owns the tasks, per-task simulated database state, APIs, state mutations, and the evaluator. The repo owns the lifecycle around it. The former T1 and T2 tasks and their datasets are dropped, not deferred.

Split roles, canonical in `PROJECT_SPEC.md`: `train` holds development-visible tasks and the permanent fixed task set of the Tier 1 CI gate; `dev` is for calibration, execution-based harm labels, power analysis, canary and monitor development, and threshold tuning; `test_normal` is the frozen held-out final benchmark for the Tier 2 canary and Tier 3 monitoring, after which no methodology choice changes; `test_challenge` is unused unless separately pre-registered later.

**Why:** Only `train` and `dev` release the metadata a plan-quality metric needs and allow tuning under AppWorld's restrictions (`DATA.md`). Holding `test_normal` back keeps the headline an out-of-sample measurement.

## D9 — Headline

**Status:** LOCKED (handoff 2026-09-26)

The headline is the measured statistical behavior of the lifecycle: statistically validated CI, canary, and production regression detection for tool-using agents. The dashboard, Docker, CI, and the agent itself support that measurement. Raising the agent's AppWorld score is not a goal.

## D10 — Log store

**Status:** OPEN

Postgres, or DuckDB over Parquet. Either passes the change test. Choose when stage 0 starts and update this entry with the reason. Do not run both as the system of record. Episode logs hold AppWorld content and stay local (`DATA.md`).

`EpisodeStore` currently uses SQLite for the local CPU episode path. It is an interim implementation and does not resolve this decision.

## D11 — Dashboard

**Status:** OPEN

Grafana, or a static page, showing vLLM latency beside behavior metrics. Either is supporting material. Choose when stage 5 starts.

## D12 — CI runner

**Status:** OPEN — Sathvik

Self-hosted runner under the rules in `CONSTRAINTS.md`, or a local `make gate` whose result is posted as a commit status. Ask before registering a runner.

The current workflow is CPU-only on a GitHub-hosted runner. No Makefile, GPU job, or commit-status path exists.

## D13 — Alert channel

**Status:** OPEN — Sathvik

Alerts go only to a channel he sets up. None is connected.

## D14 — Two versions on one GPU

**Status:** PREFERRED PATH, not yet tried

Two vLLM processes with split `gpu_memory_utilization`, needed by the canary's paired execution. Agent prompts carry API documentation and multi-step histories, so the KV-cache budget per process is measured at the agent's real context length. LoRA adapters on one base model are the fallback, and only after the two-process path has been tried. Tell Sathvik if the fallback is taken.

## D15 — Second monitored service

**Status:** OPEN — Sathvik. Not in the core.

Project A's recommender may be added after the core stages. The agent service stays primary. Project A's handoff, as of 2026-09-26, does not yet include the served API this would need.

## D16 — Project B

**Status:** OPEN — Sathvik. Out of this repo.

Robot learning may run later as a third project. This repo does not implement it. The handoff records that this project replaces the HNSW vector-engine idea from the same planning session.

## D17 — Publish, name, paper, contact

**Status:** OPEN — Sathvik

Repo name, when the repo becomes public, upstream PRs, paper venue, and faculty outreach are his. A one-page summary may be prepared here. Do not contact anyone, open an upstream PR, or submit a paper without asking.

No code, data, or numbers from outside employment enter this repo. Before any publish step, confirm that nothing public contains AppWorld's protected content in plain text (`DATA.md`).

## D18 — Agent runtime

**Status:** LOCKED as smolagents (Sathvik 2026-09-27). Action interface OPEN.

smolagents runs the agent loop against the configuration's vLLM server. Two interfaces pass the change test: a `CodeAgent` whose Python actions run in AppWorld's execution shell, which is how AppWorld tasks are designed to be solved; or a `ToolCallingAgent` with AppWorld APIs exposed as tools, which puts many tool schemas in context. Choose when stage 1 starts and record the reason. Either way, every mutation goes through AppWorld, and smolagents' `LocalPythonExecutor` never executes an action.

The current `runtime/agent.py` supplies an injectable agent-loop protocol and a direct HTTP `VLLMAgent` adapter for CPU-testable episode composition. smolagents is not wired, so this adapter does not resolve the action-interface choice.

## D19 — Evaluation lifecycle

**Status:** LOCKED (Sathvik 2026-09-27)

Three tiers, in order: Tier 1, the offline CI regression gate on plan traces over the fixed `train` tasks, without tool execution; Tier 2, canary evaluation with paired execution in isolated AppWorld worlds and sequential rollback; Tier 3, continuous production monitoring against the previous known-good configuration. In the final benchmark a candidate that escapes the gate enters the canary on `test_normal`, and a promoted candidate enters monitoring on `test_normal`. Details in `PROJECT_SPEC.md` and `STAGES.md` stages 3–6.

## D20 — Code-execution boundary

**Status:** OPEN — Sathvik

Model-generated actions run on his machine. AppWorld's in-process shell restricts destructive modules by default; `appworld serve` can also run the environment in Docker. Choose before the first agent episode runs, and record which.

## D21 — Canary served-result rule

**Status:** LOCKED (Sathvik 2026-09-27)

In each Tier 2 canary pair, the candidate's episode is the served result and production's episode is the shadow reference. Episodes served before rollback, and failures among them, are counted from the candidate side.

**Why:** A canary's cost is what a bad candidate served. If production's episode were served, that cost would always be zero.

`run_pair` currently calls the reference episode and then the candidate. That call order is recorded and is not this served-result rule. `CanaryController` counts candidate episodes as served and can roll back from `CanarySettings`, but no gateway serves an episode.
