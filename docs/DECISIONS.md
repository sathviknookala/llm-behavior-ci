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

**Current implementation note (2026-09-28):** This is the target boundary, not an installed-stack inventory. The CPU suite currently uses `unittest`; pytest is pinned but unused. `requirements.txt` does not yet include smolagents or Transformers. `service.py` `create_app` and `scripts/service/serve.py` exist; Docker Compose and the vLLM environment do not.

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

**Status:** LOCKED as smolagents for the model and action interfaces, with the episode loop in this repo (Sathvik 2026-09-27; amended 2026-09-29). Action interface LOCKED as `code` (Sathvik 2026-09-29).

`runtime/episode.py` `run_episode` owns the agent loop: it builds each turn's messages, enforces `agent.step_limit`, parses the model's action, executes it, feeds the observation back, and records every `ModelStep` and `ToolStep` as it happens. smolagents supplies the two interfaces that loop calls. `runtime/agent.py` `SmolagentsVLLMAgent` subclasses `smolagents.Model`, and each turn's model call goes through `generate` against the configuration's vLLM server. `build_appworld_executor` returns a `smolagents` `PythonExecutor`-shaped `AppWorldActionExecutor` or a `Tool`-shaped `AppWorldExecuteTool`, chosen by `config.agent.action_interface`. Neither `CodeAgent` nor `ToolCallingAgent` is constructed, and `smolagents.LocalPythonExecutor` never executes an action. Every mutation goes through `AppWorldSession.execute`.

The loop stays in this repo because the lifecycle depends on what it records: per-step persistence, termination reasons, plan-only mode, and teacher-forced log-probabilities. smolagents' `run` owns its own memory, prompt templates, output parsing, and `final_answer` handling, and would have to be re-instrumented to produce the same records. With `prompt-v2` and the locked `code` action interface, execute mode asks for one AppWorld-native `apis.<app>.<api>(...)` call per turn, parsed by `runtime/actions.py` `parse_model_output`; legacy `prompt-v1` / `prompt-no-api-guidance` still use the `CALL <app> <api>` text. That format is this repo's, not smolagents'.

The baseline action interface is `code` (`configs/models/qwen3_4b_production.json`). It matches how AppWorld tasks are designed to be solved and keeps AppWorld's large API catalog out of the context as structured tool schemas. This selects `AppWorldActionExecutor`; it does not adopt `CodeAgent`. Adopting `CodeAgent` (Python actions in AppWorld's shell, which is how AppWorld tasks are designed to be solved) or `ToolCallingAgent` (AppWorld APIs as tools, which puts many tool schemas in context) as the loop is a change to this decision, and needs a live AppWorld catalog and an installed smolagents to test. On CPU, smolagents is not installed, so `SmolagentsVLLMAgent` and the executors run on their plain-object fallback bases with the same control flow.

**Live check (2026-09-29).** `train_smoke` indices 0–2 each persisted an execute episode and reached `agent.step_limit` (40) without `apis.supervisor.complete_task`. Those runs returned before `evaluate()`. The public summary is in `docs/RUNTIME_CONTEXT.md`. The shell accepts keyword arguments only. A malformed model output is not executed; the episode records it and continues. A bare call leaves stdout empty, so AppWorld reports `Execution successful.` and drops the return value; `LiveAppWorldSession.execute` prints a single call expression before execution. Opening a world freezes time; `runtime/clock.py` keeps episode timestamps and latencies on the real clock.

**Live runtime integration closed (2026-09-29).** Execute mode calls `session.evaluate()` when the step limit is reached and stores that `EvaluatorOutcome` with `status="failed"` and `termination_reason="step_limit"`. Evaluator success does not complete the episode. `prompt-v2` tells the agent to obtain credentials through documented AppWorld and supervisor APIs, not to guess them, and to reuse values returned by earlier calls. One later `train_smoke` index-0 episode against the already-running Qwen3-4B server terminated `appworld_completed` with `status="completed"` and persisted a non-null evaluator outcome: success false, 1 of 8 requirements. Public counts are in `docs/RUNTIME_CONTEXT.md`. That closes the live runtime integration. Model task success is not part of the close. Baseline AppWorld capability is stage 1. A later failure to solve a task is an experimental outcome unless it exposes a runtime defect.

## D19 — Evaluation lifecycle

**Status:** LOCKED (Sathvik 2026-09-27)

Three tiers, in order: Tier 1, the offline CI regression gate on plan traces over the fixed `train` tasks, without tool execution; Tier 2, canary evaluation with paired execution in isolated AppWorld worlds and sequential rollback; Tier 3, continuous production monitoring against the previous known-good configuration. In the final benchmark a candidate that escapes the gate enters the canary on `test_normal`, and a promoted candidate enters monitoring on `test_normal`. Details in `PROJECT_SPEC.md` and `STAGES.md` stages 3–6.

## D20 — Code-execution boundary

**Status:** LOCKED as in-process AppWorld (Sathvik 2026-09-29)

Model-generated actions run on his machine, in AppWorld's in-process shell, which restricts destructive modules by default. The path is `run_episode` → `LiveAppWorldSession` → `AppWorld(task_id)` → `world.execute` → `world.evaluate`. AppWorld's Docker or `appworld serve` mode is not used: it adds RPC, lifecycle, networking, and state management without changing the statistical experiment. Revisit only if the in-process executor causes an actual security or reliability problem.

**Live package (2026-09-29).** `world.execute` returns a string. Failure text starts with `Execution failed.` and is a recoverable tool error; a raise is the closed-database case. `world.evaluate` returns a `TestTracker`. The adapter reads `pass_count` and `num_tests`. There is no `initial_state_identity` method. `str(task.api_docs)` does not fit in `max_model_len` 32768; `context()` renders one sorted `app.api:` line per API and drops response schemas. That render is what `api-docs-corrupt-v1` can still redact. It does not use the solution's `required_apis`.

## D21 — Canary served-result rule

**Status:** LOCKED (Sathvik 2026-09-27)

In each Tier 2 canary pair, the candidate's episode is the served result and production's episode is the shadow reference. Episodes served before rollback, and failures among them, are counted from the candidate side.

**Why:** A canary's cost is what a bad candidate served. If production's episode were served, that cost would always be zero.

`run_pair` currently calls the reference episode and then the candidate on one runtime, so a paired episode cannot use two base URLs. That call order is recorded and is not this served-result rule. `CanaryController` counts candidate episodes as served and can roll back from `CanarySettings`. `service.py` `create_app` is a FastAPI gateway; HTTP `POST /deployment/rollback` sets admission only and does not roll `CanaryController` back. That is not this served-result rule.
