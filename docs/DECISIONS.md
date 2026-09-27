# Decisions

Current design choices and open gates. Update a decision in place when evidence resolves it. Git history is the previous state. A change is logged here only after it passes the five-question test in `PROJECT_SPEC.md`.

Nothing below was settled by a measurement in this repo. Locked items were fixed by the 2026-09-26 handoff. Open items are Sathvik's, or they wait on a stage that has not started.

No fallback in `PROJECT_SPEC.md` has been taken.

## D1 — Research question

**Status:** LOCKED (handoff 2026-09-26)

The three-part question in `PROJECT_SPEC.md`: which pre-deploy gates block which harmful changes; how fixed-window, CUSUM, ADWIN, and anytime-valid methods trade detection delay against false alarms; and whether harmful changes separate from benign ones. Cost is monitor compute, replay GPU-hours, and bad requests served before rollback.

**Why:** A result that names the best method at a stated cost is the headline. Dropping a part makes that headline unanswerable.

## D2 — Evidence set

**Status:** LOCKED (handoff 2026-09-26)

The five kinds in `PROJECT_SPEC.md`: the gateway service, the CI release gate, canary analysis with rollback, production monitors with alerts, and the fault-injection benchmark.

## D3 — Ground truth

**Status:** LOCKED (handoff 2026-09-26)

Harm is an accuracy drop on dataset labels against the margin in `EVAL_PROTOCOL.md`, measured before any gate or monitor sees the fault. An LLM judge is not ground truth.

The margin itself is OPEN. See `EVAL_PROTOCOL.md`.

## D4 — Where the statistics live

**Status:** LOCKED (handoff 2026-09-26)

Bootstrap, KL, MMD, CUSUM, ADWIN, confidence sequences, e-detectors, and the canary test are implemented in this repo. River, confseq, and SciPy are reference checks on shared inputs. A method that fails its null check is fixed or dropped before it enters the benchmark.

**Why:** confseq's last release noted in the handoff is v0.0.11 (January 2023). The implementation has to be explainable in an interview and valid under this project's null.

## D5 — Hardware and money

**Status:** LOCKED (handoff 2026-09-26)

One GPU, planned for 24 GB, including a canary that holds two versions. No cloud compute, paid API, paid judge, or paid CI runner unless Sathvik approves. Log GPU-hours per run anyway.

## D6 — Stack boundary

**Status:** LOCKED at the boundary (handoff 2026-09-26)

Python, FastAPI, vLLM, Transformers, Postgres or DuckDB over Parquet, Prometheus, Grafana optional, Docker Compose, GitHub Actions, pytest. No custom CUDA or Triton. No Kubernetes in the core.

**Still open inside the boundary:** Postgres versus DuckDB (D10); Grafana versus a static page (D11).

## D7 — Default model

**Status:** LOCKED as the family and size (handoff 2026-09-26). Revision OPEN.

Qwen3-4B chat, Apache 2.0, thinking mode off. The exact revision is recorded when the checkpoint is pinned. A second model family may be added only after the core stages, and this model stays primary.

## D8 — Tasks

**Status:** LOCKED as the two tasks (handoff 2026-09-26). Label sets OPEN.

T1, primary: arXiv abstract → primary category, JSON `{label, confidence}`. T2: HuffPost headline and short description → category, with a pre-registered merge of duplicate categories. A third task may be added later and does not replace these.

## D9 — Headline

**Status:** LOCKED (handoff 2026-09-26)

The headline is the measured statistical behavior of the lifecycle. The dashboard, Docker, and CI support that measurement.

## D10 — Log store

**Status:** OPEN

Postgres, or DuckDB over Parquet. Either passes the change test. Choose when stage 0 starts and update this entry with the reason. Do not run both as the system of record.

## D11 — Dashboard

**Status:** OPEN

Grafana, or a static page, showing vLLM latency beside behavior metrics. Either is supporting material. Choose when stage 5 starts.

## D12 — CI runner

**Status:** OPEN — Sathvik

Self-hosted runner under the rules in `CONSTRAINTS.md`, or a local `make gate` whose result is posted as a commit status. Ask before registering a runner.

## D13 — Alert channel

**Status:** OPEN — Sathvik

Alerts go only to a channel he sets up. None is connected.

## D14 — Two versions on one GPU

**Status:** PREFERRED PATH, not yet tried

Two vLLM processes with split `gpu_memory_utilization`. LoRA adapters on one base model are the fallback, and only after the two-process path has been tried. Tell Sathvik if the fallback is taken.

## D15 — Second monitored service

**Status:** OPEN — Sathvik. Not in the core.

Project A's recommender may be added after the core stages. The LLM service stays primary. Project A's handoff, as of 2026-09-26, does not yet include the served API this would need.

## D16 — Project B

**Status:** OPEN — Sathvik. Out of this repo.

Robot learning may run later as a third project. This repo does not implement it. The handoff records that this project replaces the HNSW vector-engine idea from the same planning session.

## D17 — Publish, name, paper, contact

**Status:** OPEN — Sathvik

Repo name, when the repo becomes public, upstream PRs, paper venue, and faculty outreach are his. A one-page summary may be prepared here. Do not contact anyone, open an upstream PR, or submit a paper without asking.

No code, data, or numbers from outside employment enter this repo.
