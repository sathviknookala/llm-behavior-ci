# Project specification

Living contract for what this repo is. Seeded 2026-09-26 from `project-c.md`. Record a change that passes the test below in `DECISIONS.md`. A change that fails the test is a scope change: stop and ask Sathvik, naming which non-negotiable it touches.

## What it is

A small LLM service on one 24 GB GPU, with a release lifecycle around it: request logging, a pre-deploy release gate, canary rollout with automatic rollback, production monitors, and alerts. The project then measures whether that lifecycle does its job.

The traffic is a replayed public dataset with real timestamps. Natural drift is real. The users are simulated. Say "replayed traffic from [dataset]", "canary on replayed traffic", and "operator-side monitoring". The service owner has log-probabilities and configuration. This is not a claim about auditing third-party APIs, and it is not a claim about real users.

## The question

> Can statistically valid release gates and monitors catch real behavior regressions in an LLM service quickly, without false alarms on healthy traffic, and which methods do it best at what cost?

Three parts, all of which stay answerable:

1. **Before deploy.** Which offline gates (paired bootstrap on task metrics, token-level KL, output two-sample tests) block which harmful changes, at what replay-set size, while passing no-op changes?
2. **After deploy.** For canaries and production monitors, how do fixed-window tests, classical change detectors (CUSUM, ADWIN), and anytime-valid methods (confidence sequences, e-detectors) trade detection delay against false alarms over long healthy traffic?
3. **Harmful versus benign.** Can the system tell harmful changes from benign ones (natural topic drift, no-op redeploys, inference nondeterminism), so that it alerts on lost quality rather than on any change?

"Cost" means three quantities, none of them measured yet: the monitor's compute, the replay GPU-hours a gate needs, and, for canaries, how many requests a bad version served before it was rolled back. Quote one only after it is written under `results/` and the file is named.

**Ground truth.** Every fault is injected with a known start time. Its real harm is measured offline against dataset labels before the benchmark runs. Task quality is measured against dataset labels. An LLM judge is not ground truth.

## Five kinds of evidence

All five survive any allowed change:

1. **A running service.** vLLM behind a gateway API of this repo, which logs every request and response, versions deploys, splits canary traffic, and rolls back.
2. **A release gate in CI.** Replays logged traffic against a candidate and blocks the merge when it fails.
3. **Canary analysis.** A sequential test and automatic rollback.
4. **Production monitors with alerts.** Input drift, output behavior, label-based quality once labels arrive late, and a label-free quality estimate.
5. **The fault-injection benchmark.** Detection delay, miss rate, and false alarms over long healthy traffic, per method.

## Non-negotiables

1. The central question stays answerable, all three parts.
2. All five kinds of evidence survive.
3. The statistics are implemented in this repo and validated. That includes the bootstrap, KL, MMD, CUSUM, ADWIN, confidence sequences, e-detectors, and the canary test. Every test's false-alarm rate or coverage is checked by simulation under its null, and on real A/A traffic (the same model deployed as both versions). River, confseq, and SciPy are reference checks only. A monitor whose false-alarm rate has not been checked cannot appear in the benchmark.
4. Harm is defined by each fault's measured accuracy drop on labeled data against a margin fixed in advance. Quality comes from dataset labels.
5. `docs/EVAL_PROTOCOL.md` is committed in pre-registered form before any benchmark run. Thresholds are tuned only on a separate calibration stream. Every headline number has replicates and confidence intervals. Null and negative results are reported plainly, including a finding that anytime-valid methods were slower and not worth it.
6. It runs on Sathvik's single GPU. No cloud compute, paid APIs, or paid CI runners unless he approves.
7. The headline is the measured statistical behavior of the lifecycle. Prometheus, Grafana, Docker, and CI are supporting pieces. SRE/DevOps platform work and LLM-app building are out of scope. Do not grow the observability stack at the expense of the three parts of the question.
8. Sathvik must be able to explain every component. Python throughout: FastAPI, vLLM, Transformers, Postgres or DuckDB over Parquet, Docker Compose, GitHub Actions. No custom CUDA or Triton kernels. No Kubernetes in the core.
9. Honest wording, as in the traffic paragraph above. Never "first". `PRIOR_WORK.md` lists the close prior work. Never report a number that was not measured.
10. No code, data, or results from outside employment. Prior deploy-gate experience informs the design and never appears as project data or numbers.

## Change test

Run this before adopting any change, whether it is an inferred shortcut, a new idea, a swapped tool, or a decision from Sathvik:

1. Is the central question still answerable, all three parts?
2. Do all five kinds of evidence survive?
3. Is every monitor in the benchmark still validated, and is harm still defined by labeled data?
4. Does it still fit 24 GB?
5. Can Sathvik still explain it, and does the headline stay statistical ML rather than infrastructure?

If all five answers are yes, log the change in `DECISIONS.md` with the reason. If any answer is no, stop and ask.

**Additions** are welcome only after the core stages are done, and must not replace a core piece. Examples that stay allowed later: a second model family (the default service model stays primary); a tool-using agent endpoint as a third task (not in place of the classification tasks); Project A's recommender as a second monitored service (the LLM service stays primary); Kubernetes, for example a local kind cluster, for deployment (never as the headline); more detectors (the core set in `STAGES.md` stays).

**Fallbacks that weaken the evidence** are allowed only after the preferred path has been tried. Tell Sathvik in the next update. Examples: LoRA adapters on one base model for canaries, if two vLLM processes do not fit; a smaller replay horizon, with every method run on the same streams; KL from vLLM's top-k log-probabilities instead of full-vocabulary KL, with the truncation error measured against full KL on a sample.

## Ask first

Spending money; accepting a dataset license or gated-access terms; connecting alerts to any real account; registering a self-hosted runner or making the repo public; opening upstream PRs; contacting anyone; submitting a paper; dropping or reordering a core stage; changing the research question. The full list with the CI notes is in `CONSTRAINTS.md`.

## Why it exists

What this project is for:

- One public system that runs the whole lifecycle end to end.
- Evidence of operating a service under traffic. The traffic here is replayed.
- A public codebase that an ML-platform or backend reviewer would read as software engineering.

## Definition of done

- Stages 0–7 in `STAGES.md` are complete, and the non-negotiables above hold.
- The A/A noise floor, and the null checks for every method, are committed before the benchmark.
- The Gao et al. anchor matched, or its gap is documented (`PRIOR_WORK.md`, `EVAL_PROTOCOL.md`).
- `EVAL_PROTOCOL.md` was committed in pre-registered form before any benchmark run.
- Every headline number has confidence intervals, a baseline, and a commit in `RESUME_FACTS.md`.
- The README reproduces the headline figure from committed scripts, and `docker compose up` plus one command runs the service and a demo fault end to end.
- A workshop-style draft exists. Submitting it requires asking first.
