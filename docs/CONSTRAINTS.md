# Constraints

Read before launching a job, adding a dependency, touching CI, spending money, or publishing.

## GPU

NVIDIA RTX PRO 4000 Blackwell (SM120, 70 SMs). NVIDIA's spec is 24 GB. Sathvik's other repos log 25.2 GB; that figure is not a measurement from this repo. Plan every stage for 24 GB, including a canary that runs two model versions at once.

CUDA 12.9 and PyTorch 2.9.1 are what his other repos use. vLLM already runs on this card in the quantization study. Pin the versions this repo actually installs when the environment exists, and record them here.

Check `nvidia-smi` before a launch. Project A, and Project B if it runs, share this card. Do not start a job that would push the card past 24 GB beside one that is already running. Sathvik sets the priority. Another process's residency shows up as an error or as inflated timings. `scripts/evaluation/capture_aa.py` reads `nvidia-smi` before a live run and refuses to start when compute processes are present. `--observe-hardware` is the only path that records memory and wall time, and only from that reading. A synthetic capture is not a card measurement. `scripts/evaluation/validate_method.py`, `compare_plan_kl.py`, and `assess_harm_study.py` do not read `nvidia-smi`. `scripts/benchmark/run_lifecycle_benchmark.py` and `scripts/replay/replay_detectors.py` do not read `nvidia-smi`. A memory field on a validation report is the caller's number.

Agent episodes are multi-turn with long prompts, so memory and GPU-hours are measured at the agent's real context length, not at a short-prompt benchmark. AppWorld itself runs on CPU.

Time is not a constraint. Log GPU-hours per run anyway. They are part of the cost answer and of `RESUME_FACTS.md`.

## Stack

The target stack boundary is Python 3.11+ (AppWorld's floor), vLLM for serving, smolagents for the agent loop, AppWorld for tasks, tools, state, and evaluation, FastAPI for the gateway, Hugging Face Transformers for full-vocabulary teacher-forced scoring, Postgres or DuckDB over Parquet for logs (`DECISIONS.md` D10), Prometheus, optional Grafana (`DECISIONS.md` D11), Docker Compose, and GitHub Actions. The current CPU suite is written with the standard-library `unittest` runner; pytest is pinned but unused.

No custom CUDA or Triton kernels. No Kubernetes in the core. No custom simulated environment or tool set unless AppWorld integration requires it.

### Environments

Two Python environments, because AppWorld 0.1.3 pins `pydantic<2`, `fastapi<0.111`, and `pytest<9`, while vLLM 0.30 requires `pydantic>=2.12` and `fastapi>=0.133`. They cannot share one.

- **CPU/test subset**, pinned once in `environments/test-requirements.txt`: FastAPI, Uvicorn, Prometheus, SciPy, River, pytest, and `httpx==0.27.2`. httpx is what `fastapi.testclient.TestClient` (built on Starlette's `TestClient`) needs at import time; without it, any test module importing `TestClient` fails at discovery, not at the assertion it was meant to check. The pin sits inside AppWorld 0.1.3's own `httpx>=0.27.0,<0.28.0` constraint, so the CPU subset and the service environment resolve to the same httpx without conflict. This file has no AppWorld dependency and installs cleanly with only `pip`, no compilers.
- **Service environment**, `environments/service-requirements.txt`: `-r test-requirements.txt` plus `appworld==0.1.3.post1`. Root `requirements.txt` is `-r environments/service-requirements.txt`. smolagents and Transformers are not pinned. `service.py` `create_app` and `scripts/service/serve.py` exist; Docker Compose does not. This environment reaches vLLM only over its OpenAI-compatible HTTP API and uses the FastAPI and pydantic versions AppWorld allows. The CPU/test subset is the single source of truth for what a CPU job installs; the service and root files only add AppWorld on top of it, so there is nothing left to inline or restate in a workflow.
- **vLLM environment**, not yet created: `docker/vllm/` is an empty context placeholder. `environments/vllm-requirements.txt` pins `vllm==0.30.0` and cannot share AppWorld's pydantic v1 pin. Full-vocabulary teacher-forced KL scoring runs there because it needs torch and Transformers on the GPU.

The GitHub-hosted CPU job installs `environments/test-requirements.txt` directly (`pip install -r environments/test-requirements.txt`), not an inline package list. It does not install AppWorld and does not install `environments/service-requirements.txt` or root `requirements.txt`, both of which pull in AppWorld through the `-r` chain. A version bump belongs in `environments/test-requirements.txt` (or, for AppWorld itself, in `environments/service-requirements.txt`) and nowhere else; the workflow file has no package versions of its own to drift out of sync.

confseq is not in `requirements.txt`. Version 0.0.11 ships only as source and needs system Boost to build, which this machine does not have. Installing Boost is a system package change; ask first. The validation runner does not import confseq to compute a boundary. It records whether the module can be found.

## Code execution

The agent's actions are model-generated code or API calls. They execute only through AppWorld (`DECISIONS.md` D18). smolagents documents its `LocalPythonExecutor` as not a security boundary; it never runs an action here. Where AppWorld executes, in process or in Docker via `appworld serve`, is `DECISIONS.md` D20 and is settled before the first episode.

No CI job, hosted or self-hosted, prints anything on the local-only list in `DATA.md`, because job logs are public on a public repo. The Tier 1 gate resolves its fixed `train` task set locally, checks it against the committed hash, and reports only the split name, task-set hash, counts, configuration hashes, and aggregate statistics.

No package build command exists. The current CPU workflow installs `environments/test-requirements.txt`, sets `PYTHONPATH=src`, discovers `tests/unit`, `tests/ci`, `tests/integration`, and `tests/validity` with `python -m unittest`, checks that bare `python scripts/run_offline_gate.py`, `scripts/benchmark/run_lifecycle_benchmark.py`, `scripts/replay/replay_detectors.py`, `scripts/service/serve.py`, and `scripts/evaluation/lock_protocol.py` each exit 2, then runs `scripts/synthetic/run_connected_lifecycle.py`. There is no end-to-end, lint, service, or GPU command. The validation commands refuse `results/`, reject `test_normal` and `test_challenge`, and do not start AppWorld or vLLM. A lifecycle-harness call with injected tasks is not the frozen benchmark.

Shared contracts the validation runner does not change:

- `MonitorObservation` has no scenario id, task id, or repetition. Clustering and repeated-task effects take an aligned `AAContext`, or `AAStudyRow`s copied from an A/A capture by `aa_study_rows`. Task and scenario ids stay out of the public summary.
- `runtime/scoring.py` `score_top_k` does not accept a vocabulary size. `compare_plan_kl` calls `score_full` for the full distribution and `truncated_next_token_kl` for the top-k approximation so the vocabulary size is recorded. Full-vocabulary teacher forcing still belongs in the vLLM environment above. Separately, `runtime/scoring.py`'s `ScoringContract`/`verify_scoring_contracts`/`score_scoring_contracts` are the mechanically-enforced teacher-forced KL path the offline gate calls: a `"full"` claim is only honored when every scored position's support exactly matches a caller-declared vocabulary size and is normalized; the live vLLM chat-completions agent (`runtime/agent.py` `teacher_force_plan`) only ever returns top-k/truncated support, so it can only ever satisfy a `"top_k"` claim. `score_full`/`score_top_k` themselves stay raw-array primitives with no proof of their own, unchanged, for the validation runner above.

## CI

GitHub advises that self-hosted runners should almost never be used with public repositories, because a pull request from a fork can run code on the machine. GitHub-hosted GPU runners exist, are paid, and need an organization plan. The handoff records a price of about $0.052/min for Linux, checked 2026-09-26 at docs.github.com. That price is not a bill this repo has incurred.

So:

- The current GitHub-hosted CPU job installs `environments/test-requirements.txt` and runs unit tests, the plan-only gate CI tests, the synthetic integration contract, the local validity suite, exit-2 checks on the gate, benchmark, replay, serve, and protocol-lock commands, and the synthetic connected lifecycle. Lint is still absent. `STUDY_BUDGETS` in `experiments/validation.py` caps a `cpu_fast` or `gpu` run at 40 seeds, sample size 64, horizon 64, 40 permutations, 200 resamples, and 20,000 work units. A `simulation` run caps at 2,000 seeds, sample size 5,000, horizon 5,000, 2,000 permutations, 20,000 resamples, and 5,000,000 work units. `gpu` uses the `cpu_fast` cap for its CPU portion. These caps are runner limits. They are not α, a harm margin, or a stopping rule.
- The GPU gate runs either on a self-hosted runner triggered only by pushes to protected branches or by manual dispatch, never by fork pull requests, or as a local `make gate` whose result is posted as a commit status.
- Ask Sathvik before registering any runner (`DECISIONS.md` D12).

## Ask first

- Spending money: cloud GPUs, paid APIs, LLM judges, GitHub larger runners.
- Accepting a dataset's or environment's license or gated-access terms, AppWorld included (`DATA.md`).
- Connecting alerts to any real account (Slack, email, phone). Alerts go only to channels Sathvik sets up.
- Registering a self-hosted runner, or making the repo public.
- Opening upstream PRs.
- Contacting anyone.
- Submitting a paper.
- Dropping or reordering a core stage (`STAGES.md`).
- Changing the research question (`PROJECT_SPEC.md`).

## What carries over, and what does not

This is a new repo. No code carries over.

Method that carries over, as priors rather than as results of this repo:

- The evaluation-rig style of github.com/sathviknookala/llm_quantization_threshold: protocol committed first; 10,000-draw bootstrap; a replication floor from repeated baseline runs, reported beside every difference and never subtracted; a decision log. That study's teacher-forced KL (frozen trajectories, per-position token KL) is the model for the gate's plan-trace KL. Its finding that NVFP4 diverges about 10× more than FP8 from BF16, on Llama 3.1 8B, is a prior on which quantization faults should matter. It is a prior for a different model and task. It is not a result for Qwen3-4B on AppWorld.
- Prior experience with a retrain → validate → deploy gate, as knowledge of a design. No code, data, or numbers from it.
- vLLM on this GPU, from the quantization study, as evidence the card can run vLLM. Re-measure memory for this model at agent context lengths.

Reference implementations, for checks only. The validation runner records whether each imports and does not execute it. A reference case carries the expected number. Do not copy their code, and do not ship them as this repo's tests:

- River (BSD-3-Clause): ADWIN, Page-Hinkley.
- confseq (MIT; early-stage, last release v0.0.11 in January 2023): confidence-sequence boundaries.
- SciPy: KS and chi-square.
- The released code of Gao et al.: the MMD test, as the stage 3 anchor.

smolagents (Apache 2.0) and AppWorld (Apache 2.0 with the encrypted-redistribution requirement in `DATA.md`) are runtime dependencies, not reference checks. They are not vendored.

Do not depend on Alibi Detect. Its license is BSL 1.1 since January 2024 (`PRIOR_WORK.md`).
