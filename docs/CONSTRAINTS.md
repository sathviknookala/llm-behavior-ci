# Constraints

Read before launching a job, adding a dependency, touching CI, spending money, or publishing.

## GPU

NVIDIA RTX PRO 4000 Blackwell (SM120, 70 SMs). NVIDIA's spec is 24 GB. Sathvik's other repos log 25.2 GB; that figure is not a measurement from this repo. Plan every stage for 24 GB, including a canary that runs two model versions at once.

CUDA 12.9 and PyTorch 2.9.1 are what his other repos use. vLLM already runs on this card in the quantization study. Pin the versions this repo actually installs when the environment exists, and record them here.

Check `nvidia-smi` before a launch. Project A, and Project B if it runs, share this card. Do not start a job that would push the card past 24 GB beside one that is already running. Sathvik sets the priority. Another process's residency shows up as an error or as inflated timings.

Agent episodes are multi-turn with long prompts, so memory and GPU-hours are measured at the agent's real context length, not at a short-prompt benchmark. AppWorld itself runs on CPU.

Time is not a constraint. Log GPU-hours per run anyway. They are part of the cost answer and of `RESUME_FACTS.md`.

## Stack

Python 3.11+ (AppWorld's floor). vLLM for serving. smolagents for the agent loop. AppWorld for tasks, tools, state, and evaluation. FastAPI for the gateway. Hugging Face Transformers for full-vocabulary teacher-forced scoring. Postgres or DuckDB over Parquet for logs (`DECISIONS.md` D10). Prometheus for metrics; vLLM exposes `/metrics`. Grafana is optional (`DECISIONS.md` D11). Docker Compose. GitHub Actions. pytest.

No custom CUDA or Triton kernels. No Kubernetes in the core. No custom simulated environment or tool set unless AppWorld integration requires it.

### Environments

Two Python environments, because AppWorld 0.1.3 pins `pydantic<2`, `fastapi<0.111`, and `pytest<9`, while vLLM 0.30 requires `pydantic>=2.12` and `fastapi>=0.133`. They cannot share one.

- **Service environment**, `requirements.txt` at the root: the gateway, smolagents, AppWorld, the statistics, tests, and the River and SciPy reference checks. It reaches vLLM only over its OpenAI-compatible HTTP API. The pins resolved and imported together on Python 3.12 on 2026-09-27, and the CPU offline-gate tests passed in it; nothing has run against vLLM or AppWorld data yet. The gateway's FastAPI and pydantic versions are the ones AppWorld allows.
- **vLLM environment**, pinned in `docker/vllm/` when stage 0 starts: vLLM with the torch and Transformers it pins. Full-vocabulary teacher-forced KL scoring runs here, since it needs torch and Transformers on the GPU.

confseq is not in `requirements.txt`. Version 0.0.11 ships only as source and needs system Boost to build, which this machine does not have. Installing Boost is a system package change; ask first.

## Code execution

The agent's actions are model-generated code or API calls. They execute only through AppWorld (`DECISIONS.md` D18). smolagents documents its `LocalPythonExecutor` as not a security boundary; it never runs an action here. Where AppWorld executes, in process or in Docker via `appworld serve`, is `DECISIONS.md` D20 and is settled before the first episode.

No CI job, hosted or self-hosted, prints AppWorld content, because job logs are public on a public repo (`DATA.md`). The gate reports task IDs and statistics only.

No build command exists. The standard-library offline-gate test command is recorded in `AGENTS.md`; no GPU or full-suite command exists.

## CI

GitHub advises that self-hosted runners should almost never be used with public repositories, because a pull request from a fork can run code on the machine. GitHub-hosted GPU runners exist, are paid, and need an organization plan. The handoff records a price of about $0.052/min for Linux, checked 2026-09-26 at docs.github.com. That price is not a bill this repo has incurred.

So:

- CPU jobs (unit tests, statistical-validity simulations, lint) run on GitHub-hosted runners.
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

Reference implementations, for checks only. Do not copy their code, and do not ship them as this repo's tests:

- River (BSD-3-Clause): ADWIN, Page-Hinkley.
- confseq (MIT; early-stage, last release v0.0.11 in January 2023): confidence-sequence boundaries.
- SciPy: KS and chi-square.
- The released code of Gao et al.: the MMD test, as the stage 3 anchor.

smolagents (Apache 2.0) and AppWorld (Apache 2.0 with the encrypted-redistribution requirement in `DATA.md`) are runtime dependencies, not reference checks. They are not vendored.

Do not depend on Alibi Detect. Its license is BSL 1.1 since January 2024 (`PRIOR_WORK.md`).
