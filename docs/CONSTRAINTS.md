# Constraints

Read before launching a job, adding a dependency, touching CI, spending money, or publishing.

## GPU

NVIDIA RTX PRO 4000 Blackwell (SM120, 70 SMs). NVIDIA's spec is 24 GB. Sathvik's other repos log 25.2 GB; that figure is not a measurement from this repo. Plan every stage for 24 GB, including a canary that runs two model versions at once.

CUDA 12.9 and PyTorch 2.9.1 are what his other repos use. vLLM already runs on this card in the quantization study. Pin the versions this repo actually installs when the environment exists, and record them here.

Check `nvidia-smi` before a launch. Project A, and Project B if it runs, share this card. Do not start a job that would push the card past 24 GB beside one that is already running. Sathvik sets the priority. Another process's residency shows up as an error or as inflated timings.

Time is not a constraint. Log GPU-hours per run anyway. They are part of the cost answer and of `RESUME_FACTS.md`.

## Stack

Python. vLLM for serving. FastAPI for the gateway. Hugging Face Transformers for full-vocabulary scoring. Postgres or DuckDB over Parquet for logs (`DECISIONS.md` D10). Prometheus for metrics; vLLM exposes `/metrics`. Grafana is optional (`DECISIONS.md` D11). Docker Compose. GitHub Actions. pytest.

No custom CUDA or Triton kernels. No Kubernetes in the core.

No build command exists. The standard-library offline-gate test command is recorded in `AGENTS.md`; no GPU or full-suite command exists.

## CI

GitHub advises that self-hosted runners should almost never be used with public repositories, because a pull request from a fork can run code on the machine. GitHub-hosted GPU runners exist, are paid, and need an organization plan. The handoff records a price of about $0.052/min for Linux, checked 2026-09-26 at docs.github.com. That price is not a bill this repo has incurred.

So:

- CPU jobs (unit tests, statistical-validity simulations, lint) run on GitHub-hosted runners.
- The GPU gate runs either on a self-hosted runner triggered only by pushes to protected branches or by manual dispatch, never by fork pull requests, or as a local `make gate` whose result is posted as a commit status.
- Ask Sathvik before registering any runner (`DECISIONS.md` D12).

## Ask first

- Spending money: cloud GPUs, paid APIs, LLM judges, GitHub larger runners.
- Accepting a dataset's license or gated-access terms (`DATA.md`).
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

- The evaluation-rig style of github.com/sathviknookala/llm_quantization_threshold: protocol committed first; 10,000-draw bootstrap; a replication floor from repeated baseline runs, reported beside every difference and never subtracted; a decision log. That study's teacher-forced KL (frozen trajectories, per-position token KL) is the model for the gate's KL metric. Its finding that NVFP4 diverges about 10× more than FP8 from BF16, on Llama 3.1 8B, is a prior on which quantization faults should matter. It is a prior for a different model. It is not a result for Qwen3-4B.
- Prior experience with a retrain → validate → deploy gate, as knowledge of a design. No code, data, or numbers from it.
- Representation-health diagnostics from that same work, as knowledge: per-dimension statistics and effective rank, reused as a monitor on the embeddings the input-drift detector uses.
- vLLM on this GPU, from the quantization study, as evidence the card can run vLLM. Re-measure memory for this model.

Reference implementations, for checks only. Do not copy their code, and do not ship them as this repo's tests:

- River (BSD-3-Clause): ADWIN, Page-Hinkley.
- confseq (MIT; early-stage, last release v0.0.11 in January 2023): confidence-sequence boundaries.
- SciPy: KS and chi-square.
- The released code of Gao et al.: the MMD test, as the stage 3 anchor.

Do not depend on Alibi Detect. Its license is BSL 1.1 since January 2024 (`PRIOR_WORK.md`).
