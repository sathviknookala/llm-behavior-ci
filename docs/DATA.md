# Data

Read before downloading anything or accepting terms. AppWorld is the only item installed, recorded under **AppWorld install record** below. Accepting a license or gated-access terms requires asking Sathvik first.

## Environment

| Item | Role | License as recorded | Source | Status |
|---|---|---|---|---|
| AppWorld | The task environment: tasks, per-task database state, 457 APIs across 9 apps, the execution shell, and the evaluator | Public portion Apache 2.0. Protected portion (task, app, and API-specific code and data, including API docs, solutions, and evaluation tests) ships in encrypted `.bundle` files under Apache 2.0 plus a requirement that any public redistribution of it, or of derivatives, be encrypted. Training models and serving their outputs are stated not to be redistribution. Requires Python 3.11+ | github.com/StonyBrookNLP/appworld ; arxiv.org/abs/2407.18901 | installed 2026-09-29 (record below) |

License and facts were read from the repository README, `LICENSE`, and `pyproject.toml` on 2026-09-27. Re-read them at the pinned version before install and correct this file if they differ.

### Rules that follow from AppWorld's terms

- The authors ask that no code or data extracted or derived from the `.bundle` files be posted online in plain text or images. This repo goes further and keeps everything episode-level local, under the artifact policy below.
- AppWorld's canary string stays in any local file that carries its content.

### Public and local artifacts

This is the single artifact policy; `EVAL_PROTOCOL.md`, `CONSTRAINTS.md`, and the `results/`, `configs/`, and `data/` READMEs point here. It applies to git, CI logs, and any publication.

The public repo may contain only:

- AppWorld, package, model, and configuration versions and hashes;
- split names;
- deterministic selection rules and seeds;
- scenario, task, and episode counts;
- cryptographic hashes of the resolved local task sets;
- aggregate statistics, confidence intervals, detector outputs, cost and latency summaries, and figures built from them.

Local only, never committed or published:

- resolved task IDs;
- task instructions and any task content;
- API documentation derived from protected content;
- plans and plan traces;
- trajectories and tool outputs;
- evaluator reports;
- per-task outcomes;
- database state;
- raw episode logs.

A public validation summary follows the same list. Scenario and task ids used to cluster a run stay local. The summary may carry the input hash, counts, seeds, method parameters, and aggregate intervals.

Local artifacts live under `data/raw/` and `data/processed/`, which a local Git exclude file keeps out of git, or in the log store (`DECISIONS.md` D10). `data/manifests/` is tracked and holds only public items; a manifest with resolved task ids goes under `data/processed/`. Task-id lists that drive a builder are local inputs, not literals in tracked code. The exclude file is not cloned, so recreate it on every clone and check `git status` before every commit.

### Split restrictions

AppWorld has four splits: `train`, `dev`, `test_normal`, `test_challenge`. As recorded on 2026-09-27:

- `train` and `dev` release full ground truth (setup, solution, evaluation, required apps and APIs). `test_*` release only evaluation programs and difficulty indicators.
- `train` may be used to teach the agent and for manual error analysis; `dev` for tuning and manual error analysis.
- `test_*` may be used only for testing and the aggregate score. Do not inspect those tasks or their task-wise reports, and do not tune prompts or hyperparameters on them.
- Hardcoding API calls into the agent's logic is not allowed; generic prompt hints drawn from `train` or `dev` failures are.
- State checkpointing inside an episode gives an unrealistic advantage and is not used by the agent.

### Split roles in this repo

Canonical in `PROJECT_SPEC.md`; `DECISIONS.md` D8 locks it.

| Split | Role here | Ground truth released | Human inspection |
|---|---|---|---|
| `train` | Development-visible tasks; the permanent fixed task set of the Tier 1 CI gate | Full | Allowed |
| `dev` | Calibration, execution-based harm labels, power analysis, canary and monitor development, threshold tuning | Full | Allowed |
| `test_normal` | The frozen held-out final benchmark for the Tier 2 canary and Tier 3 monitoring | Evaluation programs and difficulty indicators only | Not allowed |
| `test_challenge` | Unused unless separately pre-registered later | Evaluation programs and difficulty indicators only | Not allowed |

Prompts, thresholds, and every other methodology choice are tuned only on `train` and `dev`. During the frozen `test_normal` benchmark, evaluator outcomes may be consumed programmatically by the pre-registered canary and monitor logic. No human inspects an individual `test_normal` task or its task-wise report, and no design choice changes afterward. Slices of `test_normal` results use only metadata AppWorld releases for that split: the difficulty indicators, plus the agent's own observed behavior.

## Dropped with the classification tasks

The arXiv metadata, HuffPost News Category, Wild-Time, CivilComments, and WildChat-1M entries from the 2026-09-26 plan are no longer part of the project (`DECISIONS.md` D8). None was downloaded.

## Do not download

- Stack Exchange dumps. The 2024 terms forbid LLM-training use.
- LMSYS-Chat-1M. Gated custom agreement.
- Yelp and Amazon review data. Non-commercial terms.
- MIMIC. Credentialed access.

## Before an install or download is allowed

1. Ask Sathvik to accept the terms.
2. Confirm the license line in this file against the source, and correct it if needed.
3. Record the install date, the exact package and data version, and where `APPWORLD_ROOT` lives on disk. Environment bytes, resolved task sets, and logs do not belong in git.

### AppWorld install record

- **Terms.** Accepted by Sathvik on 2026-09-29, including the rule that nothing extracted or derived from the `.bundle` files is posted online in plain text or images.
- **License check.** The installed package metadata reads `License: Apache-2.0`. The encrypted-redistribution requirement on the protected portion was not re-read at this version; it stays as recorded above.
- **Package.** `appworld==0.1.3.post1`, installed 2026-09-29 from `requirements.txt` into `.venv-service/` (Python 3.12, created with `uv`). `.venv-service/` is listed in `.git/info/exclude`. The older repo `.venv/` (Python 3.13) also contains `appworld==0.1.3.post1` beside `vllm==0.30.0` and `torch`; it is a mixed environment and is not the service environment.
- **Bundles.** `appworld install` unpacked the encrypted app bundle into the service venv's `appworld` package and the tests bundle into `~/.appworld/tests`, outside the repo.
- **Data.** `appworld download data --root data/raw/appworld` on 2026-09-29 wrote data version `0.1.0` to `data/raw/appworld/data/`. `data/raw/**` is in `.git/info/exclude`.
- **`APPWORLD_ROOT`.** `data/raw/appworld/`, relative to the repo root. It is not set persistently; pass `APPWORLD_ROOT=$PWD/data/raw/appworld` on each command.
- **Standalone check.** On 2026-09-29, `AppWorld(task_id=...)` on one `train` task loaded, `evaluate()` returned a `TestTracker`, and `close()` succeeded, outside this package's code. Only hashes, lengths, and the pass flag were printed; no task text, API docs, or evaluator details. AppWorld writes that run's experiment output under `data/raw/appworld/experiments/`.
