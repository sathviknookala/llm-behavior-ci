# Data

Read before downloading anything or accepting terms. Nothing in this file has been downloaded or installed. Accepting a license or gated-access terms requires asking Sathvik first.

## Environment

| Item | Role | License as recorded | Source | Status |
|---|---|---|---|---|
| AppWorld | The task environment: tasks, per-task database state, 457 APIs across 9 apps, the execution shell, and the evaluator | Public portion Apache 2.0. Protected portion (task, app, and API-specific code and data, including API docs, solutions, and evaluation tests) ships in encrypted `.bundle` files under Apache 2.0 plus a requirement that any public redistribution of it, or of derivatives, be encrypted. Training models and serving their outputs are stated not to be redistribution. Requires Python 3.11+ | github.com/StonyBrookNLP/appworld ; arxiv.org/abs/2407.18901 | not installed |

License and facts were read from the repository README, `LICENSE`, and `pyproject.toml` on 2026-09-27. Re-read them at the pinned version before install and correct this file if they differ.

### Rules that follow from AppWorld's terms

- The authors ask that no code or data extracted or derived from the `.bundle` files be posted online in plain text or images. Task instructions, API documentation, plan traces, trajectories, and evaluator reports are treated as derived content: they stay in local logs under `data/` or the log store, never in git or any public artifact.
- Committed artifacts under `results/` carry task IDs, outcomes, counts, and statistics.
- AppWorld's canary string stays in any local file that carries its content.

### Split restrictions

AppWorld has four splits: `train`, `dev`, `test_normal`, `test_challenge`. As recorded on 2026-09-27:

- `train` and `dev` release full ground truth (setup, solution, evaluation, required apps and APIs). `test_*` release only evaluation programs and difficulty indicators.
- `train` may be used to teach the agent and for manual error analysis; `dev` for tuning and manual error analysis.
- `test_*` may be used only for testing and the aggregate score. Do not inspect those tasks or their task-wise reports, and do not tune prompts or hyperparameters on them.
- Hardcoding API calls into the agent's logic is not allowed; generic prompt hints drawn from `train` or `dev` failures are.
- State checkpointing inside an episode gives an unrealistic advantage and is not used by the agent.

For this repo: thresholds and prompts are tuned only on `train` and `dev`. Automated per-episode consumption of `test_*` outcomes by a gate, canary, or monitor is not manual inspection, but no one reads `test_*` task content or task-wise reports, and no choice is revised after seeing them. The split roles are pre-registered in `EVAL_PROTOCOL.md`.

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
3. Record the install date, the exact package and data version, and where `APPWORLD_ROOT` lives on disk. Environment bytes and logs do not belong in git.
