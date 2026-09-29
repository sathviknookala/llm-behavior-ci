# Configurations

`replay/` and `monitoring/` contain only placeholders. `models/qwen3_4b_production.json` is the canonical healthy production `model` and `agent` sections; a `RunConfiguration` adds a `task` section, `run_seed`, and `git_commit`. `tasks/` holds the public metadata of the resolved `train`, `dev`, and 3-task `train_smoke` sets: the `TaskConfiguration` fields plus `scenario_count`. The permanent Tier 1 set is not frozen. `faults/` holds versioned configuration diffs. Those diffs are not resolved production or candidate configurations, and two of them are not representable until the schema gains the requested leaves. The typed settings in `src/llm_behavior_ci/config.py` are schemas, not resolved run configurations.

- `models/`: exact model, tokenizer, serving, quantization, and agent-runtime revisions (smolagents version, action interface, step limit). A production or candidate configuration is a hashed combination of these with a prompt.
- `tasks/`: agent and plan-mode prompts, the plan-trace format, AppWorld version, and per split role the deterministic selection rule, seed, and hash of the resolved local task set.
- `replay/`: task-stream schedules: task sampling, task-mix shifts, arrival rate, concurrency, and seeds.
- `faults/`: versioned configuration diffs from the fault catalog.
- `monitoring/`: monitor definitions after their thresholds are frozen in the protocol.

Values governed by `docs/EVAL_PROTOCOL.md` are not set here until the protocol locks them. Nothing here may hold an item from the local-only list in `docs/DATA.md`; resolved task IDs and task content stay local.
