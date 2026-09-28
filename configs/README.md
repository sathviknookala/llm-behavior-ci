# Configurations

- `models/`: exact model, tokenizer, serving, quantization, and agent-runtime revisions (smolagents version, action interface, step limit). A production or candidate configuration is a hashed combination of these with a prompt.
- `tasks/`: agent and plan-mode prompts, the plan-trace format, AppWorld version, and per split role the deterministic selection rule, seed, and hash of the resolved local task set.
- `replay/`: task-stream schedules: task sampling, task-mix shifts, arrival rate, concurrency, and seeds.
- `faults/`: versioned configuration diffs from the fault catalog.
- `monitoring/`: monitor definitions after their thresholds are frozen in the protocol.

Values governed by `docs/EVAL_PROTOCOL.md` are not set here until the protocol locks them. Nothing here may hold an item from the local-only list in `docs/DATA.md`; resolved task IDs and task content stay local.
