# Configurations

- `models/`: exact model, tokenizer, serving, quantization, and agent-runtime revisions (smolagents version, action interface, step limit). A production or candidate configuration is a hashed combination of these with a prompt.
- `tasks/`: agent and plan-mode prompts, the plan-trace format, AppWorld version, and task-ID lists per split role.
- `replay/`: task-stream schedules: task sampling, task-mix shifts, arrival rate, concurrency, and seeds.
- `faults/`: versioned configuration diffs from the fault catalog.
- `monitoring/`: monitor definitions after their thresholds are frozen in the protocol.

Values governed by `docs/EVAL_PROTOCOL.md` are not set here until the protocol locks them. Task-ID lists are tracked; AppWorld task content is not (`docs/DATA.md`).
