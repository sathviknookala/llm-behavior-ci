# Configurations

- `models/`: exact model, tokenizer, serving, and quantization revisions.
- `tasks/`: prompts, label sets, mappings, and split identities.
- `replay/`: rates, clock compression, concurrency, and seeds.
- `faults/`: versioned configuration diffs from the fault catalog.
- `monitoring/`: monitor definitions after their thresholds are frozen in the protocol.

Values governed by `docs/EVAL_PROTOCOL.md` are not set here until the protocol locks them.
