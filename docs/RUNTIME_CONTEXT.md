# Runtime context size (API documentation)

Measurement note only. Not a protocol threshold and not a `results/` artifact.
Date: 2026-09-29.

## Setup

- Model: `Qwen/Qwen3-4B`
- `max_model_len`: 32768
- Production sampling `max_tokens`: 1024
- Production `step_limit`: 40
- Prompt settings measured: `prompt-v1`, `plan-v1`, `thinking_enabled=False`, `action_interface=code`, mode `execute`
- Task set: public `configs/tasks/train_smoke.json` verified against the local processed set via `verify_task_set`; task index 0 only (id not recorded here)
- Tokenize endpoint: `POST http://127.0.0.1:8000/tokenize` with model `Qwen/Qwen3-4B`
- Initial-context shape that worked: **messages** (`add_generation_prompt=true`, `chat_template_kwargs.enable_thinking=false`), matching `plan_prefix_payload`
- Component counts below use the prompt form of `/tokenize` (`add_special_tokens=false`) so each string is scored alone
- No `world.execute`, no agent run, no world mutation; session closed after measurement
- Protected content (instructions, docs text, actions, observations, task ids, evaluator details) is omitted; only types, lengths, token counts, hashes, and booleans appear here

## Measured quantities (task index 0, current coercion)

| Quantity | Value |
| --- | --- |
| `type(world.task.api_docs).__name__` | `ApiDocCollection` |
| API-documentation character length (`None`→`""`, `str` kept, else `str()`, same as `LiveAppWorldSession.context()`) | 425546 |
| API-documentation token count (prompt tokenize) | 115183 |
| System-prompt character length (`render_system_text` as above) | 456 |
| System-prompt token count (prompt tokenize) | 103 |
| Prompt-v2 extra string character length (synthetic one-sentence execute instruction; no API docs) | 78 |
| Prompt-v2 extra token count (prompt tokenize) | 21 |
| Task-instruction character length | 176 |
| Task-instruction token count (prompt tokenize) | 33 |
| Initial-context token count (messages: system + user=`instruction\\n`+docs) | 115336 |
| Headroom (`32768 − initial`) | −82568 |
| Headroom after prompt-v2 reserve (`headroom − 21`) | −82589 |
| Tokens remaining for history (after initial + prompt-v2) | 0 (budget already exceeded) |
| Turns of 1024 generated tokens that fit in that remainder (before any observation text) | 0 |
| Initial + prompt-v2 + 8×1024 reserve | 123549 |
| Exceeds `32768` with an 8-turn / 8192-token reserve? | **yes** |

## Multi-turn arithmetic

```
max_model_len                 = 32768
initial_context_tokens        = 115336
prompt_v2_extra_tokens        = 21
headroom                      = 32768 - 115336 = -82568
headroom_after_prompt_v2      = -82568 - 21 = -82589
history_budget                = max(0, headroom_after_prompt_v2) = 0
turns_of_1024_in_history      = 0 // 1024 = 0
eight_turn_reserve            = 8 * 1024 = 8192
initial + v2 + eight_turn     = 115336 + 21 + 8192 = 123549 > 32768
```

Production may run up to `step_limit=40` turns at `max_tokens=1024`, but the first request alone already exceeds `max_model_len`, so no multi-turn history budget exists under the current documentation coercion.

## Docs selector surface (names only)

Public callables on the live objects (no task-relevant or app-relevant docs selector among them):

- `Task` methods: `close`, `load`, `save`
- `AppWorld` methods: `close`, `close_all`, `evaluate`, `execute`, `expose_internals`, `initialize`, `load_state`, `parse_api_calls_log`, `parse_environment_io_log`, `save_logs`, `save_state`, `shell`, `task_completed`, `time_freezer`

Related non-method surface (types / counts only):

- `Task.allowed_apps`: `list` length 11 (already the `ApiDocCollection.load(load_apps=...)` argument AppWorld uses)
- `Task.api_docs`: `ApiDocCollection` (dict subclass) with 11 app keys
- `AppWorld.show_api_response_schemas`: `bool` (True in this open); maps to `Task.load(..., include_api_response_schemas=...)`
- `GroundTruth` (full mode) exposes `required_apps` (list length 2 on this task) and `required_apis` (set length 10 on this task); minimal mode left `required_apps` unset / `required_apis` empty

`ApiDocCollection` public methods (AppWorld library; not visible via a plain instance `dir` because of the Munch base, but present on the class): `build`, `load`, `compress_parameters`, `compress_response_schemas`, `keep_apps`, `remove_apps`, `keep_apis`, `remove_apis`, `remove_fields`, `function_calling`, `openapi`, `copy`, `save`.

There is **no** task- or world-method named as a docs selector. App-relevant filtering is `ApiDocCollection.keep_apps` / `remove_apps` / `load(load_apps=...)`. Task-relevant filtering is available only through full-mode `GroundTruth.required_apps` / `required_apis` fed into `keep_apps` / `keep_apis`, not as a method on `Task` or `AppWorld`.

## Corruption compatibility note

`api-docs-corrupt-v1` redacts lines that match `^app.api`. On this task, `str(ApiDocCollection)` is a single-line dict-style dump: corruption against every app key left the string unchanged (`corruption_changes=false`). A reduction that keeps the fault meaningful must emit line-oriented `app.api: ...` text (or another form the corruptor can match), not bare `str(collection)`.

## Recommendation (not implemented in this lane)

Reduction is required: current initial context plus an 8-turn / 8192-token reserve exceeds 32768, so documentation cannot stay as `str(world.task.api_docs)`.

Smallest deterministic reduction that still lets the API-documentation corruption fault change the docs string:

1. Load full-mode ground truth and take `required_apis` (deterministic per task id).
2. `docs = world.task.api_docs.keep_apis(sorted(required_apis))` (optionally `.compress_parameters().compress_response_schemas()` and/or reload with `include_response_schemas=False` for further shrink).
3. Coerce to a line-oriented string `app.api: <deterministic JSON of that API's doc dict>` (sorted app and api names), not `str(collection)`.

On this smoke task that line-oriented `keep_apis(required_apis)` form measured about 2176 documentation tokens and about 2329 initial-context tokens (messages shape), which leaves headroom for prompt-v2 and eight full 1024-token turns. Do not implement that reduction in this lane; leave agent, prompts, and appworld adapters unchanged until a follow-up decides the exact coerce path.

## Conclusion

Reduction is recommended: initial context is 115336 tokens against `max_model_len` 32768, so even before multi-turn history the prompt does not fit; keep Python coercion unchanged in this lane and apply a `keep_apis(required_apis)` plus corruption-compatible line render later.
