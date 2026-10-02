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

## Recommendation (not taken)

Reduction is required: current initial context plus an 8-turn / 8192-token reserve exceeds 32768, so documentation cannot stay as `str(world.task.api_docs)`.

The `keep_apis(required_apis)` path below was not taken. `required_apis` is parsed from the compiled solution, and the shipped render keeps the full catalog.

Smallest deterministic reduction that still lets the API-documentation corruption fault change the docs string:

1. Load full-mode ground truth and take `required_apis` (deterministic per task id).
2. `docs = world.task.api_docs.keep_apis(sorted(required_apis))` (optionally `.compress_parameters().compress_response_schemas()` and/or reload with `include_response_schemas=False` for further shrink).
3. Coerce to a line-oriented string `app.api: <deterministic JSON of that API's doc dict>` (sorted app and api names), not `str(collection)`.

On this smoke task that line-oriented `keep_apis(required_apis)` form measured about 2176 documentation tokens and about 2329 initial-context tokens (messages shape), which leaves headroom for prompt-v2 and eight full 1024-token turns. Do not implement that reduction in this lane; leave agent, prompts, and appworld adapters unchanged until a follow-up decides the exact coerce path.

## Conclusion

The unreduced `str(ApiDocCollection)` does not fit: initial context is 115336 tokens against `max_model_len` 32768.

## Reduction applied

`LiveAppWorldSession.context()` no longer uses `str(api_docs)`. For a mapping it writes one sorted line per API, `app.api: description | name:type`, and drops response schemas. Strings are left unchanged. This is not `GroundTruth.required_apis`: that list is parsed from the compiled solution, and putting it in the prompt would show the solution's API set. `Task.allowed_apps` is already what `ApiDocCollection.load` receives, and on this task it is the full 11-app catalog, so an app filter does not shrink the text.

The line form is what `api-docs-corrupt-v1` matches. On this same task index, corrupting `supervisor` changed the rendered string.

Measured again on 2026-09-29 through `POST /tokenize` with the messages shape (`add_generation_prompt=true`, `enable_thinking=false`) and `prompt-v2`:

| Quantity | Value |
| --- | --- |
| Rendered documentation characters | 63580 |
| Rendered documentation lines | 457 |
| Documentation tokens (prompt tokenize) | 13309 |
| System-prompt tokens (`prompt-v2`, execute, code, thinking off) | 122 |
| Task-instruction tokens | 33 |
| Initial-context tokens (messages) | 13481 |
| Headroom (`32768 − 13481`) | 19287 |
| Headroom after 8 turns of 1024 generated tokens | 11095 |
| Headroom after 16 turns of 1024 generated tokens | 2903 |

Eight full turns fit, and sixteen still fit before observation text. `step_limit` remains 40; a turn that actually emits 1024 tokens plus a long observation can still fill the window late in an episode. That is history growth, not a reason to cut the catalog down to the solution's APIs. The 122 system-prompt tokens were measured before `prompt-v2` gained its credential sentences. That count was not remeasured.

`AppWorld` has no `initial_state_identity` method. `execute` returns `Execution failed.` text instead of raising. `evaluate` returns a `TestTracker` with `pass_count`, `fail_count`, `num_tests`, `passes`, and `failures`. The adapter follows that shape.

## Live execute episode

On 2026-09-29 one `train_smoke` execute episode was run with `scripts/evaluation/smoke_live_episode.py` against the already-running Qwen3-4B server at `127.0.0.1:8000`, task index 0. The SQLite log stayed under `data/processed/`. It is not a `results/` artifact. No task text, action, or observation is copied here.

The run used working-tree code on top of `a26eef9`. The script's configuration hash was `d4ea3f9877b77b094cffb029d363d45239f866dc95f84ac494134cfc186eda77`, which embeds that git revision. Prompt-body edits are not hashed fields. A later run will not reproduce that hash.

| Field | Value |
| --- | --- |
| `termination_reason` | `step_limit` |
| `status` | `failed` |
| `model_step_count` | 40 |
| `tool_step_count` | 39 |
| `model_latency_seconds` | 91.585 |
| `wall_seconds` | 92.476 |
| `evaluator_success` | null |
| `passed_requirements` | null |
| `total_requirements` | null |

`apis.supervisor.complete_task` was not called, so `evaluate()` did not run. Of the 39 tool steps, 12 returned an output and 27 were `Execution failed.` API errors. 39 of 40 model outputs started with `apis.`. The limit is the production `agent.step_limit` of 40.

Earlier attempts in the same session stopped at `invalid_action` when a prose turn was fatal. This run is after that turn is recorded and skipped, and after a failed shell call is fed back as its last line rather than the traceback. The shell accepts keyword arguments only. Opening a world freezes `datetime` and `perf_counter`; episode timestamps use `runtime/clock.py` so they stay on the real clock.

## Smoke tasks 0–2

On 2026-09-29 the same production configuration was run on `train_smoke` indices 0, 1, and 2. Logs stayed under `data/processed/`. They are not `results/` artifacts. No task text, action, or observation is copied here.

The first index-0 run above is the one whose 27 `Execution failed.` steps were classified. Twelve of its tool steps returned `Execution successful.` AppWorld records stdout and uses that sentence when stdout is empty. A bare `apis.<app>.<api>(...)` expression does not print its return value, and the parser rejects a `print(...)` wrapper because that is two calls. `LiveAppWorldSession.execute` now prints a single call expression before handing it to AppWorld. A direct check of `apis.supervisor.show_profile()` returned `Execution successful.` without that wrap and a JSON object with the profile keys with it.

Indices 1 and 2, before that wrap, and all three indices after it, still stopped at `step_limit` with no `complete_task` and no evaluator outcome. The configuration hash for these six runs is `3ec22b933490026f011a40d2f59f171126d7c671bb701493132dddebbafa9a19`.

| Task index | Wrap | Termination | Model steps | Tool steps | Tool outputs | Tool errors | `complete_task` | Evaluator | Model latency (s) | Wall (s) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | no | `step_limit` | 40 | 39 | 12 | 27 | no | no | 91.585 | 92.476 |
| 1 | no | `step_limit` | 40 | 40 | 0 | 40 | no | no | 81.063 | 81.920 |
| 2 | no | `step_limit` | 40 | 40 | 0 | 40 | no | no | 109.860 | 110.827 |
| 0 | yes | `step_limit` | 40 | 39 | 1 | 38 | no | no | 92.110 | 93.016 |
| 1 | yes | `step_limit` | 40 | 40 | 0 | 40 | no | no | 82.549 | 83.388 |
| 2 | yes | `step_limit` | 40 | 40 | 0 | 40 | no | no | 110.855 | 111.825 |

None of these episodes called `apis.supervisor.complete_task` or `evaluate()`. The index-0 failures are credential and mailbox state, not missing parameter names or a parser mismatch. Indices 1 and 2 are login credential rejections and unauthorized calls, with no successful tool output for the print wrap to change. Compact API lines already include the supervisor password API. Restoring parameter descriptions was not required for these failures.

## Live runtime integration closed

On 2026-09-29 one further `train_smoke` index-0 execute episode was run with `scripts/evaluation/smoke_live_episode.py` against the already-running Qwen3-4B server at `127.0.0.1:8000`. The SQLite log is `data/processed/smoke_block1_task0.sqlite`. It is not a `results/` artifact. No task text, action, observation, or credential is copied here.

Execute mode now calls `session.evaluate()` when `agent.step_limit` is reached and stores the outcome with `status="failed"` and `termination_reason="step_limit"`. Evaluator success does not complete the episode. `prompt-v2` tells the agent to obtain credentials through documented AppWorld and supervisor APIs, not to guess them, and to reuse values returned by earlier calls.

This episode terminated through `apis.supervisor.complete_task` before the step limit. The run used working-tree code on top of `9ce4e46`. The script's configuration hash was `2971c38a4c770b88eb96fc45dd3cdb865c9ee0ad0a27b4f7b0f3749df3bbc80c`, which embeds that git revision. Prompt-body edits are not hashed fields. Committing this work changes the hash.

| Field | Value |
| --- | --- |
| `termination_reason` | `appworld_completed` |
| `status` | `completed` |
| `model_step_count` | 3 |
| `tool_step_count` | 3 |
| `successful_tool_call_count` | 1 |
| `error_tool_call_count` | 2 |
| `model_latency_seconds` | 6.932 |
| `wall_seconds` | 7.341 |
| `evaluator_success` | false |
| `passed_requirements` | 1 |
| `total_requirements` | 8 |

The stored episode is `finished` and its `evaluator_outcome` is non-null. That closes the live runtime integration: a real vLLM call, real AppWorld actions and observations, termination, `evaluate()`, and a persisted `EpisodeResult`. Model task success is not part of the close. One of eight requirements passed. Baseline AppWorld capability is stage 1. A later failure to solve a task is an experimental outcome unless it exposes a runtime defect.

## Workflow controller

This section is an ownership note, not a measurement.

The AppWorld runtime owns initialization, authentication and session setup, credential and token handling, API transport, and capability filtering. `WorkflowControlledAgent` owns the model-generated semantic plan, the model-declared progress ledger, structural validation of that ledger, exact-repeat detection, generic stall signaling, and completion gating. It has no access to evaluator results or ground truth.

An execute turn performs one model generation. The initial plan is part of that same first generation as the first action. The model still chooses the API, the entity, ids, filters, pagination, how to read an observation, which mutation to make, the task-specific sequence, and whether the observation supports marking a plan step complete.

`plan_progress_v1` on the 14B Spotify capability configuration is a Stage-1 baseline candidate. It is not a qualified baseline. No 20-task pilot of this controller is recorded here.
