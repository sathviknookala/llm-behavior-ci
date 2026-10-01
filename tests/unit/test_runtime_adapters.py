import builtins
import json
import math
import threading
import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch
from urllib.error import URLError

from llm_behavior_ci.config import LoRASettings, RunConfiguration
from llm_behavior_ci.records import TokenLogprob
from llm_behavior_ci.runtime.actions import ActionRejected
from llm_behavior_ci.runtime.agent import (
    AppWorldActionExecutor,
    AppWorldExecuteTool,
    SmolagentsVLLMAgent,
    action_execution_backend,
    bind_appworld_action_executor,
    build_appworld_executor,
    check_model_identity,
    parse_logprobs,
    parse_model_output,
    reject_local_python_executor,
    resolve_action_interface,
    served_model_id,
    validate_chat_request,
)
from llm_behavior_ci.runtime.api_docs import resolve_api_documentation
from llm_behavior_ci.runtime.appworld import (
    LiveAppWorldSession,
    TaskContext,
    _open_appworld,
    render_api_documentation,
)
from llm_behavior_ci.runtime.episode import RuntimeUnavailable
from llm_behavior_ci.runtime.prompts import (
    UnknownPromptVersion,
    render_system_text,
    resolve_prompt_template,
)
from llm_behavior_ci.runtime.scoring import score_full, score_top_k


def _payload() -> dict[str, object]:
    return {
        "model": {
            "model": {
                "repository": "Qwen/Qwen3-4B",
                "revision": "0123456789abcdef0123456789abcdef01234567",
            },
            "tokenizer": {
                "repository": "Qwen/Qwen3-4B",
                "revision": "fedcba9876543210fedcba9876543210fedcba98",
            },
            "quantization": {"method": "none"},
            "vllm_version": "0.30.0",
            "serving": {
                "dtype": "bfloat16",
                "max_model_len": 8192,
                "gpu_memory_utilization": 0.9,
                "max_num_seqs": 16,
                "max_num_batched_tokens": 8192,
                "kv_cache_dtype": "bfloat16",
                "enable_prefix_caching": False,
                "enable_chunked_prefill": False,
                "enforce_eager": False,
                "tensor_parallel_size": 1,
                "max_logprobs": 20,
                "batch_invariant": False,
                "sampler_backend": "native",
            },
        },
        "agent": {
            "smolagents_version": "1.22.0",
            "action_interface": "code",
            "prompt": {
                "prompt_version": "prompt-v1",
                "plan_format_version": "plan-v1",
                "thinking_enabled": False,
            },
            "step_limit": 40,
            "sampling": {
                "temperature": 0.0,
                "top_p": 1.0,
                "top_k": 20,
                "min_p": 0.0,
                "seed": 17,
                "max_tokens": 512,
            },
        },
        "task": {
            "appworld_version": "0.1.3.post1",
            "split": "train",
            "selection_rule": "deterministic_sample",
            "selection_seed": 20260926,
            "task_count": 50,
            "task_set_hash": "c" * 64,
        },
        "run_seed": 7,
        "git_commit": "a" * 40,
        "protocol_hash": "e" * 64,
    }


def _config() -> RunConfiguration:
    return RunConfiguration.from_dict(_payload())


def _context() -> TaskContext:
    return TaskContext(
        task_id="task-1",
        instruction="solve the task",
        api_documentation="docs",
    )


def _completion_body(text: str = "STOP") -> dict[str, object]:
    return {
        "choices": [
            {
                "message": {"content": text},
                "token_ids": [7],
                "logprobs": {
                    "content": [
                        {
                            "token": "token_id:7",
                            "bytes": [55],
                            "logprob": -0.5,
                            "top_logprobs": [],
                        }
                    ]
                },
            }
        ]
    }


class FakeWorld:
    def __init__(self) -> None:
        self.task = SimpleNamespace(
            instruction="book a slot",
            api_docs="calendar docs",
        )
        self.closed = 0

    def execute(self, action: str) -> str:
        if action == "fail":
            raise RuntimeError("missing field")
        return f"ok:{action}"

    def evaluate(self) -> SimpleNamespace:
        return SimpleNamespace(
            success=True,
            passes=["a"],
            failures=[],
            pass_count=1,
            fail_count=0,
            num_tests=1,
            difficulty=2,
        )

    def close(self) -> None:
        self.closed += 1


class _FakeResponse:
    def __init__(self, payload: dict[str, object]) -> None:
        self._payload = payload

    def read(self) -> bytes:
        return json.dumps(self._payload).encode("utf-8")

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        del exc_type, exc, tb


class RuntimeAdapterTests(unittest.TestCase):
    def test_live_session_reports_missing_appworld(self) -> None:
        real_import = builtins.__import__

        def blocked(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "appworld" or name.startswith("appworld."):
                raise ImportError("No module named 'appworld'")
            return real_import(name, globals, locals, fromlist, level)

        with patch("builtins.__import__", side_effect=blocked):
            with self.assertRaises(RuntimeUnavailable):
                _open_appworld("task-1")
        with self.assertRaises(RuntimeUnavailable):
            LiveAppWorldSession(
                "task-1",
                opener=lambda task_id: (_ for _ in ()).throw(ImportError(task_id)),
            )

    def test_live_execute_maps_a_returned_string_and_a_raised_error(self) -> None:
        world = FakeWorld()
        session = LiveAppWorldSession("task-1", opener=lambda task_id: world)
        context = session.context()
        self.assertEqual(context.instruction, "book a slot")
        self.assertEqual(context.api_documentation, "calendar docs")
        ok = session.execute("lookup")
        self.assertEqual(ok.output_text, "ok:lookup")
        self.assertIsNone(ok.error_message)
        self.assertFalse(ok.recoverable)
        failed = session.execute("fail")
        self.assertIsNone(failed.output_text)
        self.assertEqual(failed.error_message, "missing field")
        self.assertTrue(failed.recoverable)
        evaluation = session.evaluate()
        self.assertEqual(evaluation.passed_requirements, 1)
        self.assertEqual(evaluation.total_requirements, 1)
        session.close()
        session.close()
        self.assertEqual(world.closed, 1)

    def test_live_execute_treats_appworld_failure_text_as_a_tool_error(self) -> None:
        world = FakeWorld()

        def execute(action: str) -> str:
            del action
            return "Execution failed. Traceback:\nmissing field\n"

        world.execute = execute
        session = LiveAppWorldSession("task-1", opener=lambda task_id: world)
        failed = session.execute("apis.supervisor.complete_task()")
        self.assertIsNone(failed.output_text)
        self.assertTrue(failed.recoverable)
        self.assertTrue(failed.error_message.startswith("Execution failed."))
        successful = LiveAppWorldSession(
            "task-1",
            opener=lambda task_id: SimpleNamespace(
                task=world.task,
                execute=lambda action: "Execution successful.",
                close=lambda: None,
            ),
        )
        ok = successful.execute("apis.calendar.show()")
        self.assertEqual(ok.output_text, "Execution successful.")
        self.assertIsNone(ok.error_message)
        session.close()
        successful.close()

    def test_live_execute_prints_a_single_call_so_stdout_keeps_the_body(self) -> None:
        seen: list[str] = []

        def execute(action: str) -> str:
            seen.append(action)
            return '{"access_token": "token"}\n'

        world = FakeWorld()
        world.execute = execute
        session = LiveAppWorldSession("task-1", opener=lambda task_id: world)
        result = session.execute("apis.simple_note.login(username='a', password='b')")
        self.assertEqual(
            seen,
            ["print(apis.simple_note.login(username='a', password='b'))"],
        )
        self.assertEqual(result.output_text, '{"access_token": "token"}\n')
        self.assertIsNone(result.error_message)
        session.execute("print(apis.supervisor.show_profile())")
        self.assertEqual(seen[-1], "print(apis.supervisor.show_profile())")
        session.execute("name = apis.supervisor.show_profile()")
        self.assertEqual(seen[-1], "name = apis.supervisor.show_profile()")
        session.close()

    def test_live_evaluate_uses_num_tests_not_the_recorded_pair_count(self) -> None:
        world = FakeWorld()
        world.evaluate = lambda: SimpleNamespace(
            success=False,
            passes=["a"],
            failures=[],
            pass_count=1,
            fail_count=0,
            num_tests=3,
            difficulty=1,
        )
        session = LiveAppWorldSession("task-1", opener=lambda task_id: world)
        evaluation = session.evaluate()
        self.assertFalse(evaluation.success)
        self.assertEqual(evaluation.passed_requirements, 1)
        self.assertEqual(evaluation.total_requirements, 3)
        session.close()

    def test_rendered_api_docs_are_sorted_lines_the_corruptor_can_redact(self) -> None:
        rendered = render_api_documentation(
            {
                "supervisor": {
                    "complete_task": {
                        "description": "Mark the task done.",
                        "parameters": [
                            {"name": "answer", "type": "string", "required": False},
                            {"name": "status", "type": "string", "required": True},
                        ],
                        "response_schemas": {"success": {"type": "object"}},
                    }
                },
                "calendar": {
                    "show": {
                        "description": "Show the day.",
                        "parameters": [
                            {"name": "date", "type": "string", "required": True}
                        ],
                    }
                },
            }
        )
        self.assertEqual(
            rendered,
            "calendar.show: Show the day. | date:string\n"
            "supervisor.complete_task: Mark the task done. | "
            "answer:string?, status:string\n",
        )
        corrupted = resolve_api_documentation(
            rendered,
            api_docs_version="api-docs-corrupt-v1",
            api_docs_app="supervisor",
        )
        self.assertIn("calendar.show: Show the day. | date:string", corrupted)
        self.assertIn(
            "supervisor.complete_task: [documentation removed]",
            corrupted,
        )
        self.assertNotIn("Mark the task done.", corrupted)
        self.assertEqual(render_api_documentation("calendar docs"), "calendar docs")
        self.assertEqual(render_api_documentation(None), "")

    def test_completion_payload_contains_sampling_prompt_and_thinking(self) -> None:
        agent = SmolagentsVLLMAgent("http://127.0.0.1:9")
        agent.begin(_context(), _config())
        payload = agent.completion_payload(agent.messages())
        self.assertEqual(payload["temperature"], 0.0)
        self.assertEqual(payload["seed"], 17)
        self.assertNotIn("extra_body", payload)
        self.assertEqual(payload["top_k"], 20)
        self.assertEqual(payload["min_p"], 0.0)
        self.assertIs(payload["return_tokens_as_token_ids"], True)
        self.assertFalse(payload["chat_template_kwargs"]["enable_thinking"])
        system = payload["messages"][0]["content"]
        self.assertIn("You are an AppWorld tool-using agent.", system)
        self.assertIn("Prompt registry id: prompt-v1", system)

    def test_unknown_prompt_version_is_rejected(self) -> None:
        with self.assertRaises(UnknownPromptVersion):
            resolve_prompt_template("prompt-missing")
        with self.assertRaises(UnknownPromptVersion):
            render_system_text(
                prompt_version="prompt-missing",
                plan_format_version="plan-v1",
                thinking_enabled=False,
                action_interface="code",
                mode="execute",
            )
        agent = SmolagentsVLLMAgent("http://127.0.0.1:9")
        config = replace(
            _config(),
            agent=replace(
                _config().agent,
                prompt=replace(
                    _config().agent.prompt,
                    prompt_version="prompt-missing",
                ),
            ),
        )
        with self.assertRaisesRegex(RuntimeUnavailable, "unknown prompt_version"):
            agent.begin(_context(), config)

    def test_known_prompt_version_is_more_than_an_interpolated_id(self) -> None:
        text = render_system_text(
            prompt_version="prompt-v1",
            plan_format_version="plan-v1",
            thinking_enabled=False,
            action_interface="code",
            mode="plan",
        )
        self.assertIn("You are an AppWorld tool-using agent.", text)
        self.assertIn("Emit a numbered plan before acting.", text)
        self.assertNotEqual(text, "prompt-v1")
        self.assertNotEqual(
            text,
            "prompt_version=prompt-v1\nplan_format_version=plan-v1\n",
        )
        agent = SmolagentsVLLMAgent("http://127.0.0.1:9")
        agent.set_mode("plan")
        agent.begin(_context(), _config())
        system = agent.messages()[0]["content"]
        self.assertIn("Do not execute tools while planning.", system)
        self.assertIn("Prompt registry id: prompt-v1", system)

    def test_local_python_executor_is_not_on_the_action_path(self) -> None:
        self.assertEqual(action_execution_backend(), "appworld_session.execute")
        with self.assertRaisesRegex(RuntimeUnavailable, "LocalPythonExecutor"):
            reject_local_python_executor()
        actions: list[str] = []

        def execute(action: str) -> str:
            actions.append(action)
            return f"ok:{action}"

        executor = bind_appworld_action_executor(execute)
        self.assertEqual(executor("calendar.lookup()"), "ok:calendar.lookup()")
        self.assertEqual(actions, ["calendar.lookup()"])
        self.assertFalse(hasattr(executor, "authorized_imports"))

    def test_unsupported_action_interface_is_rejected(self) -> None:
        with self.assertRaisesRegex(RuntimeUnavailable, "unsupported action_interface"):
            resolve_action_interface("local_python")
        with self.assertRaisesRegex(RuntimeUnavailable, "unsupported action_interface"):
            resolve_action_interface("shell")

    def test_serving_flags_are_left_to_the_launch_spec(self) -> None:
        serving = replace(
            _config().model.serving,
            batch_invariant=True,
            enforce_eager=True,
            enable_prefix_caching=True,
            enable_chunked_prefill=True,
        )
        config = replace(_config(), model=replace(_config().model, serving=serving))
        validate_chat_request(config)
        agent = SmolagentsVLLMAgent("http://127.0.0.1:9")
        agent.begin(_context(), config)
        payload = agent.completion_payload(agent.messages())
        self.assertNotIn("batch_invariant", payload)
        self.assertNotIn("enable_chunked_prefill", payload)

    def test_parse_logprobs_reads_vllm_token_id_strings(self) -> None:
        choice = {
            "token_ids": [7],
            "logprobs": {
                "content": [
                    {
                        "token": "token_id:7",
                        "bytes": [55],
                        "logprob": -0.5,
                        "top_logprobs": [
                            {"token": "token_id:7", "bytes": [55], "logprob": -0.5},
                            {"token": "token_id:9", "bytes": [57], "logprob": -1.5},
                        ],
                    }
                ]
            }
        }
        self.assertEqual(
            parse_logprobs(choice),
            (
                (
                    TokenLogprob(token_id=7, logprob=-0.5, rank=0),
                    TokenLogprob(token_id=9, logprob=-1.5, rank=1),
                ),
            ),
        )

    def test_parse_logprobs_rejects_decoded_token_strings(self) -> None:
        choice = {
            "logprobs": {
                "content": [
                    {"token": "ok", "bytes": [111, 107], "logprob": -0.5, "top_logprobs": []}
                ]
            }
        }
        with self.assertRaisesRegex(RuntimeUnavailable, "token_id:<id>"):
            parse_logprobs(choice)

    def test_model_identity_check_reports_unchecked_fields(self) -> None:
        config = _config()
        unchecked = check_model_identity(
            config,
            {
                "id": "Qwen/Qwen3-4B",
                "revision": "0123456789abcdef0123456789abcdef01234567",
            },
        )
        self.assertIn("weights_digest", unchecked)
        self.assertIn("model.serving.batch_invariant", unchecked)
        self.assertIn("model.serving.dtype", unchecked)
        self.assertNotIn("model.model.repository", unchecked)
        self.assertNotIn("model.model.revision", unchecked)
        with self.assertRaisesRegex(RuntimeUnavailable, "served model id"):
            check_model_identity(config, {"id": "other/model", "root": "other/model"})

    def _teacher_force(
        self,
        forced: dict[str, object],
        prefix_tokens: object,
        posted: list[tuple[str, dict[str, object]]] | None = None,
    ) -> tuple[tuple[TokenLogprob, ...], ...]:
        agent = SmolagentsVLLMAgent("http://127.0.0.1:9")
        agent.begin(_context(), _config())

        def fake_urlopen(request, timeout=None):
            del timeout
            payload = json.loads(request.data.decode("utf-8"))
            if posted is not None:
                posted.append((request.full_url, payload))
            if request.full_url.endswith("/tokenize"):
                return _FakeResponse({"tokens": prefix_tokens, "count": 0})
            return _FakeResponse(forced)

        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            return agent.teacher_force_plan(
                messages=[
                    {"role": "system", "content": "system"},
                    {"role": "user", "content": "plan please"},
                ],
                plan_text="1. open the calendar\n2. book the slot",
            )

    def test_teacher_force_plan_posts_frozen_plan_not_max_tokens_generation(
        self,
    ) -> None:
        posted: list[tuple[str, dict[str, object]]] = []
        logprobs = self._teacher_force(
            {
                "choices": [{"message": {"content": ""}, "logprobs": None}],
                "prompt_token_ids": [11, 3],
                "prompt_logprobs": [None, {"3": {"logprob": -0.25}}],
            },
            [11],
            posted,
        )
        urls = [url for url, _ in posted]
        self.assertEqual(
            urls,
            ["http://127.0.0.1:9/tokenize", "http://127.0.0.1:9/v1/chat/completions"],
        )
        prefix_body = posted[0][1]
        self.assertIs(prefix_body["add_generation_prompt"], True)
        self.assertEqual(prefix_body["messages"][-1]["role"], "user")
        body = posted[1][1]
        self.assertEqual(body["max_tokens"], 1)
        self.assertEqual(body["messages"][-1]["role"], "assistant")
        self.assertEqual(
            body["messages"][-1]["content"], "1. open the calendar\n2. book the slot"
        )
        self.assertNotEqual(body["max_tokens"], _config().agent.sampling.max_tokens)
        self.assertNotIn("extra_body", body)
        self.assertIs(body["return_token_ids"], True)
        self.assertIs(body["add_generation_prompt"], False)
        self.assertEqual(body["prompt_logprobs"], _config().model.serving.max_logprobs)
        self.assertFalse(body["chat_template_kwargs"]["enable_thinking"])
        self.assertEqual(
            logprobs,
            ((TokenLogprob(token_id=3, logprob=-0.25, rank=0),),),
        )

    def test_teacher_force_plan_returns_only_positions_after_the_prompt_prefix(
        self,
    ) -> None:
        logprobs = self._teacher_force(
            {
                "choices": [{"message": {"content": ""}, "logprobs": None}],
                "prompt_token_ids": [11, 12, 13, 7, 8],
                "prompt_logprobs": [
                    None,
                    {"12": {"logprob": -1.0}},
                    {"13": {"logprob": -2.0}},
                    {"7": {"logprob": -0.3}},
                    {"8": {"logprob": -0.6}},
                ],
            },
            [11, 12, 13],
        )
        self.assertEqual(
            logprobs,
            (
                (TokenLogprob(token_id=7, logprob=-0.3, rank=0),),
                (TokenLogprob(token_id=8, logprob=-0.6, rank=0),),
            ),
        )

    def test_teacher_force_plan_prefix_mismatch_is_runtime_unavailable(self) -> None:
        forced = {
            "choices": [{"message": {"content": ""}, "logprobs": None}],
            "prompt_token_ids": [11, 12, 7],
            "prompt_logprobs": [None, {"12": {"logprob": -1.0}}, {"7": {"logprob": -0.3}}],
        }
        with self.assertRaisesRegex(RuntimeUnavailable, "plan boundary"):
            self._teacher_force(forced, [11, 99])
        with self.assertRaisesRegex(RuntimeUnavailable, "plan boundary"):
            self._teacher_force(forced, [11, 12, 7])
        with self.assertRaisesRegex(RuntimeUnavailable, "plan prefix"):
            self._teacher_force(forced, [])

    def test_teacher_force_plan_uses_prompt_token_ids_to_identify_the_forced_token(
        self,
    ) -> None:
        logprobs = self._teacher_force(
            {
                "choices": [{"message": {"content": ""}, "logprobs": None}],
                "prompt_token_ids": [11, 3, 9],
                "prompt_logprobs": [
                    None,
                    {
                        "3": {"logprob": -0.25, "rank": 2},
                        "5": {"logprob": -0.1, "rank": 1},
                        "1": {"logprob": -3.0, "rank": 3},
                    },
                    {
                        "9": {"logprob": -0.4, "rank": 1},
                    },
                ],
            },
            [11],
        )
        self.assertEqual(
            logprobs,
            (
                (
                    TokenLogprob(token_id=3, logprob=-0.25, rank=0),
                    TokenLogprob(token_id=1, logprob=-3.0, rank=1),
                    TokenLogprob(token_id=5, logprob=-0.1, rank=2),
                ),
                (TokenLogprob(token_id=9, logprob=-0.4, rank=0),),
            ),
        )

    def test_teacher_force_plan_without_prompt_token_ids_is_runtime_unavailable(
        self,
    ) -> None:
        with self.assertRaises(RuntimeUnavailable):
            self._teacher_force(
                {
                    "choices": [{"message": {"content": ""}, "logprobs": None}],
                    "prompt_logprobs": [None, {"3": {"logprob": -0.25, "rank": 1}}],
                },
                [11],
            )

    def test_teacher_force_plan_missing_forced_token_is_runtime_unavailable(
        self,
    ) -> None:
        with self.assertRaises(RuntimeUnavailable):
            self._teacher_force(
                {
                    "choices": [{"message": {"content": ""}, "logprobs": None}],
                    "prompt_token_ids": [11, 3],
                    "prompt_logprobs": [None, {"5": {"logprob": -0.1, "rank": 1}}],
                },
                [11],
            )

    def test_teacher_force_plan_none_position_after_zero_is_runtime_unavailable(
        self,
    ) -> None:
        with self.assertRaises(RuntimeUnavailable):
            self._teacher_force(
                {
                    "choices": [{"message": {"content": ""}, "logprobs": None}],
                    "prompt_token_ids": [11, 3, 9],
                    "prompt_logprobs": [None, None, {"9": {"logprob": -0.4, "rank": 1}}],
                },
                [11],
            )

    def test_concurrent_begins_isolate_history(self) -> None:
        agent = SmolagentsVLLMAgent("http://127.0.0.1:9")
        barrier = threading.Barrier(2, timeout=5)
        seen: dict[str, list[str]] = {"a": [], "b": []}
        errors: list[BaseException] = []
        markers = threading.local()

        def urlopen(request, timeout=None):
            del request, timeout
            return _FakeResponse(
                _completion_body(f"CALL calendar lookup\n{markers.marker}")
            )

        def worker(name: str, marker: str) -> None:
            try:
                agent.begin(_context(), _config())
                barrier.wait()
                markers.marker = marker
                turn = agent.next_turn(tool_output=None)
                barrier.wait()
                messages = agent.messages()
                seen[name] = [item["content"] for item in messages if item["role"] == "assistant"]
                self.assertEqual(turn.action, marker)
            except BaseException as error:
                errors.append(error)

        first = threading.Thread(target=worker, args=("a", "action-a()"))
        second = threading.Thread(target=worker, args=("b", "action-b()"))
        with patch("urllib.request.urlopen", side_effect=urlopen):
            first.start()
            second.start()
            first.join()
            second.join()
        self.assertEqual(errors, [])
        self.assertEqual(seen["a"], ["CALL calendar lookup\naction-a()"])
        self.assertEqual(seen["b"], ["CALL calendar lookup\naction-b()"])

    def test_parse_model_output_stop_and_call(self) -> None:
        self.assertEqual(parse_model_output("STOP"), (None, None, None))
        self.assertEqual(parse_model_output("STOP\nmore"), (None, None, None))
        self.assertEqual(
            parse_model_output("CALL calendar lookup\napp.lookup()"),
            ("app.lookup()", "calendar", "lookup"),
        )
        with self.assertRaises(ActionRejected):
            parse_model_output("plain action")
        self.assertEqual(
            SmolagentsVLLMAgent("http://127.0.0.1:9").parse_model_output("STOP"),
            (None, None, None),
        )

    def test_full_score_uses_next_token_kl_and_is_labeled_full(self) -> None:
        values = ((math.log(0.5), math.log(0.5)),)
        score = score_full(values, values)
        self.assertEqual(score.kind, "full")
        self.assertAlmostEqual(score.mean_kl_nats, 0.0, delta=1e-9)

    def test_top_k_score_is_labeled_top_k(self) -> None:
        fake = SimpleNamespace(
            approximation="top_k",
            position_kl_nats=(0.25,),
            mean_kl_nats=0.25,
        )
        with patch(
            "llm_behavior_ci.stats.kl.truncated_next_token_kl",
            create=True,
            return_value=fake,
        ):
            score = score_top_k(((0.0,),), ((0.0,),))
        self.assertEqual(score.kind, "top_k")
        self.assertEqual(score.mean_kl_nats, 0.25)

    def test_http_failure_is_runtime_unavailable(self) -> None:
        agent = SmolagentsVLLMAgent("http://127.0.0.1:9")
        agent.begin(_context(), _config())
        with patch(
            "urllib.request.urlopen",
            side_effect=URLError("down"),
        ):
            with self.assertRaises(RuntimeUnavailable):
                agent.next_turn(tool_output=None)

    def test_generate_is_the_smolagents_model_call_next_turn_runs_on(self) -> None:
        agent = SmolagentsVLLMAgent("http://127.0.0.1:9")
        agent.begin(_context(), _config())
        with patch(
            "urllib.request.urlopen",
            side_effect=lambda request, timeout=None: _FakeResponse(
                _completion_body("CALL calendar lookup\napp.lookup()")
            ),
        ):
            chat_message = agent.generate(agent.messages())
        self.assertEqual(chat_message.content, "CALL calendar lookup\napp.lookup()")
        self.assertIn("choices", chat_message.raw)
        action, app_name, api_name = parse_model_output(chat_message.content)
        self.assertEqual((action, app_name, api_name), ("app.lookup()", "calendar", "lookup"))

    def test_appworld_execute_tool_delegates_to_appworld_and_is_not_a_sandbox(
        self,
    ) -> None:
        calls: list[str] = []

        def execute(action: str) -> str:
            calls.append(action)
            return f"ok:{action}"

        tool = AppWorldExecuteTool(execute)
        self.assertEqual(tool.name, "appworld_execute")
        self.assertEqual(tool.output_type, "object")
        self.assertFalse(hasattr(tool, "authorized_imports"))
        self.assertEqual(tool("calendar.lookup()"), "ok:calendar.lookup()")
        self.assertEqual(calls, ["calendar.lookup()"])

    def test_build_appworld_executor_selects_the_configured_action_interface(
        self,
    ) -> None:
        executed: list[str] = []

        def execute(action: str) -> str:
            executed.append(action)
            return f"ok:{action}"

        code_executor = build_appworld_executor("code", execute)
        self.assertIsInstance(code_executor, AppWorldActionExecutor)
        self.assertEqual(code_executor("a()"), "ok:a()")

        tool_executor = build_appworld_executor("tool_calling", execute)
        self.assertIsInstance(tool_executor, AppWorldExecuteTool)
        self.assertEqual(tool_executor("b()"), "ok:b()")

        self.assertEqual(executed, ["a()", "b()"])
        with self.assertRaisesRegex(RuntimeUnavailable, "unsupported action_interface"):
            build_appworld_executor("local_python", execute)


class LoRAServedModelTests(unittest.TestCase):
    def test_served_model_id_is_the_base_repository_when_lora_is_unset(self) -> None:
        config = _config()
        self.assertIsNone(config.model.lora)
        self.assertEqual(served_model_id(config), "Qwen/Qwen3-4B")

    def test_served_model_id_is_the_adapter_repository_when_lora_is_set(self) -> None:
        config = replace(
            _config(),
            model=replace(
                _config().model,
                lora=LoRASettings(
                    repository="org/adapter",
                    revision="d" * 40,
                ),
            ),
        )
        self.assertEqual(served_model_id(config), "org/adapter")

    def test_completion_and_teacher_force_payloads_name_the_lora_adapter(self) -> None:
        config = replace(
            _config(),
            model=replace(
                _config().model,
                lora=LoRASettings(
                    repository="org/adapter",
                    revision="d" * 40,
                ),
            ),
        )
        agent = SmolagentsVLLMAgent("http://127.0.0.1:9")
        agent.begin(_context(), config)
        completion_payload = agent.completion_payload(agent.messages())
        self.assertEqual(completion_payload["model"], "org/adapter")
        teacher_force_payload = agent.teacher_force_payload(
            messages=agent.messages(),
            plan_text="1. call calendar.lookup()",
        )
        self.assertEqual(teacher_force_payload["model"], "org/adapter")

    def test_healthy_configuration_still_names_the_base_repository(self) -> None:
        agent = SmolagentsVLLMAgent("http://127.0.0.1:9")
        agent.begin(_context(), _config())
        payload = agent.completion_payload(agent.messages())
        self.assertEqual(payload["model"], "Qwen/Qwen3-4B")


class ApiDocsCorruptionWiringTests(unittest.TestCase):
    def test_messages_leave_documentation_unchanged_when_unset(self) -> None:
        agent = SmolagentsVLLMAgent("http://127.0.0.1:9")
        agent.begin(_context(), _config())
        content = agent.messages()[1]["content"]
        self.assertIn("docs", content)

    def test_messages_apply_the_corruption_transform_when_configured(self) -> None:
        context = TaskContext(
            task_id="task-1",
            instruction="solve the task",
            api_documentation="calendar.lookup: start_time, end_time\n",
        )
        config = replace(
            _config(),
            agent=replace(
                _config().agent,
                api_docs_version="api-docs-corrupt-v1",
                api_docs_app="calendar",
            ),
        )
        agent = SmolagentsVLLMAgent("http://127.0.0.1:9")
        agent.begin(context, config)
        content = agent.messages()[1]["content"]
        self.assertIn("calendar.lookup: [documentation removed]", content)
        self.assertNotIn("start_time, end_time", content)

    def test_unknown_api_docs_version_is_rejected(self) -> None:
        config = replace(
            _config(),
            agent=replace(
                _config().agent,
                api_docs_version="api-docs-corrupt-does-not-exist",
                api_docs_app="calendar",
            ),
        )
        agent = SmolagentsVLLMAgent("http://127.0.0.1:9")
        agent.begin(_context(), config)
        with self.assertRaisesRegex(RuntimeUnavailable, "unknown api_docs_version"):
            agent.messages()
