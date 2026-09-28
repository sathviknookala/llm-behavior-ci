import builtins
import math
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from llm_behavior_ci.config import RunConfiguration
from llm_behavior_ci.runtime.agent import VLLMAgent, parse_model_output
from llm_behavior_ci.runtime.appworld import (
    LiveAppWorldSession,
    TaskContext,
    _open_appworld,
)
from llm_behavior_ci.runtime.episode import RuntimeUnavailable
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
        return SimpleNamespace(success=True, passes=["a"], fails=[], difficulty=2)

    def close(self) -> None:
        self.closed += 1


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

    def test_completion_payload_contains_sampling_prompt_and_thinking(self) -> None:
        agent = VLLMAgent("http://127.0.0.1:9")
        context = TaskContext(
            task_id="task-1",
            instruction="solve the task",
            api_documentation="docs",
        )
        agent.begin(context, _config())
        payload = agent.completion_payload(agent.messages())
        self.assertEqual(payload["temperature"], 0.0)
        self.assertEqual(payload["seed"], 17)
        extra = payload["extra_body"]
        self.assertEqual(extra["top_k"], 20)
        self.assertEqual(extra["min_p"], 0.0)
        self.assertFalse(extra["chat_template_kwargs"]["enable_thinking"])
        system = payload["messages"][0]["content"]
        self.assertIn("prompt_version=prompt-v1", system)

    def test_parse_model_output_stop_and_call(self) -> None:
        self.assertEqual(parse_model_output("STOP"), (None, None, None))
        self.assertEqual(parse_model_output("STOP\nmore"), (None, None, None))
        self.assertEqual(
            parse_model_output("CALL calendar lookup\napp.lookup()"),
            ("app.lookup()", "calendar", "lookup"),
        )
        self.assertEqual(parse_model_output("plain action"), ("plain action", None, None))
        self.assertEqual(
            VLLMAgent("http://127.0.0.1:9").parse_model_output("STOP"),
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
