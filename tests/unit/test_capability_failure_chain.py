import importlib.util
import io
import json
import os
import tempfile
import unittest
import urllib.error
from datetime import datetime, timedelta, timezone
from email.message import EmailMessage
from pathlib import Path
from unittest.mock import patch

from llm_behavior_ci.config import RunConfiguration, new_episode_identity, new_run_identity
from llm_behavior_ci.records import (
    EpisodeResult,
    LocalTaskRef,
    ModelStep,
    RecordedError,
    TokenLogprob,
)
from llm_behavior_ci.runtime.agent import SmolagentsVLLMAgent
from llm_behavior_ci.runtime.appworld import EvaluationResult, TaskContext, ToolResult
from llm_behavior_ci.runtime.episode import (
    RuntimeDependencies,
    RuntimeUnavailable,
    run_episode,
)
from llm_behavior_ci.storage import EpisodeStore

_ROOT = Path(__file__).resolve().parents[2]
_START = datetime(2026, 10, 4, 12, 0, tzinfo=timezone.utc)
_SECRET = "Bearer supervisor-password-secret"
_CONTEXT_BODY = (
    "This model's maximum context length is 22528 tokens. However, you "
    "requested 192 output tokens and your prompt contains at least 22337 "
    "input tokens, for a total of at least 22529 tokens. Please reduce the "
    "length of the input prompt. (parameter=input_tokens, value=22337)"
)


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
                "max_num_seqs": 1,
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


def _clock():
    current = _START

    def tick() -> datetime:
        nonlocal current
        value = current
        current = current + timedelta(minutes=1)
        return value

    return tick


def _http_error(status: int, reason: str, body: bytes, *, secret: bool = False):
    headers = EmailMessage()
    if secret:
        headers["Authorization"] = _SECRET
    return urllib.error.HTTPError(
        "http://127.0.0.1:9/v1/chat/completions",
        status,
        reason,
        headers,
        io.BytesIO(body),
    )


class _Unreadable:
    def read(self, size: int = -1) -> bytes:
        del size
        raise OSError("body closed")

    def close(self) -> None:
        return None


class HttpErrorBodyTests(unittest.TestCase):
    def _post(self, error: BaseException) -> RuntimeUnavailable:
        agent = SmolagentsVLLMAgent("http://127.0.0.1:9")
        with patch("urllib.request.urlopen", side_effect=error):
            with self.assertRaises(RuntimeUnavailable) as caught:
                agent._post({"messages": []})
        return caught.exception

    def test_context_length_body_is_retained_and_classified(self) -> None:
        error = self._post(
            _http_error(400, "Bad Request", _CONTEXT_BODY.encode(), secret=True)
        )
        text = str(error)
        self.assertIn("HTTP 400 Bad Request", text)
        self.assertIn("at least 22337 input tokens", text)
        self.assertIn("endpoint: http://127.0.0.1:9/v1/chat/completions", text)
        self.assertNotIn(_SECRET, text)
        self.assertNotIn("Authorization", text)
        self.assertEqual(error.reason, "context_length_exceeded")

    def test_http_500_body_is_retained_without_context_classification(self) -> None:
        error = self._post(_http_error(500, "Internal Server Error", b"engine exploded"))
        text = str(error)
        self.assertIn("HTTP 500 Internal Server Error", text)
        self.assertIn("engine exploded", text)
        self.assertIsNone(error.reason)

    def test_empty_and_unreadable_bodies_stay_safe(self) -> None:
        empty = self._post(_http_error(400, "Bad Request", b"  "))
        self.assertIn("<empty body>", str(empty))
        self.assertIsNone(empty.reason)
        headers = EmailMessage()
        headers["Authorization"] = _SECRET
        unreadable = urllib.error.HTTPError(
            "http://127.0.0.1:9/v1/chat/completions",
            400,
            "Bad Request",
            headers,
            _Unreadable(),
        )
        error = self._post(unreadable)
        self.assertIn("unreadable body", str(error))
        self.assertIn("OSError", str(error))
        self.assertNotIn(_SECRET, str(error))
        self.assertIsNone(error.reason)

    def test_plain_context_phrase_is_classified(self) -> None:
        error = self._post(
            _http_error(400, "Bad Request", b"context length exceeds maximum")
        )
        self.assertEqual(error.reason, "context_length_exceeded")
        self.assertIn("context length exceeds maximum", str(error))


class _Session:
    def __init__(self) -> None:
        self.close_count = 0
        self.evaluate_count = 0
        self.actions: list[str] = []

    def context(self) -> TaskContext:
        return TaskContext(
            task_id="task-1",
            instruction="solve the task",
            api_documentation="docs",
        )

    def execute(self, action: str) -> ToolResult:
        self.actions.append(action)
        return ToolResult(
            output_text="page of results",
            error_message=None,
            recoverable=False,
            app_name="calendar",
            api_name="lookup",
        )

    def evaluate(self) -> EvaluationResult:
        self.evaluate_count += 1
        return EvaluationResult(
            success=True,
            passed_requirements=1,
            total_requirements=1,
            difficulty=1,
        )

    def close(self) -> None:
        self.close_count += 1


class _Agent:
    def __init__(self, clock) -> None:
        self._clock = clock
        self.calls = 0

    def begin(self, context: TaskContext, config: RunConfiguration) -> None:
        del context, config

    def next_turn(self, *, tool_output: str | None):
        del tool_output
        self.calls += 1
        if self.calls > 1:
            raise RuntimeUnavailable(
                "vLLM request failed with HTTP 400 Bad Request:\n"
                "context length exceeds maximum",
                reason="context_length_exceeded",
            )
        from llm_behavior_ci.runtime.agent import AgentTurn

        return AgentTurn(
            prompt_text="next action",
            output_text="calendar.lookup()",
            top_k_logprobs=(),
            generated_token_count=4,
            latency_seconds=0.2,
            started_at=self._clock(),
            action="calendar.lookup()",
            app_name="calendar",
            api_name="lookup",
        )


class EpisodeRuntimeFailureTests(unittest.TestCase):
    def test_generation_failure_closes_without_an_evaluator_outcome(self) -> None:
        config = _config()
        clock = _clock()
        session = _Session()
        with tempfile.TemporaryDirectory() as directory:
            store = EpisodeStore(Path(directory) / "episodes.sqlite")
            seen: dict[str, str] = {}

            def on_start(identity, run) -> None:
                store.start_episode(identity, run, "task-1")
                seen["episode_id"] = identity.episode_id

            def on_step(step) -> None:
                store.append_step(seen["episode_id"], step)

            result = run_episode(
                "task-1",
                config,
                "execute",
                run=new_run_identity(config),
                runtime=RuntimeDependencies(
                    session_factory=lambda task_id: session,
                    agent=_Agent(clock),
                    clock=clock,
                ),
                on_start=on_start,
                on_step=on_step,
            )
            self.assertEqual(session.close_count, 1)
            self.assertEqual(session.evaluate_count, 0)
            self.assertIsNone(result.evaluator_outcome)
            self.assertEqual(result.termination_reason, "runtime_error")
            self.assertEqual(result.status, "failed")
            self.assertEqual(len(result.model_steps), 1)
            self.assertEqual(len(result.tool_steps), 1)
            self.assertIn("context_length_exceeded", result.episode_errors[0].message)
            self.assertIn("context length exceeds maximum", result.episode_errors[0].message)
            store.finish_episode(result)
            loaded = store.load_episode(result.episode.episode_id)
            self.assertEqual(loaded.termination_reason, "runtime_error")
            self.assertIsNone(loaded.evaluator_outcome)
            self.assertEqual(len(loaded.model_steps), 1)
            self.assertEqual(loaded.model_steps[0].output_text, "calendar.lookup()")
            self.assertEqual(loaded.tool_steps[0].output_text, "page of results")
            with self.assertRaises(Exception):
                store.load_open_episode(result.episode.episode_id)


def _pilot():
    path = _ROOT / "scripts/evaluation/run_capability_pilot.py"
    spec = importlib.util.spec_from_file_location("capability_failure_pilot", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fake_task_load(payload: object):
    del payload
    document = json.loads(
        (_ROOT / "configs/tasks/train_spotify_capability.json").read_text(encoding="utf-8")
    )
    from llm_behavior_ci.config import TaskConfiguration
    from llm_behavior_ci.tasks.selection import TaskSet

    fields = {
        key: document[key]
        for key in (
            "appworld_version",
            "split",
            "selection_rule",
            "selection_seed",
            "task_count",
            "task_set_hash",
            "appworld_setup_profile",
        )
    }
    task = TaskConfiguration.from_dict(fields)
    task_ids = tuple(f"task-{index}" for index in range(20))
    scenario_ids = tuple(f"scenario-{index}" for index in range(20))
    task_set = TaskSet(
        appworld_version=task.appworld_version,
        split=task.split,
        selection_rule=task.selection_rule,
        selection_seed=task.selection_seed,
        task_count=20,
        scenario_count=20,
        task_ids=task_ids,
        scenario_ids=scenario_ids,
        task_set_hash=task.task_set_hash,
    )
    return task, task_set


def _pilot_argv(directory: Path) -> list[str]:
    return [
        "--configuration",
        str(_ROOT / "configs/models/qwen3_32b_awq_spotify_capability_v2_interface.json"),
        "--task-set",
        str(_ROOT / "configs/tasks/train_spotify_capability.json"),
        "--base-url",
        "http://127.0.0.1:9",
        "--store",
        str(directory / "episodes.sqlite"),
        "--output",
        str(directory / "aggregate.json"),
    ]


class PilotFailureTests(unittest.TestCase):
    def _run(self, directory: Path, side_effect):
        pilot = _pilot()
        stderr = io.StringIO()
        with (
            patch.object(pilot, "enforce_committed_provenance", lambda config: None),
            patch.object(pilot, "_load_task_set", _fake_task_load),
            patch.object(pilot, "_adopt_committed_setup_profile", lambda task: task),
            patch.object(pilot, "run_episode", side_effect=side_effect),
            patch.object(pilot.sys, "stderr", stderr),
        ):
            previous = os.environ.get("APPWORLD_ROOT")
            os.environ["APPWORLD_ROOT"] = str(directory)
            try:
                code = pilot.main(_pilot_argv(directory))
            finally:
                if previous is None:
                    os.environ.pop("APPWORLD_ROOT", None)
                else:
                    os.environ["APPWORLD_ROOT"] = previous
        return code, stderr.getvalue(), directory / "aggregate.json"

    def test_system_exit_codes_are_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            code, text, output = self._run(root, SystemExit(1))
            self.assertEqual(code, 1)
            self.assertNotIn("AttributeError", text)
            self.assertFalse(output.exists())
            code, text, output = self._run(root, SystemExit("message"))
            self.assertEqual(code, 1)
            self.assertNotIn("AttributeError", text)
            self.assertNotIn("invalid literal", text)
            self.assertFalse(output.exists())

    def test_raised_runtime_failure_writes_an_aborted_aggregate(self) -> None:
        message = (
            "vLLM request failed with HTTP 400 Bad Request:\n"
            "context length exceeds maximum"
        )

        def fail(*args, **kwargs):
            del args, kwargs
            raise RuntimeUnavailable(message, reason="context_length_exceeded")

        with tempfile.TemporaryDirectory() as directory:
            code, text, output = self._run(Path(directory), fail)
            self.assertEqual(code, 1)
            self.assertIn("capability pilot failed", text)
            self.assertIn("context length exceeds maximum", text)
            self.assertNotIn("AttributeError", text)
            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(payload["pilot_status"], "aborted")
            self.assertEqual(payload["episodes_expected"], 20)
            self.assertEqual(payload["episodes_attempted"], 1)
            self.assertEqual(payload["episodes_completed"], 0)
            self.assertEqual(payload["failed_episode"], 0)
            self.assertEqual(payload["failure_type"], "runtime_error")
            self.assertIn("context length exceeds maximum", payload["failure_reason"])
            self.assertIn("context_length_exceeded", payload["failure_reason"])
            self.assertNotEqual(payload["task_count"], 20)

    def test_generic_exception_is_not_replaced(self) -> None:
        def fail(*args, **kwargs):
            del args, kwargs
            raise RuntimeError("sampler backend disconnected")

        with tempfile.TemporaryDirectory() as directory:
            code, text, output = self._run(Path(directory), fail)
            self.assertEqual(code, 1)
            self.assertIn("capability pilot failed", text)
            self.assertIn("sampler backend disconnected", text)
            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(payload["pilot_status"], "aborted")
            self.assertIn("sampler backend disconnected", payload["failure_reason"])

    def test_returned_runtime_error_aborts_and_keeps_stored_steps(self) -> None:
        def fail(task_id, config, mode, *, run, runtime, on_start, on_step, scenario_id):
            del runtime, mode
            identity = new_episode_identity(run)
            on_start(identity, run)
            started = datetime(2026, 10, 4, 12, 0, tzinfo=timezone.utc)
            step = ModelStep(
                index=0,
                prompt_text="next action",
                output_text="calendar.lookup()",
                top_k_logprobs=(
                    (TokenLogprob(token_id=3, logprob=-0.2, rank=0),),
                ),
                generated_token_count=1,
                latency_seconds=0.1,
                started_at=started,
            )
            on_step(step)
            message = (
                "context_length_exceeded\n"
                "vLLM request failed with HTTP 400 Bad Request:\n"
                "context length exceeds maximum"
            )
            return EpisodeResult(
                episode=identity,
                run=run,
                task=LocalTaskRef(
                    task_id=task_id,
                    scenario_id=scenario_id,
                    split=config.task.split,
                ),
                mode="execute",
                execution_seed=config.agent.sampling.seed,
                status="failed",
                started_at=started,
                ended_at=started,
                model_steps=(step,),
                tool_steps=(),
                plan_text=None,
                evaluator_outcome=None,
                termination_reason="runtime_error",
                episode_errors=(
                    RecordedError(
                        source="runtime",
                        recoverable=False,
                        message=message,
                        step_index=None,
                    ),
                ),
                role=None,
            )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            code, text, output = self._run(root, fail)
            self.assertEqual(code, 1)
            self.assertIn("capability pilot failed", text)
            self.assertIn("context length exceeds maximum", text)
            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(payload["pilot_status"], "aborted")
            self.assertEqual(payload["episodes_attempted"], 1)
            self.assertEqual(payload["episodes_completed"], 0)
            self.assertEqual(payload["model_generation_count"], 1)
            self.assertIn("context length exceeds maximum", payload["failure_reason"])
            store = EpisodeStore(root / "episodes.sqlite")
            rows = store._connection.execute(
                "SELECT state FROM episodes"
            ).fetchall()
            self.assertEqual(rows, [("finished",)])
            loaded_id = store._connection.execute(
                "SELECT episode_id FROM episodes"
            ).fetchone()[0]
            loaded = store.load_episode(loaded_id)
            self.assertEqual(loaded.termination_reason, "runtime_error")
            self.assertIsNone(loaded.evaluator_outcome)
            self.assertEqual(len(loaded.model_steps), 1)

    def test_raised_after_a_stored_step_finalizes_the_open_episode(self) -> None:
        def fail(task_id, config, mode, *, run, runtime, on_start, on_step, scenario_id):
            del task_id, config, mode, runtime, scenario_id
            identity = new_episode_identity(run)
            on_start(identity, run)
            on_step(
                ModelStep(
                    index=0,
                    prompt_text="next action",
                    output_text="partial",
                    top_k_logprobs=(
                        (TokenLogprob(token_id=4, logprob=-0.3, rank=0),),
                    ),
                    generated_token_count=1,
                    latency_seconds=0.1,
                    started_at=datetime(2026, 10, 4, 12, 5, tzinfo=timezone.utc),
                )
            )
            raise RuntimeUnavailable(
                "vLLM request failed with HTTP 400 Bad Request:\n"
                "context length exceeds maximum",
                reason="context_length_exceeded",
            )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            code, text, output = self._run(root, fail)
            self.assertEqual(code, 1)
            self.assertIn("capability pilot failed", text)
            self.assertNotIn("AttributeError", text)
            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertIn("context length exceeds maximum", payload["failure_reason"])
            store = EpisodeStore(root / "episodes.sqlite")
            state = store._connection.execute("SELECT state FROM episodes").fetchone()[0]
            self.assertEqual(state, "finished")
            episode_id = store._connection.execute(
                "SELECT episode_id FROM episodes"
            ).fetchone()[0]
            loaded = store.load_episode(episode_id)
            self.assertEqual(loaded.status, "failed")
            self.assertEqual(loaded.termination_reason, "runtime_error")
            self.assertIsNone(loaded.evaluator_outcome)
            self.assertEqual(loaded.model_steps[0].output_text, "partial")


if __name__ == "__main__":
    unittest.main()
