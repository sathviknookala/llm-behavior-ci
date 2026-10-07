from __future__ import annotations

import io
import json
import os
import tempfile
import unittest
import urllib.error
from datetime import datetime, timezone
from email.message import EmailMessage
from pathlib import Path
from unittest.mock import patch

from llm_behavior_ci.config import (
    MISSING_HASHED_LEAF,
    AgentConfiguration,
    AnthropicModelConfiguration,
    AnthropicSamplingSettings,
    ConfigError,
    ModelConfiguration,
    RunConfiguration,
    SamplingSettings,
    canonical_configuration_json,
    hashed_values,
    load_model_configuration,
    new_episode_identity,
    recorded_execution_seed,
    run_configuration_hash,
)
from llm_behavior_ci.records import (
    EpisodeResult,
    EvaluatorOutcome,
    LocalTaskRef,
)
from llm_behavior_ci.runtime.agent import (
    ANTHROPIC_MESSAGES_URL,
    SmolagentsAnthropicAgent,
    UnsupportedCapability,
)
from llm_behavior_ci.runtime.appworld import TaskContext
from llm_behavior_ci.runtime.episode import (
    EpisodeRejected,
    RuntimeUnavailable,
    build_runtime,
    is_live_runtime,
)
from llm_behavior_ci.runtime.prompts import render_system_text
from llm_behavior_ci.runtime.workflow import WorkflowControlledAgent

_ROOT = Path(__file__).resolve().parents[2]
_SONNET = _ROOT / "configs" / "models" / "claude_sonnet_5_5_spotify_capability.json"
_DIAGNOSTIC = (
    _ROOT / "configs" / "models" / "qwen3_32b_awq_spotify_short_horizon_diagnostic.json"
)
_VLLM_CAPABILITY = (
    _ROOT / "configs" / "models" / "qwen3_32b_awq_spotify_capability_v2_interface.json"
)
_TASK_SET = _ROOT / "configs" / "tasks" / "train_spotify_capability.json"
_API_KEY = "sk-ant-test-secret-value"
_TASK_HASH = "a20fe52d28164e1d458266331c242277788d2ed0af29b054b7df926345db3a04"
_GIT = "a" * 40
_START = datetime(2026, 10, 6, 16, 0, tzinfo=timezone.utc)
_ENVELOPE = json.dumps(
    {
        "plan": ["List playlists", "Finish the task"],
        "active_step": 1,
        "action": "apis.spotify.show_playlist_library()",
    }
)
_VLLM_CONTROLS = ("temperature", "top_p", "top_k", "min_p", "seed")
_UNSUPPORTED_REQUEST_KEYS = (
    "temperature",
    "top_p",
    "top_k",
    "min_p",
    "seed",
    "logprobs",
    "top_logprobs",
    "chat_template_kwargs",
    "return_token_ids",
    "return_tokens_as_token_ids",
    "extra_body",
)


def _task() -> dict[str, object]:
    return {
        "appworld_version": "0.1.3.post1",
        "split": "train",
        "selection_rule": "fixed_spotify_capability",
        "selection_seed": 17,
        "task_count": 20,
        "task_set_hash": _TASK_HASH,
        "appworld_setup_profile": "spotify_authenticated_v1",
    }


def _sonnet_document() -> dict[str, object]:
    document = json.loads(_SONNET.read_text(encoding="utf-8"))
    document["task"] = _task()
    document["run_seed"] = 17
    document["git_commit"] = _GIT
    document["protocol_hash"] = None
    return document


def _sonnet_config() -> RunConfiguration:
    return RunConfiguration.from_dict(_sonnet_document())


def _context() -> TaskContext:
    return TaskContext(
        task_id="task-1",
        instruction="List the playlists",
        api_documentation="spotify.show_playlist_library: list playlists",
    )


class _Response:
    def __init__(self, raw: bytes) -> None:
        self._raw = raw

    def read(self) -> bytes:
        return self._raw

    def __enter__(self) -> _Response:
        return self

    def __exit__(self, *args: object) -> bool:
        del args
        return False


def _http_error(status: int, reason: str, body: bytes) -> urllib.error.HTTPError:
    return urllib.error.HTTPError(
        ANTHROPIC_MESSAGES_URL,
        status,
        reason,
        EmailMessage(),
        io.BytesIO(body),
    )


def _message(text: str, *, output_tokens: int = 37) -> dict[str, object]:
    return {
        "stop_reason": "end_turn",
        "content": [
            {"type": "thinking", "thinking": "hidden-thinking-block"},
            {"type": "text", "text": text},
        ],
        "usage": {"input_tokens": 12, "output_tokens": output_tokens},
    }


class AnthropicConfigurationTests(unittest.TestCase):
    def test_sonnet_configuration_round_trips_and_hashes_deterministically(self) -> None:
        original = _sonnet_config()
        self.assertIsInstance(original.model, AnthropicModelConfiguration)
        self.assertIsInstance(original.agent.sampling, AnthropicSamplingSettings)
        restored = RunConfiguration.from_dict(json.loads(canonical_configuration_json(original)))
        self.assertEqual(restored, original)
        self.assertEqual(run_configuration_hash(original), run_configuration_hash(restored))
        self.assertEqual(run_configuration_hash(original), run_configuration_hash(_sonnet_config()))
        sampling = original.agent.sampling.to_dict()
        self.assertEqual(
            set(sampling),
            {"max_tokens", "plan_max_tokens", "execute_max_tokens"},
        )
        self.assertEqual(sampling["execute_max_tokens"], 192)
        for control in _VLLM_CONTROLS:
            self.assertNotIn(control, sampling)
        model = original.model.to_dict()
        self.assertEqual(model["provider"], "anthropic")
        self.assertEqual(model["model_id"], "claude-sonnet-5-5")
        self.assertEqual(model["api_version"], "2023-06-01")
        self.assertEqual(model["thinking_mode"], "between_tools")
        self.assertEqual(model["effort"], "medium")
        self.assertNotIn("repository", json.dumps(model))
        self.assertNotIn("vllm", json.dumps(model))
        self.assertNotIn(_API_KEY, canonical_configuration_json(original))
        self.assertNotIn("api_key", canonical_configuration_json(original))
        self.assertEqual(recorded_execution_seed(original), 17)
        self.assertEqual(original.run_seed, 17)

    def test_anthropic_identity_changes_the_hash_and_vllm_leaves_stay_unset(self) -> None:
        base = _sonnet_config()
        base_hash = run_configuration_hash(base)
        values = hashed_values(base)
        self.assertEqual(values["model.provider"], "anthropic")
        self.assertEqual(values["model.model_id"], "claude-sonnet-5-5")
        self.assertEqual(values["model.api_version"], "2023-06-01")
        self.assertEqual(values["model.thinking_mode"], "between_tools")
        self.assertEqual(values["model.effort"], "medium")
        for path in (
            "model.model.repository",
            "model.serving.dtype",
            "model.vllm_version",
            "model.lora.repository",
            "agent.sampling.temperature",
            "agent.sampling.top_p",
            "agent.sampling.top_k",
            "agent.sampling.min_p",
            "agent.sampling.seed",
        ):
            self.assertIs(values[path], MISSING_HASHED_LEAF)
        changes = {
            "model_id": "claude-sonnet-other",
            "effort": "high",
            "thinking_mode": "adaptive",
            "api_version": "2024-01-01",
        }
        for field, value in changes.items():
            with self.subTest(field=field):
                document = _sonnet_document()
                assert isinstance(document["model"], dict)
                document["model"][field] = value
                changed = RunConfiguration.from_dict(document)
                self.assertNotEqual(run_configuration_hash(changed), base_hash)
                self.assertEqual(hashed_values(changed)[f"model.{field}"], value)

    def test_between_tools_rejects_effort_above_high(self) -> None:
        document = _sonnet_document()
        assert isinstance(document["model"], dict)
        document["model"]["effort"] = "xhigh"
        with self.assertRaises(ConfigError):
            RunConfiguration.from_dict(document)

    def test_sampling_controls_are_all_or_nothing(self) -> None:
        document = _sonnet_document()
        assert isinstance(document["agent"], dict)
        sampling = document["agent"]["sampling"]
        assert isinstance(sampling, dict)
        sampling["temperature"] = 0.0
        with self.assertRaises(ConfigError):
            RunConfiguration.from_dict(document)
        vllm = json.loads(_VLLM_CAPABILITY.read_text(encoding="utf-8"))
        vllm["task"] = _task()
        vllm["run_seed"] = 17
        vllm["git_commit"] = _GIT
        loaded = RunConfiguration.from_dict(vllm)
        self.assertIsInstance(loaded.model, ModelConfiguration)
        self.assertIsInstance(loaded.agent.sampling, SamplingSettings)
        del vllm["agent"]["sampling"]["temperature"]
        with self.assertRaises(ConfigError):
            RunConfiguration.from_dict(vllm)

    def test_vllm_model_rejects_anthropic_sampling_and_the_reverse(self) -> None:
        mixed = _sonnet_document()
        assert isinstance(mixed["agent"], dict)
        mixed["agent"]["sampling"] = json.loads(_VLLM_CAPABILITY.read_text(encoding="utf-8"))[
            "agent"
        ]["sampling"]
        with self.assertRaises(ConfigError):
            RunConfiguration.from_dict(mixed)
        vllm = json.loads(_VLLM_CAPABILITY.read_text(encoding="utf-8"))
        vllm["model"] = json.loads(_SONNET.read_text(encoding="utf-8"))["model"]
        vllm["task"] = _task()
        vllm["run_seed"] = 17
        vllm["git_commit"] = _GIT
        with self.assertRaises(ConfigError):
            RunConfiguration.from_dict(vllm)

    def test_diagnostic_configuration_remains_vllm(self) -> None:
        document = json.loads(_DIAGNOSTIC.read_text(encoding="utf-8"))
        model = load_model_configuration(document["model"])
        agent = AgentConfiguration.from_dict(document["agent"])
        self.assertIsInstance(model, ModelConfiguration)
        self.assertEqual(agent.sampling.temperature, 0.0)
        self.assertEqual(agent.sampling.seed, 17)
        self.assertEqual(agent.sampling.execute_max_tokens, 192)
        self.assertEqual(agent.prompt.prompt_version, "prompt-runtime-auth-v2")
        self.assertEqual(agent.workflow.policy, "plan_progress_v2")
        self.assertEqual(agent.execute_max_model_turns, 20)


class AnthropicRequestTests(unittest.TestCase):
    def _agent(self) -> SmolagentsAnthropicAgent:
        agent = SmolagentsAnthropicAgent(_API_KEY)
        agent.set_mode("execute")
        agent.begin(_context(), _sonnet_config())
        return agent

    def test_request_preserves_prompt_content_and_omits_sampling_controls(self) -> None:
        agent = self._agent()
        config = _sonnet_config()
        captured: list[object] = []

        def urlopen(request: object, timeout: object = None) -> _Response:
            del timeout
            captured.append(request)
            return _Response(json.dumps(_message(_ENVELOPE)).encode("utf-8"))

        with patch("urllib.request.urlopen", side_effect=urlopen):
            turn = agent.generate_turn(
                tool_output=None,
                extra_instruction="controller instruction",
                parse_action=False,
            )
        self.assertEqual(len(captured), 1)
        request = captured[0]
        self.assertEqual(request.full_url, ANTHROPIC_MESSAGES_URL)
        self.assertEqual(request.get_header("X-api-key"), _API_KEY)
        self.assertEqual(request.get_header("Anthropic-version"), "2023-06-01")
        raw = request.data.decode("utf-8")
        self.assertNotIn(_API_KEY, raw)
        payload = json.loads(raw)
        self.assertEqual(payload["model"], "claude-sonnet-5-5")
        self.assertEqual(payload["max_tokens"], 192)
        self.assertEqual(payload["thinking"], {"type": "between_tools"})
        self.assertEqual(payload["output_config"], {"effort": "medium"})
        expected_system = render_system_text(
            prompt_version=config.agent.prompt.prompt_version,
            plan_format_version=config.agent.prompt.plan_format_version,
            thinking_enabled=False,
            action_interface="code",
            mode="execute",
        )
        self.assertEqual(payload["system"], expected_system)
        self.assertIn("List the playlists", payload["messages"][0]["content"])
        self.assertIn("show_playlist_library", payload["messages"][0]["content"])
        self.assertEqual(payload["messages"][-1]["content"], "controller instruction")
        for key in _UNSUPPORTED_REQUEST_KEYS:
            self.assertNotIn(key, payload)
        self.assertEqual(turn.output_text, _ENVELOPE)
        self.assertNotIn("hidden-thinking-block", turn.output_text)
        self.assertEqual(turn.generated_token_count, 37)
        self.assertEqual(turn.top_k_logprobs, ())
        self.assertNotIn(_API_KEY, repr(agent))

        def second(request: object, timeout: object = None) -> _Response:
            del timeout
            captured.append(request)
            return _Response(json.dumps(_message("second", output_tokens=4)).encode("utf-8"))

        with patch("urllib.request.urlopen", side_effect=second):
            agent.generate_turn(tool_output="OBSERVATION-TOKEN", parse_action=False)
        second_payload = json.loads(captured[-1].data.decode("utf-8"))
        contents = [message["content"] for message in second_payload["messages"]]
        self.assertIn("OBSERVATION-TOKEN", contents)
        self.assertTrue(
            any(
                isinstance(item, str)
                and "List the playlists" in item
                and "show_playlist_library" in item
                for item in contents
            )
        )
        assistant = next(item for item in contents if isinstance(item, list))
        self.assertEqual(assistant[0]["type"], "thinking")
        self.assertEqual(assistant[0]["thinking"], "hidden-thinking-block")
        self.assertEqual(assistant[1]["text"], _ENVELOPE)
        self.assertEqual(second_payload["system"], expected_system)
        self.assertNotIn(_API_KEY, json.dumps(second_payload))

    def test_workflow_receives_visible_text_and_provider_token_count(self) -> None:
        config = _sonnet_config()
        agent = SmolagentsAnthropicAgent(_API_KEY)
        agent.set_mode("execute")
        controller = WorkflowControlledAgent(agent, config.agent.workflow)
        controller.begin(_context(), config)

        def urlopen(request: object, timeout: object = None) -> _Response:
            del timeout
            payload = json.loads(request.data.decode("utf-8"))
            rendered = json.dumps(payload)
            self.assertIn("plan_progress_v2", rendered)
            self.assertIn("List the playlists", rendered)
            self.assertEqual(payload["max_tokens"], 192)
            return _Response(json.dumps(_message(_ENVELOPE, output_tokens=41)).encode("utf-8"))

        with patch("urllib.request.urlopen", side_effect=urlopen):
            turn = controller.next_turn(tool_output=None)
        self.assertEqual(turn.action, "apis.spotify.show_playlist_library()")
        self.assertEqual(turn.app_name, "spotify")
        self.assertEqual(turn.api_name, "show_playlist_library")
        self.assertIsNone(turn.rejection)
        self.assertEqual(turn.generated_token_count, 41)
        self.assertNotIn("hidden-thinking-block", turn.output_text)
        self.assertEqual(turn.top_k_logprobs, ())

    def test_http_errors_are_runtime_failures_without_the_secret(self) -> None:
        cases = (
            (401, "Unauthorized", b'{"error":"invalid x-api-key sk-ant-test-secret-value"}', None, 1),
            (429, "Too Many Requests", b"rate limited sk-ant-test-secret-value", None, 5),
            (500, "Internal Server Error", b"upstream failed", None, 5),
            (400, "Bad Request", b"prompt is too long for the context window", "context_length_exceeded", 1),
        )
        for status, reason, body, expected_reason, calls in cases:
            with self.subTest(status=status):
                agent = self._agent()
                seen = {"count": 0}

                def urlopen(request: object, timeout: object = None) -> _Response:
                    del request, timeout
                    seen["count"] += 1
                    raise _http_error(status, reason, body)

                with (
                    patch("urllib.request.urlopen", side_effect=urlopen),
                    patch("llm_behavior_ci.runtime.agent.time.sleep"),
                ):
                    with self.assertRaises(RuntimeUnavailable) as caught:
                        agent.generate_turn(tool_output=None, parse_action=False)
                self.assertEqual(seen["count"], calls)
                text = str(caught.exception)
                self.assertIn(f"HTTP {status}", text)
                self.assertNotIn(_API_KEY, text)
                self.assertEqual(caught.exception.reason, expected_reason)

    def test_malformed_responses_fail_without_the_secret(self) -> None:
        agent = self._agent()
        bodies = (
            b"not-json sk-ant-test-secret-value",
            json.dumps({"content": "text", "usage": {"output_tokens": 1}}).encode("utf-8"),
            json.dumps(
                {"content": [{"type": "text", "text": "ok"}], "usage": {}}
            ).encode("utf-8"),
        )
        for body in bodies:
            with self.subTest(body=body[:24]):
                def urlopen(request: object, timeout: object = None) -> _Response:
                    del request, timeout
                    return _Response(body)

                with patch("urllib.request.urlopen", side_effect=urlopen):
                    with self.assertRaises(RuntimeUnavailable) as caught:
                        agent.generate_turn(tool_output=None, parse_action=False)
                self.assertIn("malformed", str(caught.exception))
                self.assertNotIn(_API_KEY, str(caught.exception))

    def test_context_stop_reason_is_a_runtime_failure(self) -> None:
        agent = self._agent()

        def urlopen(request: object, timeout: object = None) -> _Response:
            del request, timeout
            return _Response(
                json.dumps(
                    {
                        "stop_reason": "model_context_window_exceeded",
                        "content": [{"type": "text", "text": "partial"}],
                        "usage": {"output_tokens": 3},
                    }
                ).encode("utf-8")
            )

        with patch("urllib.request.urlopen", side_effect=urlopen):
            with self.assertRaises(RuntimeUnavailable) as caught:
                agent.generate_turn(tool_output=None, parse_action=False)
        self.assertEqual(caught.exception.reason, "context_length_exceeded")
        self.assertNotIn(_API_KEY, str(caught.exception))

    def test_teacher_forced_scoring_is_unsupported(self) -> None:
        agent = self._agent()
        with patch("urllib.request.urlopen") as urlopen:
            with self.assertRaises(UnsupportedCapability) as caught:
                agent.teacher_force_plan(
                    messages=[{"role": "user", "content": "plan"}],
                    plan_text="1. list playlists",
                )
        urlopen.assert_not_called()
        self.assertIn("unsupported", str(caught.exception))
        self.assertNotIn("token_id", str(caught.exception))

    def test_missing_key_fails_during_runtime_construction(self) -> None:
        config = _sonnet_config()
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": ""}, clear=False):
            with self.assertRaises(EpisodeRejected) as caught:
                build_runtime(config, mode="execute")
        self.assertIn("ANTHROPIC_API_KEY is required", str(caught.exception))
        self.assertNotIn(_API_KEY, str(caught.exception))
        with self.assertRaises(EpisodeRejected):
            SmolagentsAnthropicAgent("  ")
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": _API_KEY}, clear=False):
            with self.assertRaises(EpisodeRejected):
                build_runtime(config, "http://127.0.0.1:8000", mode="execute")
            runtime = build_runtime(config, mode="execute")
        self.assertIsInstance(runtime.agent, WorkflowControlledAgent)
        self.assertIsInstance(runtime.agent.underlying_agents()[0], SmolagentsAnthropicAgent)
        self.assertTrue(is_live_runtime(runtime))
        self.assertNotIn(_API_KEY, repr(runtime.agent))
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": _API_KEY}, clear=False):
            plan_runtime = build_runtime(config, mode="plan")
        self.assertIsInstance(plan_runtime.agent, SmolagentsAnthropicAgent)
        self.assertTrue(is_live_runtime(plan_runtime))


def _pilot():
    import importlib.util

    path = _ROOT / "scripts" / "evaluation" / "run_capability_pilot.py"
    spec = importlib.util.spec_from_file_location("anthropic_capability_pilot", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fake_task_load(payload: object):
    del payload
    from llm_behavior_ci.config import TaskConfiguration
    from llm_behavior_ci.tasks.selection import TaskSet

    document = json.loads(_TASK_SET.read_text(encoding="utf-8"))
    task = TaskConfiguration.from_dict(
        {
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
    )
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


def _completed_episode(task_id, config, **kwargs) -> EpisodeResult:
    identity = new_episode_identity(kwargs["run"])
    kwargs["on_start"](identity, kwargs["run"])
    return EpisodeResult(
        episode=identity,
        run=kwargs["run"],
        task=LocalTaskRef(
            task_id=task_id,
            scenario_id=kwargs["scenario_id"],
            split=config.task.split,
        ),
        mode="execute",
        execution_seed=recorded_execution_seed(config),
        status="completed",
        started_at=_START,
        ended_at=_START,
        model_steps=(),
        tool_steps=(),
        plan_text=None,
        evaluator_outcome=EvaluatorOutcome(
            success=False,
            passed_requirements=0,
            total_requirements=2,
            difficulty=1,
        ),
        termination_reason="appworld_completed",
        episode_errors=(),
        role=None,
    )


class AnthropicPilotTests(unittest.TestCase):
    def _argv(self, directory: Path, configuration: Path, *, base_url: str | None) -> list[str]:
        argv = [
            "--configuration",
            str(configuration),
            "--task-set",
            str(_TASK_SET),
            "--store",
            str(directory / "episodes.sqlite"),
            "--output",
            str(directory / "aggregate.json"),
        ]
        if base_url is not None:
            argv.extend(["--base-url", base_url])
        return argv

    def test_vllm_pilot_still_requires_an_endpoint(self) -> None:
        pilot = _pilot()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            previous = os.environ.get("APPWORLD_ROOT")
            os.environ["APPWORLD_ROOT"] = str(root)
            try:
                with patch.object(pilot.sys, "stderr", io.StringIO()):
                    code = pilot.main(self._argv(root, _DIAGNOSTIC, base_url=None))
            finally:
                if previous is None:
                    os.environ.pop("APPWORLD_ROOT", None)
                else:
                    os.environ["APPWORLD_ROOT"] = previous
        self.assertEqual(code, 2)

    def test_anthropic_pilot_rejects_a_local_endpoint(self) -> None:
        pilot = _pilot()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            previous = os.environ.get("APPWORLD_ROOT")
            os.environ["APPWORLD_ROOT"] = str(root)
            try:
                with patch.object(pilot.sys, "stderr", io.StringIO()):
                    code = pilot.main(
                        self._argv(root, _SONNET, base_url="http://127.0.0.1:8000")
                    )
            finally:
                if previous is None:
                    os.environ.pop("APPWORLD_ROOT", None)
                else:
                    os.environ["APPWORLD_ROOT"] = previous
        self.assertEqual(code, 2)

    def test_missing_api_key_aborts_before_any_task(self) -> None:
        pilot = _pilot()
        calls: list[str] = []

        def run(*args, **kwargs):
            del args, kwargs
            calls.append("ran")
            raise AssertionError("task 0 started")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stderr = io.StringIO()
            with (
                patch.object(pilot, "enforce_committed_provenance", lambda config: None),
                patch.object(pilot, "_load_task_set", _fake_task_load),
                patch.object(pilot, "_adopt_committed_setup_profile", lambda task: task),
                patch.object(pilot, "run_episode", side_effect=run),
                patch.object(pilot.sys, "stderr", stderr),
                patch.dict(os.environ, {"ANTHROPIC_API_KEY": ""}, clear=False),
            ):
                previous = os.environ.get("APPWORLD_ROOT")
                os.environ["APPWORLD_ROOT"] = str(root)
                try:
                    code = pilot.main(self._argv(root, _SONNET, base_url=None))
                finally:
                    if previous is None:
                        os.environ.pop("APPWORLD_ROOT", None)
                    else:
                        os.environ["APPWORLD_ROOT"] = previous
            payload = json.loads((root / "aggregate.json").read_text(encoding="utf-8"))
        self.assertEqual(code, 1)
        self.assertEqual(calls, [])
        self.assertEqual(payload["run_status"], "aborted")
        self.assertEqual(payload["not_attempted_count"], 20)
        self.assertEqual(payload["episodes_attempted"], 0)
        self.assertIn("ANTHROPIC_API_KEY is required", payload["failure_reason"])
        self.assertNotIn(_API_KEY, json.dumps(payload))
        self.assertNotIn(_API_KEY, stderr.getvalue())

    def test_one_episode_local_error_continues_the_twenty_task_pilot(self) -> None:
        pilot = _pilot()
        calls: list[str] = []

        def run(task_id, config, mode, **kwargs):
            del mode
            calls.append(task_id)
            self.assertIs(kwargs["evaluate_after_runtime_failure"], True)
            self.assertIsInstance(config.model, AnthropicModelConfiguration)
            self.assertEqual(recorded_execution_seed(config), 17)
            if len(calls) == 1:
                raise RuntimeUnavailable(
                    "Anthropic request failed with HTTP 401 Unauthorized:\n"
                    "invalid x-api-key [redacted]"
                )
            return _completed_episode(task_id, config, **kwargs)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stderr = io.StringIO()
            with (
                patch.object(pilot, "enforce_committed_provenance", lambda config: None),
                patch.object(pilot, "_load_task_set", _fake_task_load),
                patch.object(pilot, "_adopt_committed_setup_profile", lambda task: task),
                patch.object(pilot, "run_episode", side_effect=run),
                patch.object(pilot, "build_runtime", return_value=object()),
                patch.object(pilot.sys, "stderr", stderr),
                patch("builtins.print"),
                patch.dict(os.environ, {"ANTHROPIC_API_KEY": _API_KEY}, clear=False),
            ):
                previous = os.environ.get("APPWORLD_ROOT")
                os.environ["APPWORLD_ROOT"] = str(root)
                try:
                    code = pilot.main(self._argv(root, _SONNET, base_url=None))
                finally:
                    if previous is None:
                        os.environ.pop("APPWORLD_ROOT", None)
                    else:
                        os.environ["APPWORLD_ROOT"] = previous
            text = stderr.getvalue()
            payload = json.loads((root / "aggregate.json").read_text(encoding="utf-8"))
        self.assertEqual(code, 0)
        self.assertEqual(len(calls), 20)
        self.assertNotIn("capability pilot failed", text)
        self.assertNotIn(_API_KEY, text)
        self.assertNotIn(_API_KEY, json.dumps(payload))
        self.assertEqual(payload["run_status"], "completed")
        self.assertEqual(payload["task_count"], 20)
        self.assertEqual(payload["runtime_failure_count"], 1)
        self.assertEqual(payload["not_attempted_count"], 0)
        self.assertEqual(payload["terminal_status_counts"]["runtime_failure"], 1)
        self.assertEqual(payload["terminal_status_counts"]["evaluator_failure"], 19)
        self.assertEqual(payload["terminal_status_counts"]["not_attempted"], 0)


if __name__ == "__main__":
    unittest.main()
