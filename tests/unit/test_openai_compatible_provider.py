from __future__ import annotations

import copy
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

import llm_behavior_ci.config as config_module
from llm_behavior_ci.config import (
    MISSING_HASHED_LEAF,
    AgentConfiguration,
    AnthropicModelConfiguration,
    AnthropicSamplingSettings,
    ConfigError,
    HostedSamplingSettings,
    ModelConfiguration,
    OpenAICompatibleModelConfiguration,
    OpenAICompatibleSamplingSettings,
    RunConfiguration,
    SamplingSettings,
    canonical_configuration_json,
    hashed_values,
    hosted_provider,
    load_model_configuration,
    load_sampling_settings,
    new_episode_identity,
    recorded_execution_seed,
    run_configuration_hash,
)
from llm_behavior_ci.records import EpisodeResult, EvaluatorOutcome, LocalTaskRef
from llm_behavior_ci.runtime.agent import (
    SmolagentsOpenAICompatibleAgent,
    UnsupportedCapability,
)
from llm_behavior_ci.runtime.appworld import TaskContext
from llm_behavior_ci.runtime.episode import (
    EpisodeRejected,
    RuntimeDependencies,
    RuntimeUnavailable,
    build_runtime,
    is_live_runtime,
)
from llm_behavior_ci.runtime.prompts import render_system_text
from llm_behavior_ci.runtime.workflow import WorkflowControlledAgent

_ROOT = Path(__file__).resolve().parents[2]
_GLM = _ROOT / "configs" / "models" / "glm_5_3_spotify_capability.json"
_SONNET = _ROOT / "configs" / "models" / "claude_sonnet_5_5_spotify_capability.json"
_DIAGNOSTIC = (
    _ROOT / "configs" / "models" / "qwen3_32b_awq_spotify_short_horizon_diagnostic.json"
)
_DIAGNOSTIC_TASKS = _ROOT / "configs" / "tasks" / "train_spotify_short_horizon_diagnostic_6.json"
_VLLM_CAPABILITY = (
    _ROOT / "configs" / "models" / "qwen3_32b_awq_spotify_capability_v2_interface.json"
)
_TASK_SET = _ROOT / "configs" / "tasks" / "train_spotify_capability.json"
_API_KEY = "zai-test-secret-value-0123456789"
_ENDPOINT = "https://api.z.ai/api/paas/v4/chat/completions"
_TASK_HASH = "a20fe52d28164e1d458266331c242277788d2ed0af29b054b7df926345db3a04"
_SONNET_COMMIT = "93ecb4aa7394b375dcec637e5d8d4e9a50664951"
_SONNET_HASH = "1215704a42034a5a5a2c2bb99e4582019f8d936bb16469cbaf36cb45c0e20e57"
_DIAGNOSTIC_COMMIT = "bf52f870003a00913d6312e0428ea629427a40b2"
_DIAGNOSTIC_HASH = "b96adc02f3cc94ac75b4b80892fedc514fbf45a01855113b8f3ad9ff86018638"
_GIT = "a" * 40
_START = datetime(2026, 10, 6, 16, 0, tzinfo=timezone.utc)
_REASONING = "hidden-reasoning-content-marker"
_ENVELOPE = json.dumps(
    {
        "plan": ["List playlists", "Finish the task"],
        "active_step": 1,
        "action": "apis.spotify.show_playlist_library()",
    }
)
_PUBLIC_TASK_FIELDS = (
    "appworld_version",
    "split",
    "selection_rule",
    "selection_seed",
    "task_count",
    "task_set_hash",
    "appworld_setup_profile",
)


def _task(path: Path = _TASK_SET) -> dict[str, object]:
    document = json.loads(path.read_text(encoding="utf-8"))
    return {key: document[key] for key in _PUBLIC_TASK_FIELDS if key in document}


def _document(path: Path = _GLM, *, git_commit: str = _GIT, task_path: Path = _TASK_SET) -> dict[str, object]:
    document = json.loads(path.read_text(encoding="utf-8"))
    task = _task(task_path)
    document["task"] = task
    document["run_seed"] = task["selection_seed"]
    document["git_commit"] = git_commit
    document["protocol_hash"] = None
    return document


def _glm_config() -> RunConfiguration:
    return RunConfiguration.from_dict(_document())


def _context() -> TaskContext:
    return TaskContext(
        task_id="task-1",
        instruction="List the playlists",
        api_documentation="spotify.show_playlist_library: list playlists",
    )


def _completion(
    content: object,
    *,
    finish_reason: object = "stop",
    completion_tokens: object = 37,
    reasoning: str | None = _REASONING,
) -> dict[str, object]:
    message: dict[str, object] = {"role": "assistant", "content": content}
    if reasoning is not None:
        message["reasoning_content"] = reasoning
    return {
        "id": "chatcmpl-test",
        "choices": [{"index": 0, "finish_reason": finish_reason, "message": message}],
        "usage": {
            "prompt_tokens": 120,
            "completion_tokens": completion_tokens,
            "total_tokens": 157,
        },
    }


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


def _respond(payload: object):
    raw = payload if isinstance(payload, bytes) else json.dumps(payload).encode("utf-8")

    def urlopen(request: object, timeout: object = None) -> _Response:
        del request, timeout
        return _Response(raw)

    return urlopen


def _http_error(status: int, reason: str, body: bytes) -> urllib.error.HTTPError:
    return urllib.error.HTTPError(_ENDPOINT, status, reason, EmailMessage(), io.BytesIO(body))


def _agent(config: RunConfiguration | None = None) -> SmolagentsOpenAICompatibleAgent:
    agent = SmolagentsOpenAICompatibleAgent("zai", _API_KEY)
    agent.set_mode("execute")
    agent.begin(_context(), config or _glm_config())
    return agent


def _with_model(field: str, value: object) -> dict[str, object]:
    document = _document()
    assert isinstance(document["model"], dict)
    document["model"][field] = value
    return document


def _with_sampling(**changes: object) -> dict[str, object]:
    document = _document()
    assert isinstance(document["agent"], dict)
    sampling = document["agent"]["sampling"]
    assert isinstance(sampling, dict)
    for key, value in changes.items():
        if value is None:
            sampling.pop(key, None)
        else:
            sampling[key] = value
    return document


class OpenAICompatibleConfigurationTests(unittest.TestCase):
    def test_glm_configuration_round_trips_and_hashes_deterministically(self) -> None:
        original = _glm_config()
        self.assertIsInstance(original.model, OpenAICompatibleModelConfiguration)
        self.assertIsInstance(original.agent.sampling, OpenAICompatibleSamplingSettings)
        restored = RunConfiguration.from_dict(json.loads(canonical_configuration_json(original)))
        self.assertEqual(restored, original)
        self.assertEqual(run_configuration_hash(restored), run_configuration_hash(original))
        self.assertEqual(run_configuration_hash(_glm_config()), run_configuration_hash(original))
        self.assertEqual(
            original.model.to_dict(),
            {
                "provider": "zai",
                "model_id": "glm-5.3",
                "api_base": "https://api.z.ai/api/paas/v4",
                "thinking_type": "enabled",
                "clear_thinking": True,
                "reasoning_effort": "low",
            },
        )
        self.assertEqual(
            original.agent.sampling.to_dict(),
            {
                "temperature": 1.0,
                "top_p": 0.95,
                "do_sample": True,
                "max_tokens": 1024,
                "plan_max_tokens": 1024,
                "execute_max_tokens": 192,
            },
        )
        document = canonical_configuration_json(original)
        self.assertNotIn(_API_KEY, document)
        self.assertNotIn("api_key", document)
        self.assertNotIn("ZAI_API_KEY", document)
        self.assertEqual(recorded_execution_seed(original), 17)
        self.assertEqual(hosted_provider(original.model), "zai")

    def test_glm_hashed_leaves_and_absent_vllm_and_anthropic_leaves(self) -> None:
        values = hashed_values(_glm_config())
        self.assertEqual(values["model.provider"], "zai")
        self.assertEqual(values["model.model_id"], "glm-5.3")
        self.assertEqual(values["model.api_base"], "https://api.z.ai/api/paas/v4")
        self.assertEqual(values["model.thinking_type"], "enabled")
        self.assertIs(values["model.clear_thinking"], True)
        self.assertEqual(values["model.reasoning_effort"], "low")
        self.assertIs(values["agent.sampling.do_sample"], True)
        self.assertEqual(values["agent.sampling.temperature"], 1.0)
        self.assertEqual(values["agent.sampling.top_p"], 0.95)
        for path in (
            "model.model.repository",
            "model.serving.dtype",
            "model.vllm_version",
            "model.api_version",
            "model.thinking_mode",
            "model.effort",
            "agent.sampling.top_k",
            "agent.sampling.min_p",
            "agent.sampling.seed",
        ):
            with self.subTest(path=path):
                self.assertIs(values[path], MISSING_HASHED_LEAF)

    def test_every_request_control_changes_the_hash(self) -> None:
        base_hash = run_configuration_hash(_glm_config())
        model_changes = {
            "model_id": "glm-5.3-air",
            "api_base": "https://open.bigmodel.cn/api/paas/v4",
            "reasoning_effort": "high",
            "thinking_type": "disabled",
            "clear_thinking": False,
        }
        for field, value in model_changes.items():
            with self.subTest(field=field):
                changed = RunConfiguration.from_dict(_with_model(field, value))
                self.assertNotEqual(run_configuration_hash(changed), base_hash)
                self.assertEqual(hashed_values(changed)[f"model.{field}"], value)
        sampling_changes = {
            "temperature": {"temperature": 0.6},
            "top_p": {"top_p": 0.8},
            "do_sample": {"do_sample": False, "temperature": None, "top_p": None},
            "execute_max_tokens": {"execute_max_tokens": 256},
            "max_tokens": {"max_tokens": 2048},
            "plan_max_tokens": {"plan_max_tokens": 512},
        }
        for field, changes in sampling_changes.items():
            with self.subTest(field=field):
                changed = RunConfiguration.from_dict(_with_sampling(**changes))
                self.assertNotEqual(run_configuration_hash(changed), base_hash)

    def test_provider_is_part_of_the_hash(self) -> None:
        base_hash = run_configuration_hash(_glm_config())
        providers = config_module.OPENAI_COMPATIBLE_PROVIDERS | {"otherhost"}
        with patch.object(config_module, "OPENAI_COMPATIBLE_PROVIDERS", providers):
            changed = RunConfiguration.from_dict(_with_model("provider", "otherhost"))
        self.assertIsInstance(changed.model, OpenAICompatibleModelConfiguration)
        self.assertNotEqual(run_configuration_hash(changed), base_hash)

    def test_unknown_provider_fails_closed(self) -> None:
        sonnet_model = json.loads(_SONNET.read_text(encoding="utf-8"))["model"]
        for provider in ("openai", "kimi", "openrouter", "", 7, None):
            with self.subTest(provider=provider):
                anthropic_shaped = dict(sonnet_model, provider=provider)
                with self.assertRaises(ConfigError):
                    load_model_configuration(anthropic_shaped)
                glm_shaped = dict(_document()["model"], provider=provider)
                with self.assertRaises(ConfigError):
                    load_model_configuration(glm_shaped)
                with self.assertRaises(ConfigError):
                    RunConfiguration.from_dict(_with_model("provider", provider))
        with self.assertRaises(ConfigError):
            hosted_provider(object())

    def test_loader_selects_each_provider_explicitly(self) -> None:
        sonnet = json.loads(_SONNET.read_text(encoding="utf-8"))
        self.assertIsInstance(load_model_configuration(sonnet["model"]), AnthropicModelConfiguration)
        self.assertIsInstance(load_sampling_settings(sonnet["agent"]["sampling"]), HostedSamplingSettings)
        self.assertIs(AnthropicSamplingSettings, HostedSamplingSettings)
        vllm = json.loads(_VLLM_CAPABILITY.read_text(encoding="utf-8"))
        self.assertIsInstance(load_model_configuration(vllm["model"]), ModelConfiguration)
        self.assertIsInstance(load_sampling_settings(vllm["agent"]["sampling"]), SamplingSettings)
        self.assertIsNone(hosted_provider(load_model_configuration(vllm["model"])))
        self.assertEqual(hosted_provider(load_model_configuration(sonnet["model"])), "anthropic")
        with self.assertRaises(ConfigError):
            AnthropicModelConfiguration.from_dict(_document()["model"])
        glm_with_anthropic_fields = dict(sonnet["model"], provider="zai")
        with self.assertRaises(ConfigError):
            load_model_configuration(glm_with_anthropic_fields)

    def test_sampling_schemas_fail_closed_when_partial_or_mixed(self) -> None:
        rejected = (
            {"do_sample": True, "temperature": None},
            {"do_sample": True, "top_p": None},
            {"do_sample": False},
            {"do_sample": False, "temperature": None},
            {"top_k": 20},
            {"min_p": 0.0},
            {"seed": 17},
            {"do_sample": "true"},
            {"top_p": 0.0},
            {"temperature": -0.1},
        )
        for changes in rejected:
            with self.subTest(changes=changes):
                with self.assertRaises(ConfigError):
                    RunConfiguration.from_dict(_with_sampling(**changes))
        greedy = RunConfiguration.from_dict(
            _with_sampling(do_sample=False, temperature=None, top_p=None)
        )
        self.assertEqual(
            set(greedy.agent.sampling.to_dict()),
            {"do_sample", "max_tokens", "plan_max_tokens", "execute_max_tokens"},
        )

    def test_providers_reject_each_others_sampling(self) -> None:
        glm_sampling = json.loads(_GLM.read_text(encoding="utf-8"))["agent"]["sampling"]
        sonnet_sampling = json.loads(_SONNET.read_text(encoding="utf-8"))["agent"]["sampling"]
        vllm_sampling = json.loads(_VLLM_CAPABILITY.read_text(encoding="utf-8"))["agent"]["sampling"]
        for sampling in (sonnet_sampling, vllm_sampling):
            document = _document()
            document["agent"]["sampling"] = copy.deepcopy(sampling)
            with self.assertRaises(ConfigError):
                RunConfiguration.from_dict(document)
        for path in (_SONNET, _VLLM_CAPABILITY):
            document = _document(path)
            document["agent"]["sampling"] = copy.deepcopy(glm_sampling)
            with self.assertRaises(ConfigError):
                RunConfiguration.from_dict(document)

    def test_api_base_must_be_a_plain_https_root(self) -> None:
        for value in (
            "http://api.z.ai/api/paas/v4",
            "https://api.z.ai/api/paas/v4/",
            "https://user:secret@api.z.ai/api/paas/v4",
            "https://api.z.ai/api/paas/v4?key=secret",
            "https://api.z.ai/api/paas/v4#frag",
            "https:///api/paas/v4",
            " https://api.z.ai/api/paas/v4",
            "",
        ):
            with self.subTest(value=value):
                with self.assertRaises(ConfigError):
                    RunConfiguration.from_dict(_with_model("api_base", value))

    def test_existing_committed_hashes_reproduce(self) -> None:
        sonnet = RunConfiguration.from_dict(_document(_SONNET, git_commit=_SONNET_COMMIT))
        self.assertEqual(run_configuration_hash(sonnet), _SONNET_HASH)
        diagnostic = RunConfiguration.from_dict(
            _document(_DIAGNOSTIC, git_commit=_DIAGNOSTIC_COMMIT, task_path=_DIAGNOSTIC_TASKS)
        )
        self.assertIsInstance(diagnostic.model, ModelConfiguration)
        self.assertEqual(run_configuration_hash(diagnostic), _DIAGNOSTIC_HASH)
        for path in (_SONNET, _DIAGNOSTIC, _VLLM_CAPABILITY):
            document = canonical_configuration_json(RunConfiguration.from_dict(_document(path)))
            for leaf in ("api_base", "thinking_type", "clear_thinking", "reasoning_effort", "do_sample"):
                self.assertNotIn(f'"{leaf}"', document)


class OpenAICompatibleRequestTests(unittest.TestCase):
    def test_request_url_auth_and_exact_body(self) -> None:
        config = _glm_config()
        agent = _agent(config)
        captured: list[object] = []

        def urlopen(request: object, timeout: object = None) -> _Response:
            del timeout
            captured.append(request)
            return _Response(json.dumps(_completion(_ENVELOPE)).encode("utf-8"))

        with patch("urllib.request.urlopen", side_effect=urlopen):
            turn = agent.generate_turn(
                tool_output=None,
                extra_instruction="controller instruction",
                parse_action=False,
            )
        self.assertEqual(len(captured), 1)
        request = captured[0]
        self.assertEqual(request.full_url, _ENDPOINT)
        self.assertEqual(request.get_method(), "POST")
        self.assertEqual(request.get_header("Authorization"), f"Bearer {_API_KEY}")
        self.assertEqual(request.get_header("Content-type"), "application/json")
        raw = request.data.decode("utf-8")
        self.assertNotIn(_API_KEY, raw)
        system = render_system_text(
            prompt_version="prompt-runtime-auth-v2",
            plan_format_version="plan-v1",
            thinking_enabled=False,
            action_interface="code",
            mode="execute",
        )
        self.assertEqual(
            json.loads(raw),
            {
                "model": "glm-5.3",
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": agent.messages()[1]["content"]},
                    {"role": "user", "content": "controller instruction"},
                ],
                "temperature": 1.0,
                "top_p": 0.95,
                "do_sample": True,
                "max_tokens": 192,
                "stream": False,
                "thinking": {"type": "enabled", "clear_thinking": True},
                "reasoning_effort": "low",
            },
        )
        first_user = agent.messages()[1]["content"]
        self.assertTrue(first_user.startswith("List the playlists\n"))
        self.assertIn("show_playlist_library", first_user)
        self.assertEqual(turn.output_text, _ENVELOPE)
        self.assertEqual(turn.generated_token_count, 37)
        self.assertEqual(turn.top_k_logprobs, ())
        self.assertNotIn(_API_KEY, repr(agent))

    def test_history_is_visible_content_only_and_alternates_roles(self) -> None:
        agent = _agent()
        captured: list[dict[str, object]] = []

        def urlopen(request: object, timeout: object = None) -> _Response:
            del timeout
            captured.append(json.loads(request.data.decode("utf-8")))
            return _Response(json.dumps(_completion(f"turn-{len(captured)}")).encode("utf-8"))

        with patch("urllib.request.urlopen", side_effect=urlopen):
            first = agent.generate_turn(tool_output=None, extra_instruction="instr-1", parse_action=False)
            second = agent.generate_turn(
                tool_output="OBSERVATION-TOKEN", extra_instruction="instr-2", parse_action=False
            )
        self.assertEqual(first.output_text, "turn-1")
        self.assertEqual(second.prompt_text, "instr-2")
        messages = captured[1]["messages"]
        self.assertEqual(
            [(message["role"], message["content"]) for message in messages[2:]],
            [
                ("assistant", "turn-1"),
                ("user", "OBSERVATION-TOKEN"),
                ("user", "instr-2"),
            ],
        )
        self.assertEqual(messages[0], captured[0]["messages"][0])
        self.assertEqual(messages[1], captured[0]["messages"][1])
        for payload in captured:
            self.assertNotIn(_REASONING, json.dumps(payload))
            self.assertNotIn("reasoning_content", json.dumps(payload))
        for item in agent._state().history:
            self.assertNotIn(_REASONING, item["content"])

    def test_workflow_parses_visible_content_and_never_reasoning(self) -> None:
        config = _glm_config()
        agent = SmolagentsOpenAICompatibleAgent("zai", _API_KEY)
        controller = WorkflowControlledAgent(agent, config.agent.workflow)
        controller.begin(_context(), config)
        self.assertEqual(controller.underlying_agents(), (agent,))
        reasoning = json.dumps(
            {"plan": ["x", "y"], "active_step": 1, "action": "apis.spotify.delete_playlist(playlist_id=1)"}
        )

        def urlopen(request: object, timeout: object = None) -> _Response:
            del timeout
            payload = json.loads(request.data.decode("utf-8"))
            self.assertIn("plan_progress_v2", json.dumps(payload))
            self.assertEqual(payload["max_tokens"], 192)
            return _Response(
                json.dumps(_completion(_ENVELOPE, completion_tokens=41, reasoning=reasoning)).encode("utf-8")
            )

        with patch("urllib.request.urlopen", side_effect=urlopen):
            turn = controller.next_turn(tool_output=None)
        self.assertEqual(turn.action, "apis.spotify.show_playlist_library()")
        self.assertEqual(turn.app_name, "spotify")
        self.assertEqual(turn.api_name, "show_playlist_library")
        self.assertIsNone(turn.rejection)
        self.assertEqual(turn.generated_token_count, 41)
        self.assertNotIn("delete_playlist", turn.output_text)

    def test_reasoning_only_response_is_not_an_action(self) -> None:
        agent = _agent()
        with patch(
            "urllib.request.urlopen",
            side_effect=_respond(_completion("", reasoning="apis.spotify.show_playlist_library()")),
        ):
            turn = agent.next_turn(tool_output=None)
        self.assertEqual(turn.output_text, "")
        self.assertIsNone(turn.action)
        self.assertIsNotNone(turn.rejection)

    def test_length_finish_preserves_visible_content(self) -> None:
        agent = _agent()
        with patch(
            "urllib.request.urlopen",
            side_effect=_respond(_completion('{"plan": ["a"', finish_reason="length", completion_tokens=192)),
        ):
            turn = agent.generate_turn(tool_output=None, parse_action=False)
        self.assertEqual(turn.output_text, '{"plan": ["a"')
        self.assertEqual(turn.generated_token_count, 192)
        with patch(
            "urllib.request.urlopen",
            side_effect=_respond(_completion(None, finish_reason="length", completion_tokens=192)),
        ):
            empty = agent.generate_turn(tool_output="observation", parse_action=False)
        self.assertEqual(empty.output_text, "")
        self.assertEqual(agent._state().history[-1], {"role": "assistant", "content": ""})

    def test_finish_reasons_that_are_runtime_failures(self) -> None:
        cases = (
            ("model_context_window_exceeded", "context_length_exceeded"),
            ("network_error", None),
            ("sensitive", None),
            ("tool_calls", None),
            (None, None),
        )
        for finish_reason, expected in cases:
            with self.subTest(finish_reason=finish_reason):
                agent = _agent()
                with patch(
                    "urllib.request.urlopen",
                    side_effect=_respond(_completion("partial", finish_reason=finish_reason)),
                ):
                    with self.assertRaises(RuntimeUnavailable) as caught:
                        agent.generate_turn(tool_output=None, parse_action=False)
                self.assertEqual(caught.exception.reason, expected)
                self.assertEqual(agent._state().history, [])

    def test_http_errors_retry_only_when_transient_and_redact_the_key(self) -> None:
        secret_body = f'{{"error":{{"message":"bad key {_API_KEY}"}}}}'.encode("utf-8")
        cases = (
            (401, "Unauthorized", secret_body, None, 1),
            (403, "Forbidden", secret_body, None, 1),
            (400, "Bad Request", b"invalid parameter", None, 1),
            (429, "Too Many Requests", secret_body, None, 5),
            (500, "Internal Server Error", b"upstream failed", None, 5),
            (503, "Service Unavailable", b"overloaded", None, 5),
            (400, "Bad Request", b"prompt exceeds the maximum context length", "context_length_exceeded", 1),
        )
        for status, reason, body, expected_reason, calls in cases:
            with self.subTest(status=status, body=body[:16]):
                agent = _agent()
                seen = {"count": 0}

                def urlopen(request: object, timeout: object = None) -> _Response:
                    del request, timeout
                    seen["count"] += 1
                    raise _http_error(status, reason, body)

                with (
                    patch("urllib.request.urlopen", side_effect=urlopen),
                    patch("llm_behavior_ci.runtime.agent.time.sleep") as sleep,
                ):
                    with self.assertRaises(RuntimeUnavailable) as caught:
                        agent.generate_turn(tool_output=None, parse_action=False)
                self.assertEqual(seen["count"], calls)
                self.assertEqual(sleep.call_count, calls - 1)
                text = str(caught.exception)
                self.assertIn(f"HTTP {status}", text)
                self.assertIn("Z.AI", text)
                self.assertNotIn(_API_KEY, text)
                self.assertEqual(caught.exception.reason, expected_reason)

    def test_transient_error_then_success_returns_the_turn(self) -> None:
        agent = _agent()
        attempts = {"count": 0}

        def urlopen(request: object, timeout: object = None) -> _Response:
            del request, timeout
            attempts["count"] += 1
            if attempts["count"] == 1:
                raise _http_error(429, "Too Many Requests", b"slow down")
            if attempts["count"] == 2:
                raise urllib.error.URLError("timed out")
            return _Response(json.dumps(_completion("ok")).encode("utf-8"))

        with (
            patch("urllib.request.urlopen", side_effect=urlopen),
            patch("llm_behavior_ci.runtime.agent.time.sleep"),
        ):
            turn = agent.generate_turn(tool_output=None, parse_action=False)
        self.assertEqual(attempts["count"], 3)
        self.assertEqual(turn.output_text, "ok")

    def test_network_failures_are_bounded_and_redacted(self) -> None:
        agent = _agent()
        seen = {"count": 0}

        def urlopen(request: object, timeout: object = None) -> _Response:
            del request, timeout
            seen["count"] += 1
            raise urllib.error.URLError(f"connection reset {_API_KEY}")

        with (
            patch("urllib.request.urlopen", side_effect=urlopen),
            patch("llm_behavior_ci.runtime.agent.time.sleep"),
        ):
            with self.assertRaises(RuntimeUnavailable) as caught:
                agent.generate_turn(tool_output=None, parse_action=False)
        self.assertEqual(seen["count"], 5)
        self.assertNotIn(_API_KEY, str(caught.exception))

    def test_malformed_responses_fail_closed(self) -> None:
        bodies = (
            f"not-json {_API_KEY}".encode("utf-8"),
            b"[]",
            {"usage": {"completion_tokens": 1}},
            {"choices": [], "usage": {"completion_tokens": 1}},
            {"choices": ["text"], "usage": {"completion_tokens": 1}},
            {"choices": [{"finish_reason": "stop"}], "usage": {"completion_tokens": 1}},
            _completion(None),
            _completion(["block"]),
            {"choices": _completion("ok")["choices"]},
            _completion("ok", completion_tokens=-1),
            _completion("ok", completion_tokens=True),
            _completion("ok", completion_tokens="5"),
        )
        for body in bodies:
            with self.subTest(body=str(body)[:40]):
                agent = _agent()
                with patch("urllib.request.urlopen", side_effect=_respond(body)):
                    with self.assertRaises(RuntimeUnavailable) as caught:
                        agent.generate_turn(tool_output=None, parse_action=False)
                self.assertIn("malformed", str(caught.exception))
                self.assertNotIn(_API_KEY, str(caught.exception))
                self.assertEqual(agent._state().history, [])

    def test_teacher_forced_plan_kl_is_unsupported(self) -> None:
        agent = _agent()
        with patch("urllib.request.urlopen") as urlopen:
            with self.assertRaises(UnsupportedCapability) as caught:
                agent.teacher_force_plan(
                    messages=[{"role": "user", "content": "plan"}],
                    plan_text="1. list playlists",
                )
        urlopen.assert_not_called()
        self.assertIn("teacher-forced plan KL is unsupported", str(caught.exception))

    def test_agent_refuses_mismatched_configurations(self) -> None:
        with self.assertRaises(EpisodeRejected):
            SmolagentsOpenAICompatibleAgent("openai", _API_KEY)
        with self.assertRaises(EpisodeRejected) as caught:
            SmolagentsOpenAICompatibleAgent("zai", "   ")
        self.assertIn("ZAI_API_KEY is required", str(caught.exception))
        agent = SmolagentsOpenAICompatibleAgent("zai", _API_KEY)
        sonnet = RunConfiguration.from_dict(_document(_SONNET))
        with self.assertRaises(RuntimeUnavailable):
            agent.begin(_context(), sonnet)
        no_clear = RunConfiguration.from_dict(_with_model("clear_thinking", False))
        with self.assertRaises(RuntimeUnavailable):
            agent.begin(_context(), no_clear)


class OpenAICompatibleRuntimeTests(unittest.TestCase):
    def test_build_runtime_requires_the_zai_key_and_no_endpoint(self) -> None:
        config = _glm_config()
        with patch.dict(
            os.environ, {"ZAI_API_KEY": "", "ANTHROPIC_API_KEY": "sk-ant-other"}, clear=False
        ):
            with self.assertRaises(EpisodeRejected) as caught:
                build_runtime(config, mode="execute")
        self.assertIn("ZAI_API_KEY is required", str(caught.exception))
        with patch.dict(os.environ, {"ZAI_API_KEY": _API_KEY}, clear=False):
            with self.assertRaises(EpisodeRejected):
                build_runtime(config, "http://127.0.0.1:8000", mode="execute")
            with self.assertRaises(EpisodeRejected):
                build_runtime(
                    RunConfiguration.from_dict(_with_model("clear_thinking", False)),
                    mode="execute",
                )
            runtime = build_runtime(config, mode="execute")
            plan_runtime = build_runtime(config, mode="plan")
        self.assertIsInstance(runtime.agent, WorkflowControlledAgent)
        base = runtime.agent.underlying_agents()[0]
        self.assertIsInstance(base, SmolagentsOpenAICompatibleAgent)
        self.assertEqual(base.provider, "zai")
        self.assertTrue(is_live_runtime(runtime))
        self.assertNotIn(_API_KEY, repr(runtime.agent))
        self.assertNotIn(_API_KEY, repr(base))
        self.assertIsInstance(plan_runtime.agent, SmolagentsOpenAICompatibleAgent)
        self.assertTrue(is_live_runtime(plan_runtime))

    def test_wrapped_fake_agent_is_not_live(self) -> None:
        config = _glm_config()

        class _Fake:
            def begin(self, context: object, config: object) -> None:
                del context, config

        with patch.dict(os.environ, {"ZAI_API_KEY": _API_KEY}, clear=False):
            runtime = build_runtime(config, mode="execute")
        fake = RuntimeDependencies(
            session_factory=runtime.session_factory,
            agent=WorkflowControlledAgent(_Fake(), config.agent.workflow),
            clock=runtime.clock,
        )
        self.assertFalse(is_live_runtime(fake))


def _script(name: str):
    import importlib.util

    path = _ROOT / "scripts" / "evaluation" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"zai_{name}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fake_task_load(payload: object):
    del payload
    from llm_behavior_ci.config import TaskConfiguration
    from llm_behavior_ci.tasks.selection import TaskSet

    task = TaskConfiguration.from_dict(_task())
    return task, TaskSet(
        appworld_version=task.appworld_version,
        split=task.split,
        selection_rule=task.selection_rule,
        selection_seed=task.selection_seed,
        task_count=20,
        scenario_count=20,
        task_ids=tuple(f"task-{index}" for index in range(20)),
        scenario_ids=tuple(f"scenario-{index}" for index in range(20)),
        task_set_hash=task.task_set_hash,
    )


def _completed_episode(task_id, config, **kwargs) -> EpisodeResult:
    identity = new_episode_identity(kwargs["run"])
    kwargs["on_start"](identity, kwargs["run"])
    return EpisodeResult(
        episode=identity,
        run=kwargs["run"],
        task=LocalTaskRef(task_id=task_id, scenario_id=kwargs["scenario_id"], split=config.task.split),
        mode="execute",
        execution_seed=recorded_execution_seed(config),
        status="completed",
        started_at=_START,
        ended_at=_START,
        model_steps=(),
        tool_steps=(),
        plan_text=None,
        evaluator_outcome=EvaluatorOutcome(
            success=True,
            passed_requirements=2,
            total_requirements=2,
            difficulty=1,
        ),
        termination_reason="appworld_completed",
        episode_errors=(),
        role=None,
    )


class _AppWorldRoot:
    def __init__(self, root: Path) -> None:
        self._root = root
        self._previous: str | None = None

    def __enter__(self) -> None:
        self._previous = os.environ.get("APPWORLD_ROOT")
        os.environ["APPWORLD_ROOT"] = str(self._root)

    def __exit__(self, *args: object) -> None:
        del args
        if self._previous is None:
            os.environ.pop("APPWORLD_ROOT", None)
        else:
            os.environ["APPWORLD_ROOT"] = self._previous


class OpenAICompatibleCliTests(unittest.TestCase):
    def _pilot_argv(self, root: Path, *, base_url: str | None = None) -> list[str]:
        argv = [
            "--configuration",
            str(_GLM),
            "--task-set",
            str(_TASK_SET),
            "--store",
            str(root / "episodes.sqlite"),
            "--output",
            str(root / "aggregate.json"),
        ]
        if base_url is not None:
            argv.extend(["--base-url", base_url])
        return argv

    def test_pilot_accepts_the_glm_profile(self) -> None:
        pilot = _script("run_capability_pilot")
        document = json.loads(_GLM.read_text(encoding="utf-8"))
        pilot._require_pilot_configuration(
            load_model_configuration(document["model"]),
            AgentConfiguration.from_dict(document["agent"]),
        )
        for changes in ({"execute_max_tokens": 256},):
            changed = _with_sampling(**changes)
            with patch.object(pilot.sys, "stderr", io.StringIO()):
                with self.assertRaises(SystemExit):
                    pilot._require_pilot_configuration(
                        load_model_configuration(changed["model"]),
                        AgentConfiguration.from_dict(changed["agent"]),
                    )

    def test_pilot_and_smoke_reject_a_local_endpoint(self) -> None:
        pilot = _script("run_capability_pilot")
        smoke = _script("smoke_live_episode")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with _AppWorldRoot(root), patch.dict(os.environ, {"ZAI_API_KEY": _API_KEY}, clear=False):
                with patch.object(pilot.sys, "stderr", io.StringIO()) as stderr:
                    code = pilot.main(self._pilot_argv(root, base_url="http://127.0.0.1:8000"))
                self.assertEqual(code, 2)
                self.assertIn("does not use --base-url", stderr.getvalue())
                with patch.object(smoke.sys, "stderr", io.StringIO()) as stderr:
                    smoke_code = smoke.main(
                        [
                            "--configuration",
                            str(_GLM),
                            "--task-set",
                            str(_TASK_SET),
                            "--task-index",
                            "0",
                            "--base-url",
                            "http://127.0.0.1:8000",
                            "--store",
                            str(root / "smoke.sqlite"),
                        ]
                    )
                self.assertEqual(smoke_code, 2)
                self.assertIn("does not use --base-url", stderr.getvalue())
                self.assertFalse((root / "aggregate.json").exists())
                self.assertFalse((root / "smoke.sqlite").exists())

    def test_missing_key_aborts_before_task_zero(self) -> None:
        pilot = _script("run_capability_pilot")
        calls: list[str] = []

        def run(*args, **kwargs):
            del args, kwargs
            calls.append("ran")
            raise AssertionError("task 0 started")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stderr = io.StringIO()
            with (
                _AppWorldRoot(root),
                patch.object(pilot, "enforce_committed_provenance", lambda config: None),
                patch.object(pilot, "_load_task_set", _fake_task_load),
                patch.object(pilot, "_adopt_committed_setup_profile", lambda task: task),
                patch.object(pilot, "run_episode", side_effect=run),
                patch.object(pilot.sys, "stderr", stderr),
                patch.dict(os.environ, {"ZAI_API_KEY": "", "ANTHROPIC_API_KEY": "sk-ant-other"}, clear=False),
            ):
                code = pilot.main(self._pilot_argv(root))
            payload = json.loads((root / "aggregate.json").read_text(encoding="utf-8"))
        self.assertEqual(code, 1)
        self.assertEqual(calls, [])
        self.assertEqual(payload["run_status"], "aborted")
        self.assertEqual(payload["episodes_attempted"], 0)
        self.assertEqual(payload["not_attempted_count"], 20)
        self.assertIn("ZAI_API_KEY is required", payload["failure_reason"])
        self.assertNotIn("sk-ant-other", json.dumps(payload) + stderr.getvalue())

    def test_one_episode_local_provider_error_continues_the_pilot(self) -> None:
        pilot = _script("run_capability_pilot")
        calls: list[str] = []

        def run(task_id, config, mode, **kwargs):
            del mode
            calls.append(task_id)
            self.assertIs(kwargs["evaluate_after_runtime_failure"], True)
            self.assertIsInstance(config.model, OpenAICompatibleModelConfiguration)
            self.assertEqual(config.task.task_set_hash, _TASK_HASH)
            self.assertEqual(recorded_execution_seed(config), 17)
            if len(calls) == 2:
                raise RuntimeUnavailable("Z.AI provider failure: finish_reason network_error")
            return _completed_episode(task_id, config, **kwargs)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stderr = io.StringIO()
            with (
                _AppWorldRoot(root),
                patch.object(pilot, "enforce_committed_provenance", lambda config: None),
                patch.object(pilot, "_load_task_set", _fake_task_load),
                patch.object(pilot, "_adopt_committed_setup_profile", lambda task: task),
                patch.object(pilot, "run_episode", side_effect=run),
                patch.object(pilot, "build_runtime", return_value=object()),
                patch.object(pilot.sys, "stderr", stderr),
                patch("builtins.print"),
                patch.dict(os.environ, {"ZAI_API_KEY": _API_KEY}, clear=False),
            ):
                code = pilot.main(self._pilot_argv(root))
            payload = json.loads((root / "aggregate.json").read_text(encoding="utf-8"))
        self.assertEqual(code, 0)
        self.assertEqual(len(calls), 20)
        self.assertNotIn(_API_KEY, stderr.getvalue() + json.dumps(payload))
        self.assertEqual(payload["run_status"], "completed")
        self.assertEqual(payload["task_count"], 20)
        self.assertEqual(payload["runtime_failure_count"], 1)
        self.assertEqual(payload["end_to_end_success_count"], 19)
        self.assertEqual(
            payload["terminal_status_counts"],
            {
                "evaluator_success": 19,
                "evaluator_failure": 0,
                "runtime_failure": 1,
                "not_attempted": 0,
            },
        )


if __name__ == "__main__":
    unittest.main()
