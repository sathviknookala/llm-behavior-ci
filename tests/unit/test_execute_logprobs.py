import json
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

from llm_behavior_ci.config import RunConfiguration, new_run_identity
from llm_behavior_ci.records import ModelStep
from llm_behavior_ci.runtime.aa_capture import teacher_forced_plan_kl
from llm_behavior_ci.runtime.agent import SmolagentsVLLMAgent
from llm_behavior_ci.runtime.appworld import EvaluationResult, TaskContext, ToolResult
from llm_behavior_ci.runtime.clock import wall_now
from llm_behavior_ci.runtime.episode import (
    RuntimeDependencies,
    RuntimeUnavailable,
    run_episode,
)

_ROOT = Path(__file__).resolve().parents[2]
_CALL = "apis.supervisor.complete_task()"
_SPOTIFY_CALL = "apis.spotify.show_song(song_id=12)"


def _config(model_file: str = "qwen3_4b_production.json") -> RunConfiguration:
    document = json.loads(
        (_ROOT / "configs/models" / model_file).read_text(encoding="utf-8")
    )
    document["task"] = {
        "appworld_version": "0.1.3.post1",
        "split": "train",
        "selection_rule": "deterministic_sample",
        "selection_seed": 17,
        "task_count": 1,
        "task_set_hash": "c" * 64,
    }
    document["run_seed"] = 17
    document["git_commit"] = "a" * 40
    document["protocol_hash"] = None
    return RunConfiguration.from_dict(document)


def _pilot() -> RunConfiguration:
    return _config("qwen3_4b_spotify_capability.json")


def _with_agent(config: RunConfiguration, **changes: object) -> RunConfiguration:
    document: dict[str, Any] = json.loads(json.dumps(config.to_dict()))
    for key, value in changes.items():
        if key in ("execute_max_tokens", "plan_max_tokens"):
            document["agent"]["sampling"][key] = value
        elif value is None:
            document["agent"].pop(key, None)
        else:
            document["agent"][key] = value
    return RunConfiguration.from_dict(document)


def _choice(
    content: str,
    *,
    logprobs: bool,
    token_ids: list[int] | None = None,
) -> dict[str, object]:
    choice: dict[str, object] = {
        "message": {"content": content},
        "token_ids": [7] if token_ids is None else token_ids,
    }
    if logprobs:
        choice["logprobs"] = {
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
    return {"choices": [choice]}


class _Response:
    def __init__(self, payload: Any) -> None:
        self._payload = payload

    def read(self) -> bytes:
        return json.dumps(self._payload).encode("utf-8")

    def __enter__(self) -> "_Response":
        return self

    def __exit__(self, *args: object) -> None:
        return None


class _Session:
    def __init__(self) -> None:
        self.actions: list[str] = []
        self.evaluate_count = 0

    def context(self) -> TaskContext:
        return TaskContext(
            task_id="task-1",
            instruction="instruction",
            api_documentation="docs",
        )

    def execute(self, action: str) -> ToolResult:
        self.actions.append(action)
        return ToolResult(
            output_text="ok",
            error_message=None,
            recoverable=False,
            app_name=None,
            api_name=None,
        )

    def evaluate(self) -> EvaluationResult:
        self.evaluate_count += 1
        return EvaluationResult(
            success=False,
            passed_requirements=0,
            total_requirements=4,
            difficulty=2,
        )

    def close(self) -> None:
        return None


def _context() -> TaskContext:
    return TaskContext(task_id="task-1", instruction="instruction", api_documentation="docs")


def _run_execute(
    config: RunConfiguration,
    content: str,
    *,
    include_logprobs: bool = False,
    token_ids: list[int] | None = None,
):
    agent = SmolagentsVLLMAgent("http://127.0.0.1:9")
    session = _Session()
    posted: list[dict[str, object]] = []

    def urlopen(request, timeout=None):
        del timeout
        posted.append(json.loads(request.data.decode("utf-8")))
        return _Response(
            _choice(content, logprobs=include_logprobs, token_ids=token_ids)
        )

    with patch("urllib.request.urlopen", side_effect=urlopen):
        result = run_episode(
            "task-1",
            config,
            "execute",
            run=new_run_identity(config),
            runtime=RuntimeDependencies(
                session_factory=lambda task_id: session,
                agent=agent,
                clock=wall_now,
            ),
        )
    return result, session, posted


class ExecuteLogprobTests(unittest.TestCase):
    def test_execute_requests_omit_top_20_logprobs(self) -> None:
        agent = SmolagentsVLLMAgent("http://127.0.0.1:9")
        agent.begin(_context(), _config())
        payload = agent.completion_payload(agent.messages())
        self.assertNotIn("logprobs", payload)
        self.assertNotIn("top_logprobs", payload)
        self.assertIs(payload["return_token_ids"], True)
        self.assertEqual(payload["max_tokens"], _config().agent.sampling.max_tokens)

    def test_pilot_execute_turn_uses_192_tokens_and_counts_token_ids(self) -> None:
        agent = SmolagentsVLLMAgent("http://127.0.0.1:9")
        agent.begin(_context(), _pilot())
        posted: list[dict[str, object]] = []

        def urlopen(request, timeout=None):
            del timeout
            posted.append(json.loads(request.data.decode("utf-8")))
            return _Response(
                _choice(_SPOTIFY_CALL, logprobs=False, token_ids=[4, 5, 6, 7, 8])
            )

        with patch("urllib.request.urlopen", side_effect=urlopen):
            turn = agent.next_turn(tool_output=None)
        self.assertEqual(posted[0]["max_tokens"], 192)
        self.assertNotIn("logprobs", posted[0])
        self.assertNotIn("top_logprobs", posted[0])
        self.assertIs(posted[0]["return_token_ids"], True)
        self.assertEqual(turn.action, _SPOTIFY_CALL)
        self.assertEqual(turn.app_name, "spotify")
        self.assertEqual(turn.api_name, "show_song")
        self.assertIsNone(turn.rejection)
        self.assertEqual(turn.top_k_logprobs, ())
        self.assertEqual(turn.generated_token_count, 5)

    def test_missing_token_ids_fail_closed(self) -> None:
        agent = SmolagentsVLLMAgent("http://127.0.0.1:9")
        agent.begin(_context(), _pilot())
        response = {"choices": [{"message": {"content": _SPOTIFY_CALL}}]}
        with patch("urllib.request.urlopen", return_value=_Response(response)):
            with self.assertRaisesRegex(RuntimeUnavailable, "token_ids"):
                agent.next_turn(tool_output=None)

    def test_plan_requests_plan_max_tokens_and_logprobs(self) -> None:
        config = _with_agent(_pilot(), plan_max_tokens=777)
        agent = SmolagentsVLLMAgent("http://127.0.0.1:9")
        agent.set_mode("plan")
        agent.begin(_context(), config)
        posted: list[dict[str, object]] = []

        def urlopen(request, timeout=None):
            del timeout
            posted.append(json.loads(request.data.decode("utf-8")))
            return _Response(_choice("1. look", logprobs=True))

        with patch("urllib.request.urlopen", side_effect=urlopen):
            turn = agent.next_turn(tool_output=None)
        self.assertEqual(posted[0]["max_tokens"], 777)
        self.assertIs(posted[0]["logprobs"], True)
        self.assertEqual(posted[0]["top_logprobs"], config.model.serving.max_logprobs)
        self.assertEqual(turn.top_k_logprobs[0][0].token_id, 7)
        self.assertEqual(len(turn.top_k_logprobs[0]), 2)
        self.assertEqual(turn.generated_token_count, 1)
        self.assertIsNone(turn.action)

    def test_pilot_plan_payload_uses_its_plan_cap(self) -> None:
        agent = SmolagentsVLLMAgent("http://127.0.0.1:9")
        agent.set_mode("plan")
        agent.begin(_context(), _pilot())
        payload = agent.completion_payload(agent.messages())
        self.assertEqual(payload["max_tokens"], 1024)
        self.assertIs(payload["logprobs"], True)

    def test_teacher_force_payload_is_unchanged(self) -> None:
        config = _pilot()
        agent = SmolagentsVLLMAgent("http://127.0.0.1:9")
        agent.set_mode("plan")
        agent.begin(_context(), config)
        forced = agent.teacher_force_payload(messages=agent.messages(), plan_text="1. look")
        self.assertEqual(forced["prompt_logprobs"], config.model.serving.max_logprobs)
        self.assertEqual(forced["max_tokens"], 1)
        self.assertNotIn("top_logprobs", forced)
        self.assertNotIn("logprobs", forced)

    def test_teacher_forced_kl_scores_through_the_live_agent(self) -> None:
        config = _pilot()
        agent = SmolagentsVLLMAgent("http://127.0.0.1:9")
        agent.set_mode("plan")
        forced_bodies: list[dict[str, object]] = []

        def urlopen(request, timeout=None):
            del timeout
            body = json.loads(request.data.decode("utf-8"))
            if request.full_url.endswith("/tokenize"):
                return _Response({"tokens": [11], "count": 1})
            forced_bodies.append(body)
            return _Response(
                {
                    "choices": [{"message": {"content": ""}, "token_ids": [1]}],
                    "prompt_token_ids": [11, 3],
                    "prompt_logprobs": [
                        None,
                        {"3": {"logprob": -0.25}, "4": {"logprob": -1.75}},
                    ],
                }
            )

        with patch("urllib.request.urlopen", side_effect=urlopen):
            kl = teacher_forced_plan_kl(
                runtime=RuntimeDependencies(
                    session_factory=lambda task_id: _Session(),
                    agent=agent,
                    clock=wall_now,
                ),
                task_id="task-1",
                configuration=config,
                frozen_plan_text="1. look",
            )
        self.assertEqual(kl.status, "scored")
        self.assertEqual(kl.mean_kl_nats, 0.0)
        self.assertEqual(len(forced_bodies), 2)
        for body in forced_bodies:
            self.assertEqual(body["prompt_logprobs"], config.model.serving.max_logprobs)
            self.assertEqual(body["max_tokens"], 1)

    def test_execute_parsing_and_evaluation_ignore_response_logprobs(self) -> None:
        outcomes = []
        for include_logprobs in (False, True):
            result, session, posted = _run_execute(
                _config(), _CALL, include_logprobs=include_logprobs, token_ids=[3, 4]
            )
            self.assertNotIn("logprobs", posted[0])
            self.assertNotIn("top_logprobs", posted[0])
            self.assertEqual(session.actions, [_CALL])
            self.assertEqual(session.evaluate_count, 1)
            self.assertEqual(result.termination_reason, "appworld_completed")
            self.assertEqual(result.model_steps[0].top_k_logprobs, ())
            self.assertEqual(result.model_steps[0].generated_token_count, 2)
            self.assertIsNotNone(result.evaluator_outcome)
            assert result.evaluator_outcome is not None
            outcomes.append(
                (
                    result.termination_reason,
                    result.status,
                    result.evaluator_outcome.success,
                    result.evaluator_outcome.passed_requirements,
                    result.evaluator_outcome.total_requirements,
                    result.tool_steps[0].action,
                    result.tool_steps[0].app_name,
                    result.tool_steps[0].api_name,
                )
            )
        self.assertEqual(outcomes[0], outcomes[1])

    def test_model_step_token_count_round_trips(self) -> None:
        result, _, _ = _run_execute(_pilot(), _CALL, token_ids=[1, 2, 3])
        step = result.model_steps[0]
        restored = ModelStep.from_dict(step.to_dict())
        assert isinstance(restored, ModelStep)
        self.assertEqual(restored.generated_token_count, 3)
        self.assertEqual(restored, step)


class ExecuteHorizonTests(unittest.TestCase):
    def test_execute_max_model_turns_overrides_step_limit(self) -> None:
        config = _pilot()
        self.assertEqual(config.agent.step_limit, 40)
        self.assertEqual(config.agent.execute_max_model_turns, 20)
        result, session, posted = _run_execute(config, _SPOTIFY_CALL)
        self.assertEqual(result.termination_reason, "step_limit")
        self.assertEqual(len(result.model_steps), 20)
        self.assertEqual(len(session.actions), 20)
        self.assertEqual(len(posted), 20)
        self.assertEqual(session.evaluate_count, 1)

    def test_step_limit_is_the_horizon_without_the_override(self) -> None:
        config = _with_agent(_pilot(), execute_max_model_turns=None)
        self.assertIsNone(config.agent.execute_max_model_turns)
        result, session, posted = _run_execute(config, _SPOTIFY_CALL)
        self.assertEqual(result.termination_reason, "step_limit")
        self.assertEqual(len(result.model_steps), 40)
        self.assertEqual(len(posted), 40)
        self.assertEqual(posted[0]["max_tokens"], 192)


class PilotConfigurationTests(unittest.TestCase):
    def test_pilot_differs_from_production_only_in_pilot_settings(self) -> None:
        production = json.loads(
            (_ROOT / "configs/models/qwen3_4b_production.json").read_text(encoding="utf-8")
        )
        pilot = json.loads(
            (_ROOT / "configs/models/qwen3_4b_spotify_capability.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(pilot["agent"]["prompt"].pop("prompt_version"), "prompt-v4")
        self.assertEqual(production["agent"]["prompt"].pop("prompt_version"), "prompt-v2")
        self.assertEqual(pilot["agent"].pop("execute_max_model_turns"), 20)
        self.assertEqual(
            pilot["agent"].pop("tool_access_profile"),
            "spotify_capability_v1",
        )
        self.assertEqual(pilot["agent"]["sampling"].pop("execute_max_tokens"), 192)
        self.assertEqual(pilot["agent"]["sampling"].pop("plan_max_tokens"), 1024)
        self.assertEqual(pilot, production)
        self.assertEqual(production["agent"]["step_limit"], 40)
        self.assertEqual(production["agent"]["sampling"]["max_tokens"], 1024)


if __name__ == "__main__":
    unittest.main()
