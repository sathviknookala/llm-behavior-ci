from __future__ import annotations

import json
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from llm_behavior_ci.config import TaskConfiguration, new_run_identity
from llm_behavior_ci.experiments.run_config import build_run_configuration
from llm_behavior_ci.runtime import aa_capture
from llm_behavior_ci.runtime.aa_capture import capture_aa
from llm_behavior_ci.runtime.agent import SmolagentsOpenAICompatibleAgent, UnsupportedCapability
from llm_behavior_ci.runtime.appworld import EvaluationResult, TaskContext, ToolResult
from llm_behavior_ci.runtime.episode import EpisodeRejected, RuntimeDependencies, run_episode
from llm_behavior_ci.tasks.streams import TaskArrival

_ROOT = Path(__file__).resolve().parents[2]
_KEY = "zai-test-secret-value-0123456789"
_REASONING = "private chain of thought that must never be stored"
_PLAN = "1. Call apis.spotify.show_playlist_library().\n2. Report the count with supervisor.complete_task."


def _config():
    task = json.loads((_ROOT / "configs" / "tasks" / "train_smoke.json").read_text(encoding="utf-8"))
    task.pop("scenario_count", None)
    model = json.loads(
        (_ROOT / "configs" / "models" / "glm_5_3_spotify_capability.json").read_text(encoding="utf-8")
    )
    return build_run_configuration(
        model, TaskConfiguration.from_dict(task), run_seed=17, git_commit="a" * 40
    )


class _Response:
    def __init__(self, raw: bytes) -> None:
        self._raw = raw

    def read(self) -> bytes:
        return self._raw

    def __enter__(self) -> _Response:
        return self

    def __exit__(self, *args: object) -> bool:
        return False


def _urlopen(request: object, timeout: object = None) -> _Response:
    del request, timeout
    body = {
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {
                    "role": "assistant",
                    "content": _PLAN,
                    "reasoning_content": _REASONING,
                },
            }
        ],
        "usage": {
            "prompt_tokens": 120,
            "completion_tokens": 40,
            "completion_tokens_details": {"reasoning_tokens": 25},
        },
    }
    return _Response(json.dumps(body).encode("utf-8"))


class _World:
    executed: list[str]

    def __init__(self, task_id: str) -> None:
        self.task_id = task_id
        self.executed = []
        self.evaluated = False

    def context(self) -> TaskContext:
        return TaskContext(
            task_id=self.task_id,
            instruction="How many playlists do I have?",
            api_documentation="spotify.show_playlist_library: list playlists",
        )

    def execute(self, action: str) -> ToolResult:
        self.executed.append(action)
        raise AssertionError("plan mode must not execute a tool")

    def evaluate(self) -> EvaluationResult:
        self.evaluated = True
        raise AssertionError("plan mode must not evaluate")

    def close(self) -> None:
        pass


def _runtime(worlds: list[_World]) -> RuntimeDependencies:
    agent = SmolagentsOpenAICompatibleAgent("zai", _KEY)
    agent.set_mode("plan")

    def factory(task_id: str) -> _World:
        world = _World(task_id)
        worlds.append(world)
        return world

    return RuntimeDependencies(
        session_factory=factory, agent=agent, clock=lambda: datetime.now(timezone.utc)
    )


class HostedPlanModeTests(unittest.TestCase):
    def test_plan_episode_is_visible_text_without_tools_outcome_or_reasoning(self) -> None:
        config = _config()
        worlds: list[_World] = []
        with patch("urllib.request.urlopen", side_effect=_urlopen):
            episode = run_episode(
                "task-1",
                config,
                "plan",
                run=new_run_identity(config),
                runtime=_runtime(worlds),
                scenario_id="scenario-1",
            )
        self.assertEqual(episode.plan_text, _PLAN)
        self.assertEqual(episode.tool_steps, ())
        self.assertIsNone(episode.evaluator_outcome)
        self.assertTrue(all(step.top_k_logprobs == () for step in episode.model_steps))
        self.assertEqual(worlds[0].executed, [])
        self.assertFalse(worlds[0].evaluated)
        (call,) = episode.provider_calls
        self.assertEqual(call.mode, "plan")
        self.assertEqual(call.input_tokens, 120)
        self.assertEqual(call.output_tokens, 40)
        self.assertEqual(call.reasoning_tokens, 25)
        self.assertIsNone(call.cache_read_tokens)
        stored = json.dumps(episode.to_dict())
        self.assertNotIn(_REASONING, stored)
        self.assertNotIn(_KEY, stored)

    def test_teacher_forced_plan_kl_stays_unsupported(self) -> None:
        agent = SmolagentsOpenAICompatibleAgent("zai", _KEY)
        with self.assertRaises(UnsupportedCapability):
            agent.teacher_force_plan(messages=[{"role": "user", "content": "i"}], plan_text=_PLAN)


class HostedAATests(unittest.TestCase):
    def _arrivals(self) -> tuple[TaskArrival, ...]:
        return (
            TaskArrival(
                index=0,
                task_id="task-1",
                scenario_id="scenario-1",
                scheduled_offset_seconds=0.0,
                stream_seed=3,
            ),
        )

    def test_hosted_capture_never_probes_gpu_and_records_wall_time(self) -> None:
        config = _config()
        worlds: list[_World] = []

        def forbidden() -> object:
            raise AssertionError("nvidia-smi must not run for a hosted configuration")

        with patch.object(aa_capture, "read_nvidia_smi_snapshot", forbidden), patch(
            "urllib.request.urlopen", side_effect=_urlopen
        ):
            result = capture_aa(
                config,
                self._arrivals(),
                task_set_hash=config.task.task_set_hash,
                repetitions=1,
                concurrency=1,
                modes=("plan",),
                runtime_factory=lambda mode: _runtime(worlds),
                observe_hardware=False,
            )
            with self.assertRaises(EpisodeRejected):
                capture_aa(
                    config,
                    self._arrivals(),
                    task_set_hash=config.task.task_set_hash,
                    repetitions=1,
                    concurrency=1,
                    modes=("plan",),
                    runtime_factory=lambda mode: _runtime(worlds),
                    observe_hardware=True,
                )
        self.assertEqual(result.cost.hardware_applicability, "not_applicable")
        self.assertFalse(result.cost.hardware_observed)
        self.assertIsNone(result.cost.memory_used_mib)
        self.assertIsNotNone(result.cost.elapsed_wall_seconds)
        self.assertEqual(len(result.records), 1)


if __name__ == "__main__":
    unittest.main()
