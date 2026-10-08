from __future__ import annotations

import json
import time
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

try:
    from freezegun import freeze_time
except ImportError:
    freeze_time = None

from llm_behavior_ci.config import TaskConfiguration, new_run_identity
from llm_behavior_ci.experiments.run_config import build_run_configuration
from llm_behavior_ci.runtime.agent import SmolagentsOpenAICompatibleAgent
from llm_behavior_ci.runtime.appworld import TaskContext
from llm_behavior_ci.runtime.clock import monotonic, wall_now
from llm_behavior_ci.runtime.episode import RuntimeDependencies, run_episode

_ROOT = Path(__file__).resolve().parents[2]
_KEY = "zai-test-secret-value-0123456789"
_FROZEN = "2023-05-18T12:00:00+00:00"
_DELAY = 0.05


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


def _slow_urlopen(request: object, timeout: object = None) -> _Response:
    del request, timeout
    time.sleep(_DELAY)
    body = {
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": "1. Call apis.spotify.show_playlist_library()."},
            }
        ],
        "usage": {"prompt_tokens": 100, "completion_tokens": 10},
    }
    return _Response(json.dumps(body).encode("utf-8"))


class _FreezingWorld:
    def __init__(self, task_id: str) -> None:
        assert freeze_time is not None
        self.task_id = task_id
        self.frozen_now: datetime | None = None
        self._freezer = freeze_time(_FROZEN)
        self._freezer.start()

    def context(self) -> TaskContext:
        self.frozen_now = datetime.now(timezone.utc)
        return TaskContext(
            task_id=self.task_id,
            instruction="How many playlists do I have?",
            api_documentation="spotify.show_playlist_library: list playlists",
        )

    def close(self) -> None:
        self._freezer.stop()


@unittest.skipUnless(freeze_time is not None, "freezegun is not installed")
class UnfrozenClockTests(unittest.TestCase):
    def test_frozen_world_pins_the_plain_clocks(self) -> None:
        assert freeze_time is not None
        with freeze_time(_FROZEN):
            first, mono = datetime.now(timezone.utc), time.monotonic()
            time.sleep(_DELAY)
            self.assertEqual(datetime.now(timezone.utc), first)
            self.assertEqual(time.monotonic(), mono)
            began = monotonic()
            time.sleep(_DELAY)
            self.assertGreaterEqual(monotonic() - began, _DELAY * 0.8)
            self.assertGreater(wall_now().year, 2023)

    def test_episode_timestamps_and_latency_stay_real_inside_a_frozen_world(self) -> None:
        config = _config()
        worlds: list[_FreezingWorld] = []

        def factory(task_id: str) -> _FreezingWorld:
            world = _FreezingWorld(task_id)
            worlds.append(world)
            return world

        agent = SmolagentsOpenAICompatibleAgent("zai", _KEY)
        agent.set_mode("plan")
        runtime = RuntimeDependencies(session_factory=factory, agent=agent, clock=wall_now)
        before = wall_now()
        with patch("urllib.request.urlopen", side_effect=_slow_urlopen):
            episode = run_episode(
                "task-1",
                config,
                "plan",
                run=new_run_identity(config),
                runtime=runtime,
                scenario_id="scenario-1",
            )
        after = wall_now()
        self.assertEqual(worlds[0].frozen_now, datetime.fromisoformat(_FROZEN))
        self.assertLessEqual(before, episode.started_at)
        self.assertLessEqual(episode.ended_at, after)
        self.assertTrue(episode.model_steps)
        for step in episode.model_steps:
            self.assertLessEqual(before, step.started_at)
            self.assertLessEqual(step.started_at, after)
            self.assertGreaterEqual(step.latency_seconds, _DELAY * 0.8)
        for call in episode.provider_calls:
            self.assertGreaterEqual(call.latency_seconds, _DELAY * 0.8)


if __name__ == "__main__":
    unittest.main()
