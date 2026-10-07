import json
import unittest
from datetime import datetime, timezone
from pathlib import Path

try:
    from freezegun import freeze_time
except ImportError:
    freeze_time = None

from llm_behavior_ci.config import RunConfiguration
from llm_behavior_ci.runtime.aa_capture import live_runtime_factories
from llm_behavior_ci.runtime.clock import wall_now

_ROOT = Path(__file__).resolve().parents[2]


def _configuration() -> RunConfiguration:
    document = json.loads(
        (_ROOT / "configs" / "models" / "qwen3_4b_production.json").read_text(encoding="utf-8")
    )
    task = json.loads((_ROOT / "configs" / "tasks" / "train_smoke.json").read_text(encoding="utf-8"))
    task.pop("scenario_count", None)
    document["task"] = task
    document["run_seed"] = 17
    document["git_commit"] = "a" * 40
    return RunConfiguration.from_dict(document)


@unittest.skipUnless(freeze_time is not None, "freezegun is not installed")
class LiveClockTests(unittest.TestCase):
    def test_capture_clock_stays_on_the_real_clock_while_datetime_now_is_frozen(self) -> None:
        reference, _candidate = live_runtime_factories(
            _configuration(), "http://127.0.0.1:8000"
        )
        runtime = reference("plan")
        with freeze_time("2020-01-01T00:00:00+00:00"):
            frozen = datetime.now(timezone.utc)
            live = runtime.clock()
            direct = wall_now()
        self.assertEqual(frozen.year, 2020)
        self.assertGreater(live.year, 2020)
        self.assertEqual(live.year, direct.year)
        self.assertLess(abs((live - direct).total_seconds()), 2)
