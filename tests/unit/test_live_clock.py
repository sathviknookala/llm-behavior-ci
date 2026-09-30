import unittest
from datetime import datetime, timezone

try:
    from freezegun import freeze_time
except ImportError:
    freeze_time = None

from llm_behavior_ci.runtime.aa_capture import live_runtime_factory
from llm_behavior_ci.runtime.clock import wall_now


@unittest.skipUnless(freeze_time is not None, "freezegun is not installed")
class LiveClockTests(unittest.TestCase):
    def test_capture_clock_stays_on_the_real_clock_while_datetime_now_is_frozen(self) -> None:
        runtime = live_runtime_factory("http://127.0.0.1:8000")("plan")
        with freeze_time("2020-01-01T00:00:00+00:00"):
            frozen = datetime.now(timezone.utc)
            live = runtime.clock()
            direct = wall_now()
        self.assertEqual(frozen.year, 2020)
        self.assertGreater(live.year, 2020)
        self.assertEqual(live.year, direct.year)
        self.assertLess(abs((live - direct).total_seconds()), 2)
