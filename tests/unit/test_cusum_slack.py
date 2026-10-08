from __future__ import annotations

import unittest

from llm_behavior_ci.config import ConfigError, MonitorSettings, StoppingRule
from llm_behavior_ci.lifecycle.detectors import build_detector


def _rule(**overrides: object) -> StoppingRule:
    values: dict[str, object] = {
        "name": "cusum",
        "alpha": 0.05,
        "horizon_episodes": 100,
        "threshold": 5.0,
    }
    values.update(overrides)
    return StoppingRule(**values)  # type: ignore[arg-type]


def _first_alarm(rule: StoppingRule, values: list[float], baseline: float) -> int | None:
    detector = build_detector(rule, signal="tool_error_count", baseline=baseline)
    for index, value in enumerate(values, start=1):
        if detector.update(value).alarm:
            return index
    return None


class CusumSlackSettingsTests(unittest.TestCase):
    def test_slack_round_trips_and_is_recorded(self) -> None:
        rule = _rule(slack=0.25)
        self.assertEqual(rule.to_dict()["slack"], 0.25)
        self.assertEqual(StoppingRule.from_dict(rule.to_dict()), rule)
        settings = MonitorSettings(
            reference_configuration_hash="a" * 64,
            outcome_delay_seconds=0.0,
            signals=("tool_error_count",),
            stopping_rules=(rule,),
        )
        self.assertEqual(settings.to_dict()["stopping_rules"][0]["slack"], 0.25)
        self.assertEqual(MonitorSettings.from_dict(settings.to_dict()), settings)

    def test_unset_slack_keeps_the_recorded_payload(self) -> None:
        payload = _rule().to_dict()
        self.assertNotIn("slack", payload)
        self.assertEqual(
            payload,
            {"name": "cusum", "alpha": 0.05, "horizon_episodes": 100, "threshold": 5.0},
        )

    def test_slack_is_cusum_only_and_nonnegative(self) -> None:
        with self.assertRaises(ConfigError):
            StoppingRule(name="fixed_window", alpha=0.05, horizon_episodes=3, slack=0.1)
        with self.assertRaises(ConfigError):
            _rule(slack=-0.1)


class CusumSlackDetectorTests(unittest.TestCase):
    def test_configured_slack_reaches_the_detector(self) -> None:
        drift = [1.0] * 100
        self.assertEqual(_first_alarm(_rule(), drift, baseline=8.0 / 9.0), 46)
        self.assertIsNone(_first_alarm(_rule(slack=0.25), drift, baseline=8.0 / 9.0))

    def test_slack_does_not_hide_a_large_shift(self) -> None:
        shifted = [11.0] * 3
        self.assertEqual(_first_alarm(_rule(slack=0.25), shifted, baseline=8.0 / 9.0), 1)


if __name__ == "__main__":
    unittest.main()
