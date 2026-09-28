from __future__ import annotations

import math
import unittest

from llm_behavior_ci.stats.adwin import ADWIN
from llm_behavior_ci.stats.canary import SequentialCanaryTest
from llm_behavior_ci.stats.confidence_sequence import BoundedMeanCS, stitched_radius
from llm_behavior_ci.stats.cusum import CUSUM
from llm_behavior_ci.stats.e_detector import BettingEDetector
from llm_behavior_ci.stats.evidence import PairedSuccess, StatisticsError
from llm_behavior_ci.stats.harmful_shift import HarmfulShiftTest


class DetectorTests(unittest.TestCase):
    def test_cusum_increase_crosses_on_second_observation(self) -> None:
        detector = CUSUM(target=0, slack=0, threshold=2, direction="increase")
        other = CUSUM(target=0, slack=0, threshold=2, direction="increase")
        first = detector.update(1)
        self.assertAlmostEqual(first.estimate, 1, places=6)
        self.assertFalse(first.alarm)
        second = detector.update(1)
        self.assertAlmostEqual(second.estimate, 2, places=6)
        self.assertTrue(second.alarm)
        self.assertEqual(other.snapshot()["sample_size"], 0)
        self.assertFalse(other.snapshot()["alarm"])
        detector.reset()
        snapshot = detector.snapshot()
        self.assertEqual(snapshot["sample_size"], 0)
        self.assertFalse(snapshot["alarm"])
        self.assertEqual(other.snapshot()["sample_size"], 0)

    def test_adwin_detects_shift_and_ignores_constant_stream(self) -> None:
        shifted = ADWIN(delta=0.5)
        evidences = []
        for value in (0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0):
            evidences.append(shifted.update(value))
        alarmed = [evidence for evidence in evidences if evidence.alarm]
        self.assertTrue(alarmed)
        self.assertGreater(dict(alarmed[-1].details)["dropped"], 0.0)
        epsilon = math.sqrt((1.0 / 5.0) * math.log(80.0))
        self.assertLess(epsilon, 1.0)
        constant = ADWIN(delta=0.5)
        for _ in range(10):
            evidence = constant.update(0.0)
            self.assertFalse(evidence.alarm)

    def test_stitched_radius_matches_formula(self) -> None:
        log_term = math.log(1.0 / 0.05) + math.log(math.log(math.e * 1))
        expected = math.sqrt((1.0 * 1.0 / 4.0) * (2.0 * log_term) / 1)
        self.assertAlmostEqual(stitched_radius(1, 0.05, 1.0), expected, places=6)

    def test_bounded_mean_cs_rejects_outside_interval(self) -> None:
        detector = BoundedMeanCS(alpha=0.05, lower=0.0, upper=1.0)
        with self.assertRaises(StatisticsError):
            detector.update(1.5)

    def test_betting_e_detector_first_update_and_reset(self) -> None:
        detector = BettingEDetector(
            null_mean=0.5,
            alpha=0.05,
            lower=0,
            upper=1,
            direction="above",
        )
        first = detector.update(0.5)
        self.assertAlmostEqual(dict(first.details)["wealth"], 1, places=6)
        detector.reset()
        self.assertAlmostEqual(detector.snapshot()["wealth"], 1, places=6)

    def test_harmful_shift_rejects_candidate_outside_unit_interval(self) -> None:
        detector = HarmfulShiftTest(alpha=0.05, harm_margin=0.0)
        with self.assertRaises(StatisticsError):
            detector.update(PairedSuccess(candidate=2, reference=0.5))

    def test_sequential_canary_stops_at_horizon_without_alarm(self) -> None:
        detector = SequentialCanaryTest(
            alpha=0.05,
            harm_margin=0.0,
            horizon_episodes=2,
        )
        detector.update(PairedSuccess(1.0, 1.0))
        second = detector.update(PairedSuccess(1.0, 1.0))
        self.assertAlmostEqual(dict(second.details)["stopped_at_horizon"], 1.0, places=6)
        self.assertFalse(second.alarm)

    def test_sequential_canary_instances_do_not_share_wealth(self) -> None:
        first = SequentialCanaryTest(
            alpha=0.05,
            harm_margin=0.0,
            horizon_episodes=4,
        )
        second = SequentialCanaryTest(
            alpha=0.05,
            harm_margin=0.0,
            horizon_episodes=4,
        )
        first.update(PairedSuccess(1.0, 1.0))
        self.assertEqual(second.snapshot()["sample_size"], 0)


if __name__ == "__main__":
    unittest.main()
