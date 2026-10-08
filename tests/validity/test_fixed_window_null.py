from __future__ import annotations

import itertools
import unittest
from fractions import Fraction

from llm_behavior_ci.lifecycle.canary import _FixedWindowCanary
from llm_behavior_ci.stats.evidence import PairedSuccess


def _exact_rollback_probability(
    success_probabilities: tuple[Fraction, ...],
    *,
    harm_margin: float,
) -> Fraction:
    horizon = len(success_probabilities)
    outcomes = tuple(itertools.product((0, 1), repeat=2))
    total = Fraction(0)
    for draw in itertools.product(outcomes, repeat=horizon):
        weight = Fraction(1)
        for (reference, candidate), p in zip(draw, success_probabilities):
            weight *= (p if reference else 1 - p) * (p if candidate else 1 - p)
        detector = _FixedWindowCanary(harm_margin=harm_margin, horizon_episodes=horizon)
        evidence = None
        for reference, candidate in draw:
            evidence = detector.update(
                PairedSuccess(candidate=float(candidate), reference=float(reference))
            )
        assert evidence is not None
        if evidence.alarm:
            total += weight
    return total


class FixedWindowExactNullTests(unittest.TestCase):
    """Exact rollback rate of the heuristic comparator on identical candidates."""

    def test_identical_candidate_rollback_rate_at_horizon_three(self) -> None:
        half = Fraction(1, 2)
        self.assertEqual(
            _exact_rollback_probability((half, half, half), harm_margin=0.2),
            Fraction(11, 32),
        )
        self.assertEqual(
            _exact_rollback_probability(
                (Fraction(0), Fraction(1), Fraction(3, 4)), harm_margin=0.2
            ),
            Fraction(3, 16),
        )

    def test_rate_far_exceeds_the_recorded_alpha(self) -> None:
        rate = _exact_rollback_probability(
            (Fraction(9, 10),) * 3, harm_margin=0.2
        )
        self.assertGreater(rate, Fraction(1, 20))

    def test_evidence_never_carries_a_p_value(self) -> None:
        detector = _FixedWindowCanary(harm_margin=0.2, horizon_episodes=3)
        evidence = None
        for reference, candidate in ((1.0, 0.0), (1.0, 1.0), (0.0, 0.0)):
            evidence = detector.update(PairedSuccess(candidate=candidate, reference=reference))
        assert evidence is not None
        self.assertTrue(evidence.alarm)
        self.assertIsNone(evidence.p_value)
        self.assertAlmostEqual(evidence.estimate, -1.0 / 3.0)


if __name__ == "__main__":
    unittest.main()
