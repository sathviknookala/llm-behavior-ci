import math
import unittest
from unittest.mock import patch

from llm_behavior_ci.lifecycle.offline_gate import run_offline_gate
from llm_behavior_ci.stats.bootstrap import paired_bootstrap
from llm_behavior_ci.stats.kl import NextTokenKLError, next_token_kl
from llm_behavior_ci.stats.mmd import MMDError, mmd_permutation_test


class OfflineGateTests(unittest.TestCase):
    def test_paired_bootstrap_candidate_vs_production(self) -> None:
        result = paired_bootstrap(
            candidate=(0.6, 0.8, 1.0, 1.2),
            production=(1.0, 1.0, 1.0, 1.0),
            confidence_level=0.95,
            resamples=1_000,
            seed=7,
        )
        self.assertAlmostEqual(result.mean_difference, -0.1)
        self.assertLess(result.confidence_low, result.mean_difference)
        self.assertGreater(result.confidence_high, result.mean_difference)

    def test_next_token_kl_candidate_vs_production(self) -> None:
        result = next_token_kl(
            production_log_probabilities=((math.log(0.5), math.log(0.5)),),
            candidate_log_probabilities=((math.log(0.25), math.log(0.75)),),
        )
        expected = 0.5 * math.log(2.0) + 0.5 * math.log(2.0 / 3.0)
        self.assertAlmostEqual(result.mean_kl_nats, expected)
        self.assertEqual(result.position_kl_nats, (result.mean_kl_nats,))

    def test_next_token_kl_rejects_underflowed_support_mismatch(self) -> None:
        with self.assertRaises(NextTokenKLError):
            next_token_kl(
                production_log_probabilities=((0.0, -1_000.0),),
                candidate_log_probabilities=((0.0, -math.inf),),
            )

    def test_mmd_candidate_vs_production(self) -> None:
        production = tuple((value / 10.0,) for value in range(8))
        candidate = tuple((5.0 + value / 10.0,) for value in range(8))
        result = mmd_permutation_test(
            production=production,
            candidate=candidate,
            bandwidth=1.0,
            permutations=999,
            seed=7,
        )
        self.assertGreater(result.mmd_squared, 0.0)
        self.assertLessEqual(result.p_value, 0.05)

        null_result = mmd_permutation_test(
            production=production,
            candidate=production,
            bandwidth=1.0,
            permutations=999,
            seed=7,
        )
        self.assertAlmostEqual(null_result.mmd_squared, 0.0)
        self.assertAlmostEqual(null_result.p_value, 1.0)

    def test_mmd_rejects_unrepresentable_bandwidths(self) -> None:
        samples = ((0.0,), (1.0,))
        with self.assertRaises(MMDError):
            mmd_permutation_test(
                production=samples,
                candidate=samples,
                bandwidth=1e-300,
                permutations=9,
                seed=7,
            )
        with self.assertRaises(MMDError):
            mmd_permutation_test(
                production=samples,
                candidate=samples,
                bandwidth=1e308,
                permutations=9,
                seed=7,
            )

    def test_orchestrator_calls_all_three_checks(self) -> None:
        with (
            patch(
                "llm_behavior_ci.lifecycle.offline_gate.paired_bootstrap",
                wraps=paired_bootstrap,
            ) as bootstrap_call,
            patch(
                "llm_behavior_ci.lifecycle.offline_gate.next_token_kl",
                wraps=next_token_kl,
            ) as kl_call,
            patch(
                "llm_behavior_ci.lifecycle.offline_gate.mmd_permutation_test",
                wraps=mmd_permutation_test,
            ) as mmd_call,
        ):
            result = run_offline_gate()
        bootstrap_call.assert_called_once()
        kl_call.assert_called_once()
        mmd_call.assert_called_once()
        self.assertTrue(result.bootstrap_passed)
        self.assertTrue(result.next_token_kl_passed)
        self.assertTrue(result.mmd_passed)
        self.assertTrue(result.passed)

    def test_orchestrator_fails_when_a_check_rejects(self) -> None:
        with patch(
            "llm_behavior_ci.lifecycle.offline_gate.DEMO_MINIMUM_SCORE_DELTA",
            0.01,
        ):
            result = run_offline_gate()
        self.assertFalse(result.bootstrap_passed)
        self.assertFalse(result.passed)


if __name__ == "__main__":
    unittest.main()
