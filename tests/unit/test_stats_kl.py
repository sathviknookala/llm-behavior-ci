from __future__ import annotations

import math
import unittest

from llm_behavior_ci.stats.kl import (
    NextTokenKLError,
    TruncatedKLError,
    next_token_kl,
    truncated_next_token_kl,
)


class NextTokenKLTests(unittest.TestCase):
    def test_equal_full_distributions_return_mean_zero(self) -> None:
        logs = ((math.log(0.25), math.log(0.75)), (math.log(0.5), math.log(0.5)))
        result = next_token_kl(logs, logs)
        self.assertEqual(result.mean_kl_nats, 0.0)
        truncated = truncated_next_token_kl(logs, logs)
        self.assertEqual(truncated.mean_kl_nats, 0.0)

    def test_candidate_missing_mass_raises(self) -> None:
        production = ((0.0, -1.0),)
        candidate = ((0.0, -math.inf),)
        with self.assertRaises(NextTokenKLError):
            next_token_kl(production, candidate)
        with self.assertRaises(TruncatedKLError):
            truncated_next_token_kl(production, candidate)

    def test_truncated_approximation_is_top_k(self) -> None:
        logs = ((math.log(0.4), math.log(0.6)),)
        result = truncated_next_token_kl(logs, logs)
        self.assertEqual(result.approximation, "top_k")

    def test_vocabulary_size_smaller_than_support_raises(self) -> None:
        logs = ((math.log(0.4), math.log(0.6)),)
        with self.assertRaises(TruncatedKLError):
            truncated_next_token_kl(logs, logs, vocabulary_size=1)

    def test_unequal_position_counts_raise(self) -> None:
        one = ((math.log(0.5), math.log(0.5)),)
        two = (
            (math.log(0.5), math.log(0.5)),
            (math.log(0.5), math.log(0.5)),
        )
        with self.assertRaises(NextTokenKLError):
            next_token_kl(one, two)
        with self.assertRaises(TruncatedKLError):
            truncated_next_token_kl(one, two)


if __name__ == "__main__":
    unittest.main()
