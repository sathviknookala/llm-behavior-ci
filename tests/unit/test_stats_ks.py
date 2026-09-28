from __future__ import annotations

import unittest

from llm_behavior_ci.stats.ks import KSError, ks_two_sample


class KSTests(unittest.TestCase):
    def test_identical_samples_have_statistic_zero(self) -> None:
        result = ks_two_sample((1.0, 2.0, 3.0), (1.0, 2.0, 3.0))
        self.assertEqual(result.statistic, 0)
        self.assertEqual(result.p_value, 1)

    def test_separated_samples_have_statistic_one(self) -> None:
        result = ks_two_sample((0.0, 0.0), (1.0, 1.0))
        self.assertEqual(result.statistic, 1)

    def test_empty_input_raises(self) -> None:
        with self.assertRaises(KSError):
            ks_two_sample((), (1.0,))
        with self.assertRaises(KSError):
            ks_two_sample((1.0,), ())


if __name__ == "__main__":
    unittest.main()
