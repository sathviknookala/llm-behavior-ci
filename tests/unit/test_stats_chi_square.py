from __future__ import annotations

import unittest

from llm_behavior_ci.stats.chi_square import (
    ChiSquareError,
    chi_square_goodness_of_fit,
    chi_square_homogeneity,
)


class ChiSquareTests(unittest.TestCase):
    def test_matching_expected_has_statistic_zero(self) -> None:
        result = chi_square_goodness_of_fit((10, 20, 30), (10.0, 20.0, 30.0))
        self.assertEqual(result.statistic, 0)
        self.assertEqual(result.p_value, 1)

    def test_zero_expected_count_raises(self) -> None:
        with self.assertRaises(ChiSquareError):
            chi_square_goodness_of_fit((10, 20), (10.0, 0.0))

    def test_negative_count_raises(self) -> None:
        with self.assertRaises(ChiSquareError):
            chi_square_goodness_of_fit((10, -1), (10.0, 20.0))

    def test_homogeneous_rows_have_statistic_zero(self) -> None:
        result = chi_square_homogeneity((10, 20), (10, 20))
        self.assertEqual(result.statistic, 0)


if __name__ == "__main__":
    unittest.main()
