from __future__ import annotations

import random
import unittest

from llm_behavior_ci.stats.mmd import (
    MMDError,
    cluster_swap_bits,
    mmd_permutation_test,
)


class MMDTests(unittest.TestCase):
    def test_clusters_none_on_identical_pairs(self) -> None:
        points = ((0.0, 0.0), (1.0, 1.0))
        result = mmd_permutation_test(
            points,
            points,
            bandwidth=1.0,
            permutations=19,
            seed=3,
            clusters=None,
        )
        self.assertGreater(result.p_value, 0.0)
        self.assertLessEqual(result.p_value, 1.0)

    def test_cluster_swap_bits_constant_inside_cluster(self) -> None:
        rng = random.Random(4)
        for _ in range(20):
            bits = cluster_swap_bits(("c", "c", "d"), rng)
            self.assertEqual(bits[0], bits[1])

    def test_wrong_length_clusters_raise(self) -> None:
        points = ((0.0, 0.0), (1.0, 1.0))
        with self.assertRaises(MMDError):
            mmd_permutation_test(
                points,
                points,
                bandwidth=1.0,
                permutations=9,
                seed=1,
                clusters=("only",),
            )


if __name__ == "__main__":
    unittest.main()
