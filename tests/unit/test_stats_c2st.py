from __future__ import annotations

import unittest

from llm_behavior_ci.stats.c2st import C2STError, classifier_two_sample_test


class C2STTests(unittest.TestCase):
    def test_identical_clouds_are_reproducible(self) -> None:
        cloud = ((0.0, 0.0), (1.0, 1.0), (0.5, 0.25))
        first = classifier_two_sample_test(
            cloud,
            cloud,
            permutations=19,
            seed=1,
            steps=20,
        )
        second = classifier_two_sample_test(
            cloud,
            cloud,
            permutations=19,
            seed=1,
            steps=20,
        )
        self.assertGreater(first.p_value, 0.0)
        self.assertLessEqual(first.p_value, 1.0)
        self.assertEqual(first, second)

    def test_zero_permutations_raises(self) -> None:
        cloud = ((0.0, 0.0), (1.0, 1.0))
        with self.assertRaises(C2STError):
            classifier_two_sample_test(cloud, cloud, permutations=0, seed=1)

    def test_ragged_rows_raise(self) -> None:
        with self.assertRaises(C2STError):
            classifier_two_sample_test(
                ((0.0, 0.0), (1.0,)),
                ((0.0, 0.0), (1.0, 1.0)),
                permutations=5,
                seed=1,
            )


if __name__ == "__main__":
    unittest.main()
