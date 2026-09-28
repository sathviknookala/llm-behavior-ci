from __future__ import annotations

import random
import unittest

from llm_behavior_ci.stats.bootstrap import (
    PairedBootstrapError,
    cluster_draw_indexes,
    clustered_paired_bootstrap,
    paired_bootstrap,
)


class ClusteredBootstrapTests(unittest.TestCase):
    def test_identical_seed_returns_identical_clustered_result(self) -> None:
        kwargs = {
            "candidate": (1.0, 0.0, 1.0, 0.0),
            "production": (0.0, 1.0, 0.0, 1.0),
            "clusters": ("a", "a", "b", "b"),
            "confidence_level": 0.9,
            "resamples": 40,
            "seed": 11,
        }
        first = clustered_paired_bootstrap(**kwargs)
        second = clustered_paired_bootstrap(**kwargs)
        self.assertEqual(first, second)

    def test_cluster_draw_indexes_keeps_s1_together(self) -> None:
        rng = random.Random(0)
        for _ in range(30):
            indexes = cluster_draw_indexes(("s1", "s1", "s2"), rng)
            has_zero = 0 in indexes
            has_one = 1 in indexes
            self.assertEqual(has_zero, has_one)

    def test_mismatched_lengths_and_empty_series_raise(self) -> None:
        with self.assertRaises(PairedBootstrapError):
            clustered_paired_bootstrap(
                candidate=(1.0, 0.0),
                production=(1.0,),
                clusters=("a", "b"),
                confidence_level=0.9,
                resamples=10,
                seed=1,
            )
        with self.assertRaises(PairedBootstrapError):
            clustered_paired_bootstrap(
                candidate=(1.0, 0.0),
                production=(1.0, 0.0),
                clusters=("a",),
                confidence_level=0.9,
                resamples=10,
                seed=1,
            )
        with self.assertRaises(PairedBootstrapError):
            clustered_paired_bootstrap(
                candidate=(),
                production=(),
                clusters=(),
                confidence_level=0.9,
                resamples=10,
                seed=1,
            )
        with self.assertRaises(PairedBootstrapError):
            paired_bootstrap(
                candidate=(),
                production=(),
                confidence_level=0.9,
                resamples=10,
                seed=1,
            )

    def test_paired_bootstrap_zero_difference(self) -> None:
        result = paired_bootstrap(
            [1.0],
            [1.0],
            confidence_level=0.9,
            resamples=20,
            seed=1,
        )
        self.assertEqual(result.mean_difference, 0.0)


if __name__ == "__main__":
    unittest.main()
