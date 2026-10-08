from __future__ import annotations

import unittest

from llm_behavior_ci.experiments.validation import (
    ValidationError,
    format_validation_summary,
    method_spec,
    method_spec_from_dict,
    public_validation_summary,
    validate_method,
)
from llm_behavior_ci.stats.mmd import paired_permutation_resolution

_SEEDS = tuple(range(1, 41))


def _check(report, name):
    for check in report.checks:
        if check.name == name:
            return check
    raise AssertionError(name)


def _mmd_spec(n: int, alternative_shift: float | None = None):
    return method_spec(
        "mmd_permutation_test",
        {
            "bandwidth": 1.0,
            "permutations": 39,
            "dimension": 2,
            "null_mean": 0.0,
            "null_scale": 1.0,
        },
        required_checks=("null_false_alarm",),
        study="cpu_fast",
        null_sample_size=n,
        uncertainty_level=0.95,
        null_draw="gaussian",
        alpha=0.05,
        false_alarm_tolerance=0.0,
        alternative_shift=alternative_shift,
    )


def _bootstrap_spec(n: int):
    return method_spec(
        "clustered_paired_bootstrap",
        {
            "confidence_level": 0.95,
            "resamples": 100,
            "null_mean": 0.0,
            "null_scale": 1.0,
            "cluster_size": 1,
        },
        required_checks=("null_false_alarm", "coverage"),
        study="cpu_fast",
        null_sample_size=n,
        uncertainty_level=0.95,
        null_draw="gaussian",
        alpha=0.05,
        false_alarm_tolerance=0.02,
        coverage_tolerance=0.02,
    )


def _validate(spec):
    return validate_method(spec, null_seeds=_SEEDS, reference_cases=(), aa_observations=())


class NullControlTests(unittest.TestCase):
    def test_conservative_permutation_test_passes_null_control(self) -> None:
        report = _validate(_mmd_spec(3))
        self.assertEqual(report.null_claim, "at_most")
        null = _check(report, "null_false_alarm")
        self.assertEqual(null.estimate, 0.0)
        self.assertEqual(null.status, "passed")

    def test_small_bootstrap_undercoverage_still_fails(self) -> None:
        report = _validate(_bootstrap_spec(3))
        coverage = _check(report, "coverage")
        self.assertEqual(coverage.status, "failed")
        self.assertLess(coverage.estimate, 0.93)
        self.assertEqual(report.null_claim, "level")


class AlternativePowerTests(unittest.TestCase):
    def test_power_is_reported_apart_from_eligibility_checks(self) -> None:
        report = _validate(_mmd_spec(8, alternative_shift=3.0))
        self.assertIsNotNone(report.power)
        assert report.power is not None
        self.assertEqual(report.power.name, "alternative_power")
        self.assertEqual(report.power.status, "measured")
        self.assertGreater(report.power.estimate, 0.5)
        self.assertNotIn("alternative_power", [check.name for check in report.checks])
        self.assertIn("power:", format_validation_summary(report))
        self.assertIn("power", public_validation_summary(report))

    def test_unresolvable_design_has_no_power(self) -> None:
        report = _validate(_mmd_spec(3, alternative_shift=3.0))
        assert report.power is not None
        self.assertEqual(report.power.estimate, 0.0)
        self.assertGreater(paired_permutation_resolution(3, 39), 0.05)

    def test_power_changes_the_input_hash(self) -> None:
        plain = _validate(_mmd_spec(8))
        shifted = _validate(_mmd_spec(8, alternative_shift=3.0))
        self.assertIsNone(plain.power)
        self.assertNotEqual(plain.input_hash, shifted.input_hash)

    def test_alternative_shift_is_limited_to_two_sample_methods(self) -> None:
        with self.assertRaises(ValidationError):
            method_spec_from_dict(
                {
                    "name": "cusum",
                    "parameters": {
                        "target": 0.0,
                        "slack": 0.0,
                        "threshold": 5.0,
                        "direction": "increase",
                    },
                    "required_checks": ["null_false_alarm"],
                    "study": "cpu_fast",
                    "null_sample_size": 10,
                    "uncertainty_level": 0.95,
                    "null_draw": "constant",
                    "horizon": 10,
                    "alternative_shift": 1.0,
                }
            )
        with self.assertRaises(ValidationError):
            _mmd_spec(8, alternative_shift=0.0)


class ResolutionTests(unittest.TestCase):
    def test_resolution_matches_the_swap_count(self) -> None:
        self.assertAlmostEqual(paired_permutation_resolution(3, 199), (1 + 199 / 4) / 200)
        self.assertGreater(paired_permutation_resolution(5, 199), 0.05)
        self.assertLess(paired_permutation_resolution(6, 199), 0.05)
        self.assertGreater(paired_permutation_resolution(30, 19), 0.05)


if __name__ == "__main__":
    unittest.main()
