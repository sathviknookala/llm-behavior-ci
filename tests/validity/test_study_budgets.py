from __future__ import annotations

import unittest

from llm_behavior_ci.experiments.validation import (
    STUDY_BUDGETS,
    AAContext,
    ValidationError,
    method_spec,
    validate_method,
)


def _canary(study: str, horizon: int = 2):
    return method_spec(
        "sequential_canary",
        {
            "alpha": 0.05,
            "harm_margin": 0.0,
            "horizon_episodes": horizon,
            "pair_value": 1.0,
        },
        required_checks=("null_false_alarm",),
        study=study,
        null_sample_size=horizon,
        uncertainty_level=0.95,
        null_draw="constant",
        alpha=0.05,
        horizon=horizon,
        false_alarm_tolerance=0.0,
    )


class StudyBudgetTests(unittest.TestCase):
    def test_cpu_fast_and_gpu_share_a_smaller_budget_than_simulation(self) -> None:
        self.assertEqual(STUDY_BUDGETS["cpu_fast"], STUDY_BUDGETS["gpu"])
        self.assertGreater(
            STUDY_BUDGETS["simulation"].max_seeds,
            STUDY_BUDGETS["cpu_fast"].max_seeds,
        )
        with self.assertRaises(ValidationError):
            validate_method(
                _canary("cpu_fast"),
                null_seeds=tuple(range(STUDY_BUDGETS["cpu_fast"].max_seeds + 1)),
                reference_cases=(),
                aa_observations=(),
            )
        report = validate_method(
            _canary("simulation"),
            null_seeds=tuple(range(STUDY_BUDGETS["cpu_fast"].max_seeds + 1)),
            reference_cases=(),
            aa_observations=(),
        )
        self.assertEqual(report.study, "simulation")
        self.assertEqual(report.sample_count, STUDY_BUDGETS["cpu_fast"].max_seeds + 1)
        self.assertEqual(report.checks[0].status, "passed")
        self.assertFalse(report.benchmark_eligible)
        with self.assertRaises(ValidationError):
            validate_method(
                _canary("simulation"),
                null_seeds=tuple(range(STUDY_BUDGETS["simulation"].max_seeds + 1)),
                reference_cases=(),
                aa_observations=(),
            )

    def test_gpu_study_does_not_invent_a_hardware_reading(self) -> None:
        with self.assertRaises(ValidationError):
            validate_method(
                _canary("gpu"),
                null_seeds=(1,),
                reference_cases=(),
                aa_observations=(),
                aa_context=AAContext(
                    provenance="gpu",
                    hardware_observed=False,
                    memory_used_mib=10,
                ),
            )
        report = validate_method(
            _canary("gpu"),
            null_seeds=(1,),
            reference_cases=(),
            aa_observations=(),
            aa_context=AAContext(provenance="gpu", hardware_observed=False),
        )
        self.assertFalse(report.gpu_evidence)
        self.assertIsNone(report.aa.memory_used_mib)
        self.assertFalse(report.benchmark_eligible)
        gpu_check = next(item for item in report.checks if item.name == "gpu_evidence")
        self.assertEqual(gpu_check.status, "unavailable")
        self.assertIsNone(gpu_check.estimate)


if __name__ == "__main__":
    unittest.main()
