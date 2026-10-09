from __future__ import annotations

import unittest

from llm_behavior_ci.experiments.qualification_batch import DEV_ARMS
from llm_behavior_ci.experiments.qualification_criteria import (
    HARM_PARAMETERS,
    HEALTHY_ARMS,
    exact_mcnemar_p,
    execution_criteria,
    failure_rule,
    harm_classification,
    plan_criteria,
    wilson_bounds,
)


def _table(pairs: list[tuple[bool | None, bool | None]]) -> list[dict[str, dict[str, object]]]:
    return [
        {
            DEV_ARMS[0]: {"state": "completed", "success": first},
            DEV_ARMS[1]: {"state": "completed", "success": second},
            DEV_ARMS[2]: {"state": "completed", "success": first},
            DEV_ARMS[3]: {"state": "completed", "success": second},
        }
        for first, second in pairs
    ]


def _scale(outcomes: int = 38, scenarios: int = 19, target: float = 0.6) -> dict[str, object]:
    sigma = 0.5
    return {
        "status": "estimated",
        "scored_outcomes": outcomes,
        "scenarios": scenarios,
        "target": target,
        "sigma": sigma,
        "slack": 0.5 * sigma,
        "threshold": 5.0 * sigma,
    }


def _label(low: float, high: float, *, harmful: bool = False) -> dict[str, object]:
    return {
        "status": "measured",
        "effect_estimate": (low + high) / 2.0,
        "interval_low": low,
        "interval_high": high,
        "harmful": harmful,
        "margin": 0.2,
    }


def _criteria(pairs, labels=None, parameters=None, scale=None):
    return execution_criteria(
        _table(pairs),
        _scale() if scale is None else scale,
        {"noop": _label(-0.1, 0.1)} if labels is None else labels,
        harm_parameters=dict(HARM_PARAMETERS) if parameters is None else parameters,
    )


class ExchangeabilityTests(unittest.TestCase):
    def test_exact_mcnemar_values(self) -> None:
        self.assertEqual(exact_mcnemar_p(0, 0), 1.0)
        self.assertEqual(exact_mcnemar_p(1, 1), 1.0)
        self.assertAlmostEqual(exact_mcnemar_p(0, 5), 0.0625)
        self.assertAlmostEqual(exact_mcnemar_p(6, 0), 0.03125)
        self.assertAlmostEqual(exact_mcnemar_p(1, 6), 0.125)

    def test_zero_discordant_pairs_pass_with_p_one(self) -> None:
        criteria = _criteria([(True, True)] * 10 + [(False, False)] * 9)
        exchange = criteria["P-AA-EXEC-1"]
        self.assertEqual(exchange["discordant"], 0)
        self.assertEqual(exchange["p_value"], 1.0)
        self.assertEqual(exchange["status"], "pass")
        self.assertEqual(exchange["reading"], "no detected directional imbalance")

    def test_zero_discordance_is_still_inconclusive_without_completeness(self) -> None:
        criteria = _criteria([(True, True)] * 16 + [(True, None), (None, False), (None, None)])
        self.assertEqual(criteria["P-AA-EXEC-3"]["status"], "inconclusive")
        self.assertEqual(criteria["P-AA-EXEC-3"]["both_scored"], 16)
        self.assertEqual(criteria["P-AA-EXEC-1"]["status"], "inconclusive")
        self.assertNotIn("p_value", criteria["P-AA-EXEC-1"])

    def test_one_sided_discordance_fails(self) -> None:
        criteria = _criteria([(True, False)] * 6 + [(True, True)] * 13)
        self.assertEqual(criteria["P-AA-EXEC-3"]["status"], "pass")
        self.assertEqual(criteria["P-AA-EXEC-1"]["status"], "fail")
        self.assertEqual(failure_rule(criteria)["failed"], ["P-AA-EXEC-1"])


class CanaryAATests(unittest.TestCase):
    def test_all_losses_roll_back(self) -> None:
        criteria = _criteria([(True, False)] * 12 + [(True, True)] * 7)
        self.assertEqual(criteria["P-AA-EXEC-2"]["status"], "fail")
        self.assertEqual(criteria["P-AA-EXEC-2"]["rollback_at"], 12)

    def test_concordant_pairs_do_not_roll_back(self) -> None:
        criteria = _criteria([(True, True)] * 19)
        self.assertEqual(criteria["P-AA-EXEC-2"]["status"], "pass")
        self.assertIsNone(criteria["P-AA-EXEC-2"]["rollback_at"])

    def test_fewer_than_twelve_scored_pairs_is_inconclusive(self) -> None:
        criteria = _criteria([(True, True)] * 11 + [(None, True)] * 8)
        self.assertEqual(criteria["P-AA-EXEC-2"]["status"], "inconclusive")


class HarmClassificationTests(unittest.TestCase):
    def test_classes_follow_the_interval(self) -> None:
        self.assertEqual(harm_classification(_label(-0.6, -0.2, harmful=True)), "confirmed_harmful")
        self.assertEqual(harm_classification(_label(-0.19, 0.1)), "confirmed_not_harmful")
        self.assertEqual(harm_classification(_label(-0.2, 0.1)), "indeterminate")
        self.assertEqual(harm_classification({"status": "unmeasurable"}), "not_classified")

    def test_a_measured_but_indeterminate_label_stays_measured(self) -> None:
        indeterminate = _label(-0.45, 0.05, harmful=True)
        labels = {"noop": _label(-0.1, 0.1), "regression": indeterminate}
        criteria = _criteria([(True, True)] * 19, labels=labels)
        harm = criteria["P-HARM-1"]
        self.assertEqual(harm["status"], "reported")
        self.assertEqual(harm["classifications"]["regression"], "indeterminate")
        self.assertEqual(harm["classifications"]["noop"], "confirmed_not_harmful")
        self.assertEqual(labels["regression"]["status"], "measured")
        self.assertTrue(labels["regression"]["harmful"])
        self.assertEqual(failure_rule(criteria)["failed"], [])

    def test_unmeasurable_labels_are_not_classified(self) -> None:
        criteria = _criteria([(True, True)] * 19, labels={"regression": {"status": "unmeasurable"}})
        self.assertEqual(criteria["P-HARM-1"]["classifications"]["regression"], "not_classified")

    def test_other_harm_parameters_fail(self) -> None:
        parameters = dict(HARM_PARAMETERS, resamples=500)
        criteria = _criteria([(True, True)] * 19, parameters=parameters)
        self.assertEqual(criteria["P-HARM-2"]["status"], "fail")
        self.assertEqual(_criteria([(True, True)] * 19)["P-HARM-2"]["status"], "pass")


class CusumCriteriaTests(unittest.TestCase):
    def test_sample_floor_keeps_the_scale_provisional(self) -> None:
        self.assertEqual(_criteria([(True, True)] * 19, scale=_scale(outcomes=33))["P-CUSUM-1"]["scale"], "provisional")
        self.assertEqual(_criteria([(True, True)] * 19, scale=_scale(scenarios=16))["P-CUSUM-1"]["status"], "inconclusive")
        self.assertEqual(_criteria([(True, True)] * 19)["P-CUSUM-1"]["scale"], "final")

    def test_sensitivity_uses_scenario_count_wilson_bounds_and_states_dependence(self) -> None:
        first = _criteria([(True, True)] * 19)["P-CUSUM-2"]
        second = _criteria([(True, True)] * 19)["P-CUSUM-2"]
        self.assertEqual(first, second)
        self.assertEqual(first["status"], "reported")
        self.assertEqual(first["wilson_trials"], 19)
        self.assertEqual((first["wilson_low"], first["wilson_high"]), wilson_bounds(0.6, 19))
        self.assertIn("binomial", first["model"])
        self.assertIn("19 scenario clusters", first["dependence"])
        rates = first["null_alarm_rate"]
        self.assertLess(rates["at_wilson_high"], rates["at_estimate"])
        self.assertLess(rates["at_estimate"], rates["at_wilson_low"])

    def test_no_positive_scale_is_inconclusive(self) -> None:
        scale = {"status": "unavailable", "scored_outcomes": 1, "scenarios": 1}
        self.assertEqual(_criteria([(True, True)] * 19, scale=scale)["P-CUSUM-2"]["status"], "inconclusive")


class PlanCriteriaTests(unittest.TestCase):
    def test_complete_pairs_and_gate_outcome(self) -> None:
        pairs = [("plan", "plan")] * 30
        criteria = plan_criteria(pairs, {"status": "decided", "outcome": "PASS"}, plan_complete=bool)
        self.assertEqual(criteria["P-AA-PLAN-2"]["status"], "pass")
        self.assertEqual(criteria["P-AA-PLAN-1"]["status"], "pass")
        blocked = plan_criteria(pairs, {"status": "decided", "outcome": "BLOCK"}, plan_complete=bool)
        self.assertEqual(blocked["P-AA-PLAN-1"]["status"], "fail")

    def test_a_missing_or_empty_plan_is_inconclusive(self) -> None:
        pairs = [("plan", "plan")] * 29 + [("plan", "")]
        criteria = plan_criteria(pairs, {"status": "incomplete"}, plan_complete=bool)
        self.assertEqual(criteria["P-AA-PLAN-2"]["status"], "inconclusive")
        self.assertEqual(criteria["P-AA-PLAN-1"]["status"], "inconclusive")
        missing = plan_criteria([("plan", "plan")] * 29 + [None], {"status": "execution_failed"}, plan_complete=bool)
        self.assertEqual(missing["P-AA-PLAN-2"]["complete_pairs"], 29)


class ArmNameTests(unittest.TestCase):
    def test_healthy_arms_match_the_batch(self) -> None:
        self.assertEqual(HEALTHY_ARMS, DEV_ARMS[:2])


if __name__ == "__main__":
    unittest.main()
