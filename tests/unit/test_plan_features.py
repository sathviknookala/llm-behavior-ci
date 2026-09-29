from __future__ import annotations

import unittest

from llm_behavior_ci.lifecycle.plan_features import (
    PLAN_FEATURE_SCHEMA_VERSION,
    PLAN_TEXT_FEATURES,
    PlanFeatureError,
    SEMANTIC_PLAN_FEATURES,
    STRUCTURAL_PLAN_FEATURES,
    extract_plan_features,
    semantic_plan_features,
    structural_plan_features,
)
from llm_behavior_ci.tasks.plan_specs import PlanSpecError, TaskPlanSpec


def _spec(**overrides: object) -> TaskPlanSpec:
    values: dict[str, object] = {
        "task_id": "task-a",
        "available_tools": ("calendar.list_events", "calendar.create_event"),
        "subgoal_keywords": (
            ("list events", "list the events"),
            ("create an event", "add an event"),
        ),
        "required_entities": ("calendar",),
        "dependency_pairs": (("calendar.list_events", "calendar.create_event"),),
    }
    values.update(overrides)
    return TaskPlanSpec(**values)


class SchemaTests(unittest.TestCase):
    def test_schema_version_is_stable_and_named(self) -> None:
        self.assertEqual(PLAN_FEATURE_SCHEMA_VERSION, "plan-features-v1")
        self.assertTrue(STRUCTURAL_PLAN_FEATURES)
        self.assertTrue(SEMANTIC_PLAN_FEATURES)
        self.assertEqual(
            PLAN_TEXT_FEATURES, STRUCTURAL_PLAN_FEATURES | SEMANTIC_PLAN_FEATURES
        )
        self.assertFalse(STRUCTURAL_PLAN_FEATURES & SEMANTIC_PLAN_FEATURES)


class StructuralFeatureTests(unittest.TestCase):
    def test_deterministic_repeat(self) -> None:
        text = "1. calendar.list_events: list events\n2. calendar.create_event: add an event\n"
        first = structural_plan_features(text)
        second = structural_plan_features(text)
        self.assertEqual(first, second)

    def test_rejects_non_string(self) -> None:
        with self.assertRaises(PlanFeatureError):
            structural_plan_features(1234)  # type: ignore[arg-type]


class SemanticFeatureDeterminismTests(unittest.TestCase):
    def test_deterministic_repeat(self) -> None:
        spec = _spec()
        text = "1. calendar.list_events: list events\n2. calendar.create_event: add an event\n"
        first = semantic_plan_features(text, spec)
        second = semantic_plan_features(text, spec)
        self.assertEqual(first, second)

    def test_rejects_non_spec(self) -> None:
        with self.assertRaises(PlanFeatureError):
            semantic_plan_features("1. do it", "not-a-spec")  # type: ignore[arg-type]


class SemanticallyDifferentPlansTests(unittest.TestCase):
    def test_similar_length_plans_differ_semantically(self) -> None:
        spec = _spec()
        covering = "1. calendar.list_events: list the events\n2. calendar.create_event: create an event\n"
        unrelated = "1. weather.get_forecast: check the weather\n2. weather.get_alerts: check for alerts\n"
        self.assertEqual(len(covering), len(unrelated))
        covering_features = semantic_plan_features(covering, spec)
        unrelated_features = semantic_plan_features(unrelated, spec)
        self.assertEqual(
            structural_plan_features(covering)["char_count"],
            structural_plan_features(unrelated)["char_count"],
        )
        self.assertGreater(
            covering_features["requirement_coverage_fraction"],
            unrelated_features["requirement_coverage_fraction"],
        )
        self.assertGreater(
            covering_features["tool_reference_fraction"],
            unrelated_features["tool_reference_fraction"],
        )
        self.assertEqual(unrelated_features["valid_tool_reference_count"], 0.0)


class FormattingInvarianceTests(unittest.TestCase):
    def test_whitespace_and_casing_do_not_dominate_semantic_score(self) -> None:
        spec = _spec()
        compact = "1. calendar.list_events: list events\n2. calendar.create_event: add an event"
        reformatted = (
            "1.   CALENDAR.LIST_EVENTS:   List   Events\n\n"
            "2.   Calendar.Create_Event:   Add   An   Event\n\n"
        )
        compact_features = semantic_plan_features(compact, spec)
        reformatted_features = semantic_plan_features(reformatted, spec)
        self.assertEqual(
            compact_features["requirement_coverage_fraction"],
            reformatted_features["requirement_coverage_fraction"],
        )
        self.assertEqual(
            compact_features["entity_coverage_fraction"],
            reformatted_features["entity_coverage_fraction"],
        )
        self.assertNotEqual(
            structural_plan_features(compact)["char_count"],
            structural_plan_features(reformatted)["char_count"],
        )


class InvalidToolReferenceTests(unittest.TestCase):
    def test_invalid_tool_reference_is_represented(self) -> None:
        spec = _spec()
        text = "1. calendar.list_events: list events\n2. calendar.delete_everything: wipe it\n"
        features = semantic_plan_features(text, spec)
        self.assertEqual(features["invalid_tool_reference_count"], 1.0)
        self.assertEqual(features["valid_tool_reference_count"], 1.0)
        self.assertAlmostEqual(features["invalid_tool_reference_fraction"], 0.5)

    def test_no_tool_mentions_gives_zero_fractions_not_a_crash(self) -> None:
        spec = _spec()
        features = semantic_plan_features("1. think about it\n", spec)
        self.assertEqual(features["tool_reference_fraction"], 0.0)
        self.assertEqual(features["invalid_tool_reference_fraction"], 0.0)
        self.assertEqual(features["valid_tool_reference_count"], 0.0)
        self.assertEqual(features["invalid_tool_reference_count"], 0.0)


class RequirementCoverageTests(unittest.TestCase):
    def test_coverage_affects_representation(self) -> None:
        spec = _spec()
        full = "1. calendar.list_events: list events\n2. calendar.create_event: add an event\n"
        partial = "1. calendar.list_events: list events\n"
        full_features = semantic_plan_features(full, spec)
        partial_features = semantic_plan_features(partial, spec)
        self.assertEqual(full_features["requirement_coverage_fraction"], 1.0)
        self.assertEqual(partial_features["requirement_coverage_fraction"], 0.5)
        self.assertEqual(full_features["subgoal_covered_count"], 2.0)
        self.assertEqual(partial_features["subgoal_covered_count"], 1.0)

    def test_no_declared_subgoals_is_vacuously_covered(self) -> None:
        spec = _spec(subgoal_keywords=())
        features = semantic_plan_features("1. calendar.list_events: go\n", spec)
        self.assertEqual(features["requirement_coverage_fraction"], 1.0)
        self.assertEqual(features["subgoal_total_count"], 0.0)

    def test_no_declared_entities_is_vacuously_covered(self) -> None:
        spec = _spec(required_entities=())
        features = semantic_plan_features("1. calendar.list_events: go\n", spec)
        self.assertEqual(features["entity_coverage_fraction"], 1.0)
        self.assertEqual(features["entity_total_count"], 0.0)


class DependencyOrderingTests(unittest.TestCase):
    def test_wrong_order_is_a_violation(self) -> None:
        spec = _spec()
        correct = "1. calendar.list_events: list events\n2. calendar.create_event: add an event\n"
        reversed_order = (
            "1. calendar.create_event: add an event\n2. calendar.list_events: list events\n"
        )
        correct_features = semantic_plan_features(correct, spec)
        reversed_features = semantic_plan_features(reversed_order, spec)
        self.assertEqual(correct_features["dependency_violation_count"], 0.0)
        self.assertEqual(correct_features["dependency_consistency_fraction"], 1.0)
        self.assertEqual(reversed_features["dependency_violation_count"], 1.0)
        self.assertEqual(reversed_features["dependency_consistency_fraction"], 0.0)

    def test_no_applicable_pairs_is_vacuously_consistent(self) -> None:
        spec = _spec()
        features = semantic_plan_features("1. calendar.list_events: go\n", spec)
        self.assertEqual(features["dependency_applicable_count"], 0.0)
        self.assertEqual(features["dependency_consistency_fraction"], 1.0)

    def test_same_step_is_not_a_violation(self) -> None:
        spec = _spec()
        text = "1. calendar.list_events and calendar.create_event together\n"
        features = semantic_plan_features(text, spec)
        self.assertEqual(features["dependency_applicable_count"], 1.0)
        self.assertEqual(features["dependency_violation_count"], 0.0)


class ExtractPlanFeaturesTests(unittest.TestCase):
    def test_without_spec_returns_only_structural(self) -> None:
        features = extract_plan_features("1. calendar.list_events: go\n")
        self.assertEqual(set(features), STRUCTURAL_PLAN_FEATURES)

    def test_with_spec_returns_full_vector(self) -> None:
        features = extract_plan_features("1. calendar.list_events: go\n", _spec())
        self.assertEqual(set(features), STRUCTURAL_PLAN_FEATURES | SEMANTIC_PLAN_FEATURES)


class TaskPlanSpecValidationTests(unittest.TestCase):
    def test_requires_at_least_one_tool(self) -> None:
        with self.assertRaises(PlanSpecError):
            TaskPlanSpec(
                task_id="task-a",
                available_tools=(),
                subgoal_keywords=(),
                required_entities=(),
                dependency_pairs=(),
            )

    def test_dependency_pairs_must_reference_available_tools(self) -> None:
        with self.assertRaises(PlanSpecError):
            TaskPlanSpec(
                task_id="task-a",
                available_tools=("calendar.list_events",),
                subgoal_keywords=(),
                required_entities=(),
                dependency_pairs=(("calendar.list_events", "calendar.unknown_api"),),
            )

    def test_tool_name_must_match_app_dot_api(self) -> None:
        with self.assertRaises(PlanSpecError):
            TaskPlanSpec(
                task_id="task-a",
                available_tools=("not-a-dotted-name",),
                subgoal_keywords=(),
                required_entities=(),
                dependency_pairs=(),
            )


if __name__ == "__main__":
    unittest.main()
