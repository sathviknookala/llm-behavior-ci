from __future__ import annotations

import unittest

from llm_behavior_ci.lifecycle.plan_features import (
    PLAN_FEATURE_SCHEMA_VERSION,
    PLAN_TEXT_FEATURES,
    PlanFeatureError,
    SEMANTIC_PLAN_FEATURES,
    STRUCTURAL_PLAN_FEATURES,
    extract_plan_features,
    plan_tool_references,
    require_semantic_coverage,
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
        self.assertEqual(PLAN_FEATURE_SCHEMA_VERSION, "plan-features-v4")
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


class ProseDottedTokenTests(unittest.TestCase):
    _TOOLS = (
        "file_system.create_file",
        "spotify.show_song_library",
        "supervisor.complete_task",
    )

    def test_file_names_and_domains_are_prose_not_invalid_references(self) -> None:
        for line in (
            '3. File System: create_file "~/backups/spotify_library.csv"',
            "4. save it as Amsterdam.zip and archive.tar.gz",
            "5. see example.com for details",
        ):
            references = plan_tool_references(line, self._TOOLS)
            self.assertFalse([item for item in references if item[0] != "file_system.create_file"], line)

    def test_unknown_apps_in_action_or_call_form_stay_invalid(self) -> None:
        self.assertEqual(
            plan_tool_references("1. apis.weather.forecast()", self._TOOLS),
            (("weather.forecast", True),),
        )
        self.assertEqual(
            plan_tool_references("1. call weather.forecast (city)", self._TOOLS),
            (("weather.forecast", True),),
        )
        spec = TaskPlanSpec(
            task_id="task-files",
            available_tools=self._TOOLS,
            subgoal_keywords=(),
            required_entities=(),
            dependency_pairs=(("file_system.create_file", "supervisor.complete_task"),),
        )
        features = semantic_plan_features(
            "1. File System: create_file spotify_library.csv\n2. weather.forecast()\n", spec
        )
        self.assertEqual(features["invalid_tool_reference_count"], 1.0)
        self.assertEqual(features["valid_tool_reference_count"], 1.0)

    def test_known_app_with_unknown_api_stays_invalid(self) -> None:
        self.assertEqual(
            plan_tool_references("2. spotify.delete_everything", self._TOOLS),
            (("spotify.delete_everything", True),),
        )


class LabelledStepReferenceTests(unittest.TestCase):
    _TOOLS = (
        "simple_note.login",
        "simple_note.search_notes",
        "supervisor.show_account_passwords",
    )

    def test_plan_v1_app_then_api_steps_are_read(self) -> None:
        self.assertEqual(
            plan_tool_references("2. Simple Note: search_notes for movies", self._TOOLS),
            (("simple_note.search_notes", True),),
        )
        self.assertEqual(
            plan_tool_references("1. **Supervisor**: show_account_passwords", self._TOOLS),
            (("supervisor.show_account_passwords", True),),
        )

    def test_dotted_reference_in_a_labelled_step_is_counted_once(self) -> None:
        self.assertEqual(
            plan_tool_references(
                "1. Supervisor: supervisor.show_account_passwords", self._TOOLS
            ),
            (("supervisor.show_account_passwords", True),),
        )

    def test_unresolved_labels_are_prose_not_invalid_references(self) -> None:
        for line in (
            "Plan:",
            "Note: the user wants movies",
            "1. Simple Note: delete_everything",
            "3. Weather: forecast",
        ):
            self.assertEqual(plan_tool_references(line, self._TOOLS), (), line)

    def test_labelled_and_dotted_plans_score_alike(self) -> None:
        spec = _spec()
        dotted = "1. calendar.list_events: list events\n2. calendar.create_event: create an event\n"
        labelled = "1. Calendar: list_events to list events\n2. Calendar: create_event to create an event\n"
        dotted_features = semantic_plan_features(dotted, spec)
        labelled_features = semantic_plan_features(labelled, spec)
        for name in (
            "valid_tool_reference_count",
            "tool_reference_fraction",
            "required_tool_coverage_fraction",
            "dependency_consistency_fraction",
        ):
            self.assertEqual(dotted_features[name], labelled_features[name], name)


class RequiredToolCoverageTests(unittest.TestCase):
    def test_plan_naming_no_tool_has_no_tool_coverage(self) -> None:
        spec = _spec()
        features = semantic_plan_features(
            "1. list the events in the calendar\n2. create an event\n", spec
        )
        self.assertEqual(features["invalid_tool_reference_fraction"], 0.0)
        self.assertEqual(features["dependency_consistency_fraction"], 1.0)
        self.assertEqual(features["required_tool_coverage_fraction"], 0.0)
        self.assertEqual(features["required_tool_total_count"], 2.0)

    def test_coverage_counts_only_valid_required_tools(self) -> None:
        spec = _spec()
        features = semantic_plan_features(
            "1. calendar.list_events: list events\n2. calendar.delete_all: wipe\n", spec
        )
        self.assertEqual(features["required_tool_coverage_fraction"], 0.5)
        self.assertEqual(features["required_tool_covered_count"], 1.0)

    def test_gate_refuses_coverage_without_declared_pairs(self) -> None:
        spec = _spec(dependency_pairs=())
        with self.assertRaises(PlanFeatureError):
            require_semantic_coverage(
                ("task-a",), {"task-a": spec}, ("required_tool_coverage_fraction",)
            )


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
