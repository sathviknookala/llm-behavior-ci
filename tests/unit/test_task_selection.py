import hashlib
import tempfile
import unittest
from pathlib import Path

from llm_behavior_ci.config import TaskConfiguration
from llm_behavior_ci.tasks import (
    CatalogError,
    CatalogUnavailable,
    SelectionError,
    canonical_task_set_bytes,
    catalog_from_mapping,
    load_appworld_catalog,
    select_task_set,
    verify_task_set,
)


def _entry(
    task_id: str,
    scenario_id: str | None,
    split: str = "train",
    difficulty: int | None = 1,
) -> dict[str, object]:
    return {
        "task_id": task_id,
        "scenario_id": scenario_id,
        "split": split,
        "difficulty": difficulty,
    }


def _catalog(*entries: dict[str, object]):
    return catalog_from_mapping(
        {
            "appworld_version": "1.0.0",
            "entries": list(entries),
        }
    )


def _select(catalog, count: int = 2, seed: int = 7):
    return select_task_set(
        catalog,
        split="train",
        selection_rule="deterministic_sample",
        seed=seed,
        count=count,
    )


class TaskSelectionTests(unittest.TestCase):
    def test_identical_seed_reproduces_the_same_ids_and_hash(self) -> None:
        catalog = _catalog(
            _entry("task-a", "scenario-1"),
            _entry("task-b", "scenario-2"),
            _entry("task-c", "scenario-1"),
        )
        first = _select(catalog, count=2, seed=11)
        second = _select(catalog, count=2, seed=11)
        self.assertEqual(first.task_ids, second.task_ids)
        self.assertEqual(first.scenario_ids, second.scenario_ids)
        self.assertEqual(first.task_set_hash, second.task_set_hash)

    def test_task_count_matches_the_requested_count(self) -> None:
        catalog = _catalog(
            _entry("task-a", "scenario-1"),
            _entry("task-b", "scenario-2"),
            _entry("task-c", "scenario-1"),
        )
        selected = _select(catalog, count=2)
        self.assertEqual(selected.task_count, 2)
        self.assertEqual(len(selected.task_ids), 2)

    def test_hash_is_independent_of_input_order(self) -> None:
        pairs = (("task-a", "scenario-1"), ("task-b", "scenario-2"))
        reversed_pairs = (("task-b", "scenario-2"), ("task-a", "scenario-1"))
        first = canonical_task_set_bytes(
            appworld_version="1.0.0",
            split="train",
            selection_rule="deterministic_sample",
            selection_seed=3,
            tasks=pairs,
        )
        second = canonical_task_set_bytes(
            appworld_version="1.0.0",
            split="train",
            selection_rule="deterministic_sample",
            selection_seed=3,
            tasks=reversed_pairs,
        )
        self.assertEqual(first, second)
        other_seed = canonical_task_set_bytes(
            appworld_version="1.0.0",
            split="train",
            selection_rule="deterministic_sample",
            selection_seed=4,
            tasks=pairs,
        )
        self.assertNotEqual(
            hashlib.sha256(first).hexdigest(),
            hashlib.sha256(other_seed).hexdigest(),
        )

    def test_verify_accepts_a_configuration_built_from_the_set(self) -> None:
        catalog = _catalog(
            _entry("task-a", "scenario-1"),
            _entry("task-b", "scenario-2"),
        )
        selected = _select(catalog, count=2)
        configuration = TaskConfiguration(
            appworld_version=selected.appworld_version,
            split=selected.split,
            selection_rule=selected.selection_rule,
            selection_seed=selected.selection_seed,
            task_count=selected.task_count,
            task_set_hash=selected.task_set_hash,
        )
        self.assertIsNone(verify_task_set(configuration, selected))

    def test_verify_rejects_a_mismatched_hash(self) -> None:
        catalog = _catalog(
            _entry("task-a", "scenario-1"),
            _entry("task-b", "scenario-2"),
        )
        selected = _select(catalog, count=2)
        configuration = TaskConfiguration(
            appworld_version=selected.appworld_version,
            split=selected.split,
            selection_rule=selected.selection_rule,
            selection_seed=selected.selection_seed,
            task_count=selected.task_count,
            task_set_hash="a" * 64,
        )
        with self.assertRaises(SelectionError):
            verify_task_set(configuration, selected)

    def test_duplicate_catalog_ids_are_rejected(self) -> None:
        with self.assertRaises(CatalogError):
            _catalog(
                _entry("task-a", "scenario-1", split="train"),
                _entry("task-a", "scenario-2", split="dev"),
            )

    def test_duplicate_would_be_rejected_if_selection_repeated_an_id(self) -> None:
        _catalog(
            _entry("task-a", "scenario-1"),
            _entry("task-b", "scenario-2"),
        )
        with self.assertRaises(SelectionError):
            canonical_task_set_bytes(
                appworld_version="1.0.0",
                split="train",
                selection_rule="deterministic_sample",
                selection_seed=1,
                tasks=(("task-a", "scenario-1"), ("task-a", "scenario-2")),
            )

    def test_unsupported_selection_rule_is_rejected(self) -> None:
        catalog = _catalog(_entry("task-a", "scenario-1"))
        with self.assertRaises(SelectionError):
            select_task_set(
                catalog,
                split="train",
                selection_rule="fixed-v1",
                seed=1,
                count=1,
            )

    def test_count_above_the_split_size_is_rejected(self) -> None:
        catalog = _catalog(
            _entry("task-a", "scenario-1"),
            _entry("task-b", "scenario-2"),
        )
        with self.assertRaises(SelectionError):
            _select(catalog, count=3)

    def test_scenario_count_groups_null_scenarios_separately(self) -> None:
        nulls = _catalog(
            _entry("task-a", None),
            _entry("task-b", None),
        )
        shared = _catalog(
            _entry("task-a", "scenario-1"),
            _entry("task-b", "scenario-1"),
        )
        self.assertEqual(_select(nulls, count=2).scenario_count, 2)
        self.assertEqual(_select(shared, count=2).scenario_count, 1)

    def test_load_missing_catalog_raises_catalog_unavailable(self) -> None:
        path = Path(tempfile.mkdtemp()) / "missing.json"
        with self.assertRaises(CatalogUnavailable):
            load_appworld_catalog(path)

    def test_catalog_mapping_rejects_unknown_fields(self) -> None:
        with self.assertRaises(CatalogError):
            catalog_from_mapping(
                {
                    "appworld_version": "1.0.0",
                    "entries": [],
                    "extra": True,
                }
            )


if __name__ == "__main__":
    unittest.main()
