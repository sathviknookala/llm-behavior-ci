import inspect
import unittest
from itertools import islice

from llm_behavior_ci.config import StreamSettings
from llm_behavior_ci.tasks import (
    StreamError,
    catalog_from_mapping,
    generate_stream,
    select_task_set,
)


def _catalog(*entries: dict[str, object]):
    return catalog_from_mapping(
        {
            "appworld_version": "1.0.0",
            "entries": list(entries),
        }
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


def _task_set():
    catalog = _catalog(
        _entry("task-a", "scenario-1"),
        _entry("task-b", "scenario-2"),
        _entry("task-c", "scenario-1"),
    )
    return select_task_set(
        catalog,
        split="train",
        selection_rule="deterministic_sample",
        seed=7,
        count=3,
    )


def _settings(
    task_set,
    *,
    task_mix_rule: str = "uniform",
    with_replacement: bool = False,
    stream_seed: int = 13,
    arrival_rate_per_second: float = 0.25,
    split: str | None = None,
    selection_rule: str | None = None,
    selection_seed: int | None = None,
    task_set_hash: str | None = None,
    concurrency: int = 4,
) -> StreamSettings:
    return StreamSettings(
        split=task_set.split if split is None else split,
        selection_rule=(
            task_set.selection_rule if selection_rule is None else selection_rule
        ),
        selection_seed=(
            task_set.selection_seed if selection_seed is None else selection_seed
        ),
        task_set_hash=(
            task_set.task_set_hash if task_set_hash is None else task_set_hash
        ),
        stream_seed=stream_seed,
        arrival_rate_per_second=arrival_rate_per_second,
        concurrency=concurrency,
        with_replacement=with_replacement,
        task_mix_rule=task_mix_rule,
    )


class TaskStreamTests(unittest.TestCase):
    def test_identical_stream_seed_reproduces_order_and_times(self) -> None:
        task_set = _task_set()
        settings = _settings(task_set, with_replacement=True)
        first = list(islice(generate_stream(task_set, settings), 8))
        second = list(islice(generate_stream(task_set, settings), 8))
        self.assertEqual(first, second)

    def test_repeated_tasks_occur_when_sampling_with_replacement(self) -> None:
        catalog = _catalog(
            _entry("task-a", "scenario-1"),
            _entry("task-b", "scenario-2"),
        )
        task_set = select_task_set(
            catalog,
            split="train",
            selection_rule="deterministic_sample",
            seed=7,
            count=2,
        )
        settings = _settings(task_set, with_replacement=True)
        arrivals = list(islice(generate_stream(task_set, settings), 30))
        counts: dict[str, int] = {}
        for arrival in arrivals:
            counts[arrival.task_id] = counts.get(arrival.task_id, 0) + 1
        self.assertTrue(any(count > 1 for count in counts.values()))

    def test_without_replacement_stops_at_the_set_size(self) -> None:
        task_set = _task_set()
        settings = _settings(task_set, with_replacement=False)
        arrivals = list(generate_stream(task_set, settings))
        self.assertEqual(len(arrivals), task_set.task_count)
        self.assertEqual(len({arrival.task_id for arrival in arrivals}), task_set.task_count)

    def test_scenario_round_robin_alternates_groups(self) -> None:
        catalog = _catalog(
            _entry("task-a", "scenario-1"),
            _entry("task-b", "scenario-2"),
        )
        task_set = select_task_set(
            catalog,
            split="train",
            selection_rule="deterministic_sample",
            seed=7,
            count=2,
        )
        settings = _settings(
            task_set,
            task_mix_rule="scenario_round_robin",
            with_replacement=True,
        )
        arrivals = list(islice(generate_stream(task_set, settings), 6))
        self.assertEqual(
            [arrival.scenario_id for arrival in arrivals],
            ["scenario-1", "scenario-2"] * 3,
        )

    def test_two_phase_changes_the_pool_at_the_cut(self) -> None:
        task_set = _task_set()
        settings = _settings(
            task_set,
            task_mix_rule="two_phase:3:scenario-1:scenario-2",
            with_replacement=True,
        )
        arrivals = list(islice(generate_stream(task_set, settings), 6))
        self.assertEqual(
            [arrival.scenario_id for arrival in arrivals[:3]],
            ["scenario-1", "scenario-1", "scenario-1"],
        )
        self.assertEqual(
            [arrival.scenario_id for arrival in arrivals[3:]],
            ["scenario-2", "scenario-2", "scenario-2"],
        )

    def test_schedule_ignores_a_caller_supplied_outcome(self) -> None:
        task_set = _task_set()
        settings = _settings(task_set, with_replacement=True)
        first = list(islice(generate_stream(task_set, settings), 5))
        _success = True
        del _success
        second = list(islice(generate_stream(task_set, settings), 5))
        self.assertEqual(first, second)
        self.assertNotIn("outcome", inspect.signature(generate_stream).parameters)

    def test_arrival_time_is_index_divided_by_rate(self) -> None:
        task_set = _task_set()
        rate = 0.5
        settings = _settings(
            task_set,
            with_replacement=True,
            arrival_rate_per_second=rate,
        )
        for arrival in islice(generate_stream(task_set, settings), 5):
            self.assertEqual(arrival.scheduled_offset_seconds, arrival.index / rate)

    def test_settings_that_disagree_with_the_task_set_are_rejected(self) -> None:
        task_set = _task_set()
        settings = _settings(task_set, split="dev")
        with self.assertRaises(StreamError):
            generate_stream(task_set, settings)

    def test_unsupported_mix_rule_is_rejected(self) -> None:
        task_set = _task_set()
        settings = _settings(task_set, task_mix_rule="difficulty-only")
        with self.assertRaises(StreamError):
            generate_stream(task_set, settings)


if __name__ == "__main__":
    unittest.main()
