"""Deterministic task-set selection and hash verification.

``select_task_set`` returns a ``TaskSet``. The caller builds a
``TaskConfiguration`` from that set. Selection does not construct a
``RunConfiguration`` and does not include instructions, API docs, or
outcomes.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from collections.abc import Sequence

from llm_behavior_ci.config import SPLITS, TaskConfiguration
from llm_behavior_ci.tasks.catalog import TaskCatalog

_SELECTION_RULE = "deterministic_sample"


class SelectionError(ValueError):
    pass


@dataclass(frozen=True)
class TaskSet:
    appworld_version: str
    split: str
    selection_rule: str
    selection_seed: int
    task_count: int
    scenario_count: int
    task_ids: tuple[str, ...]
    scenario_ids: tuple[str | None, ...]
    task_set_hash: str

    def __post_init__(self) -> None:
        if self.split not in SPLITS:
            choices = ", ".join(sorted(SPLITS))
            raise SelectionError(f"split must be one of: {choices}")
        if len(self.task_ids) != len(self.scenario_ids):
            raise SelectionError("task_ids and scenario_ids must be aligned")
        if self.task_count != len(self.task_ids):
            raise SelectionError("task_count must equal the selected length")
        if self.scenario_count != _scenario_count(self.scenario_ids):
            raise SelectionError("scenario_count does not match the selected groups")


def _scenario_count(scenario_ids: Sequence[str | None]) -> int:
    named: set[str] = set()
    lone = 0
    for scenario_id in scenario_ids:
        if scenario_id is None:
            lone += 1
        else:
            named.add(scenario_id)
    return len(named) + lone


def _shuffle(items: list[object], rng: random.Random) -> None:
    for index in range(len(items) - 1, 0, -1):
        swap = rng.randrange(index + 1)
        items[index], items[swap] = items[swap], items[index]


def canonical_task_set_bytes(
    *,
    appworld_version: str,
    split: str,
    selection_rule: str,
    selection_seed: int,
    tasks: Sequence[tuple[str, str | None]],
) -> bytes:
    seen: set[str] = set()
    for task_id, _scenario_id in tasks:
        if task_id in seen:
            raise SelectionError("task set contains duplicate task ids")
        seen.add(task_id)
    task_count = len(tasks)
    lines = [
        "task-set-v1",
        appworld_version,
        split,
        selection_rule,
        str(selection_seed),
        str(task_count),
    ]
    for task_id, scenario_id in sorted(tasks, key=lambda item: item[0]):
        lines.append(f"{task_id}\t{'' if scenario_id is None else scenario_id}")
    return ("\n".join(lines) + "\n").encode("utf-8")


def task_set_hash_from_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def select_task_set(
    catalog: TaskCatalog,
    *,
    split: str,
    selection_rule: str,
    seed: int,
    count: int,
) -> TaskSet:
    if selection_rule != _SELECTION_RULE:
        raise SelectionError("selection_rule must be deterministic_sample")
    if split not in SPLITS:
        choices = ", ".join(sorted(SPLITS))
        raise SelectionError(f"split must be one of: {choices}")
    if isinstance(count, bool) or not isinstance(count, int):
        raise SelectionError("count must be an integer")
    if count < 1:
        raise SelectionError("count must be a positive integer")
    eligible = sorted(
        (entry for entry in catalog.entries if entry.split == split),
        key=lambda entry: entry.task_id,
    )
    if count > len(eligible):
        raise SelectionError("count exceeds the split size")
    items = list(eligible)
    _shuffle(items, random.Random(seed))
    chosen = items[:count]
    tasks = tuple((entry.task_id, entry.scenario_id) for entry in chosen)
    payload = canonical_task_set_bytes(
        appworld_version=catalog.appworld_version,
        split=split,
        selection_rule=selection_rule,
        selection_seed=seed,
        tasks=tasks,
    )
    task_ids = tuple(task_id for task_id, _scenario_id in tasks)
    scenario_ids = tuple(scenario_id for _task_id, scenario_id in tasks)
    return TaskSet(
        appworld_version=catalog.appworld_version,
        split=split,
        selection_rule=selection_rule,
        selection_seed=seed,
        task_count=len(tasks),
        scenario_count=_scenario_count(scenario_ids),
        task_ids=task_ids,
        scenario_ids=scenario_ids,
        task_set_hash=task_set_hash_from_bytes(payload),
    )


def verify_task_set(configuration: TaskConfiguration, task_set: TaskSet) -> None:
    if configuration.appworld_version != task_set.appworld_version:
        raise SelectionError("appworld_version does not match the task set")
    if configuration.split != task_set.split:
        raise SelectionError("split does not match the task set")
    if configuration.selection_rule != task_set.selection_rule:
        raise SelectionError("selection_rule does not match the task set")
    if configuration.selection_seed != task_set.selection_seed:
        raise SelectionError("selection_seed does not match the task set")
    if configuration.task_count != task_set.task_count:
        raise SelectionError("task_count does not match the task set")
    if configuration.task_set_hash != task_set.task_set_hash:
        raise SelectionError("task_set_hash does not match the task set")
    payload = canonical_task_set_bytes(
        appworld_version=task_set.appworld_version,
        split=task_set.split,
        selection_rule=task_set.selection_rule,
        selection_seed=task_set.selection_seed,
        tasks=tuple(zip(task_set.task_ids, task_set.scenario_ids)),
    )
    if task_set_hash_from_bytes(payload) != task_set.task_set_hash:
        raise SelectionError("task_set_hash does not match the canonical digest")
    return None
