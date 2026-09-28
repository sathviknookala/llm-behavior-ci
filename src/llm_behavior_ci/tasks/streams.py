"""Seeded task-arrival streams over a selected task set.

Arrival times are ``index / arrival_rate_per_second``. Concurrency does
not change times, order, or membership. The schedule does not take an
outcome, completion time, or callback.
"""

from __future__ import annotations

import random
import re
from collections.abc import Callable, Iterator
from dataclasses import dataclass

from llm_behavior_ci.config import StreamSettings
from llm_behavior_ci.tasks.selection import TaskSet

_TWO_PHASE = re.compile(r"^two_phase:([1-9][0-9]*):([^:]+):([^:]+)$")


class StreamError(ValueError):
    pass


@dataclass(frozen=True)
class TaskArrival:
    index: int
    task_id: str
    scenario_id: str | None
    scheduled_offset_seconds: float
    stream_seed: int


def _pairs(task_set: TaskSet) -> list[tuple[str, str | None]]:
    return list(zip(task_set.task_ids, task_set.scenario_ids))


def _sorted_pool(
    items: list[tuple[str, str | None]],
) -> list[tuple[str, str | None]]:
    return sorted(items, key=lambda item: item[0])


def _mix_planner(
    task_set: TaskSet,
    rule: str,
) -> Callable[[int], list[tuple[str, str | None]]]:
    pairs = _pairs(task_set)
    if rule == "uniform":
        pool = _sorted_pool(pairs)
        return lambda _index: list(pool)
    if rule == "scenario_round_robin":
        groups: dict[str, list[tuple[str, str | None]]] = {}
        for task_id, scenario_id in pairs:
            key = scenario_id if scenario_id is not None else f"task:{task_id}"
            groups.setdefault(key, []).append((task_id, scenario_id))
        keys = sorted(groups)
        for key in keys:
            groups[key] = _sorted_pool(groups[key])
        if not keys:
            return lambda _index: []
        return lambda index: list(groups[keys[index % len(keys)]])
    match = _TWO_PHASE.fullmatch(rule)
    if match is not None:
        cut = int(match.group(1))
        before = match.group(2)
        after = match.group(3)
        present = {
            scenario_id for _task_id, scenario_id in pairs if scenario_id is not None
        }
        if before not in present or after not in present:
            raise StreamError("two-phase scenario is not in the task set")
        before_pool = _sorted_pool(
            [item for item in pairs if item[1] == before]
        )
        after_pool = _sorted_pool([item for item in pairs if item[1] == after])
        return lambda index: list(before_pool if index < cut else after_pool)
    raise StreamError("unsupported task mix rule")


def _arrivals(
    settings: StreamSettings,
    planner: Callable[[int], list[tuple[str, str | None]]],
) -> Iterator[TaskArrival]:
    rng = random.Random(settings.stream_seed)
    yielded: set[str] = set()
    index = 0
    while True:
        pool = planner(index)
        if not settings.with_replacement:
            pool = [item for item in pool if item[0] not in yielded]
        if not pool:
            return
        task_id, scenario_id = pool[rng.randrange(len(pool))]
        if not settings.with_replacement:
            yielded.add(task_id)
        yield TaskArrival(
            index=index,
            task_id=task_id,
            scenario_id=scenario_id,
            scheduled_offset_seconds=index / settings.arrival_rate_per_second,
            stream_seed=settings.stream_seed,
        )
        index += 1


def generate_stream(task_set: TaskSet, settings: StreamSettings) -> Iterator[TaskArrival]:
    if (
        settings.split != task_set.split
        or settings.selection_rule != task_set.selection_rule
        or settings.selection_seed != task_set.selection_seed
        or settings.task_set_hash != task_set.task_set_hash
    ):
        raise StreamError("stream settings do not match the task set")
    planner = _mix_planner(task_set, settings.task_mix_rule)
    return _arrivals(settings, planner)
