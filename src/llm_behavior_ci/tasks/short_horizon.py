"""Short-horizon Spotify train task-set selection.

``spotify_capability_short_v1`` keeps the train split, seed 17, and a
20-task target. Difficulty 1 is easy and difficulty 2 is medium, matching
AppWorld's ``metadata.difficulty``. Requirement count is the number of
evaluator tests, not their text.

Requirement count is not a horizon. A task with two tests can still paginate
a library in the reference solution. Every tier therefore also requires a
structural shape read from the reference call log before any model run:
at most one paged Spotify call, and at most ten Spotify calls after
authentication calls are excluded. The authenticated setup profile performs
authentication, so those calls are not part of the agent horizon.

Selection walks the tiers below and stops at the first pool that can fill
20 tasks with at most three tasks from one scenario. It uses the same
task-id sort and Fisher-Yates shuffle as ``deterministic_sample``. It does
not read model trajectories, evaluator outcomes, or requirement text. When
no tier can fill the set, it raises and writes nothing.

``spotify_short_horizon_diagnostic_v1`` is not that benchmark. It keeps
every train Spotify task whose reference solution has at most one paged
call and at most ten non-auth Spotify calls, in task-id order, with seed
17 recorded in the hash. It does not relax those limits to reach 20 tasks.
"""

from __future__ import annotations

import random
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from llm_behavior_ci.tasks.selection import (
    SelectionError,
    TaskSet,
    canonical_task_set_bytes,
    task_set_hash_from_bytes,
)

SELECTION_RULE = "spotify_capability_short_v1"
DIAGNOSTIC_RULE = "spotify_short_horizon_diagnostic_v1"
DIAGNOSTIC_TIER = "structural_short"
SELECTION_SEED = 17
TASK_COUNT = 20
DIAGNOSTIC_TASK_COUNT = 6
SPLIT = "train"
MAX_PAGED_CALLS = 1
MAX_REFERENCE_AGENT_CALLS = 10
MAX_TASKS_PER_SCENARIO = 3
DIFFICULTY_EASY = 1
DIFFICULTY_MEDIUM = 2

_DIFFICULTIES = frozenset({1, 2, 3})


class ShortHorizonSelectionError(SelectionError):
    def __init__(
        self,
        message: str,
        audit: Mapping[str, object] | None = None,
    ) -> None:
        super().__init__(message)
        self.audit = dict(audit or {})


@dataclass(frozen=True)
class ShortHorizonCandidate:
    task_id: str
    scenario_id: str
    difficulty: int
    requirement_count: int
    reference_paged_calls: int
    reference_agent_calls: int

    def __post_init__(self) -> None:
        _label(self.task_id, "task_id")
        _label(self.scenario_id, "scenario_id")
        _difficulty(self.difficulty)
        _count(self.requirement_count, "requirement_count")
        _count(self.reference_paged_calls, "reference_paged_calls")
        _count(self.reference_agent_calls, "reference_agent_calls")


@dataclass(frozen=True)
class _Tier:
    name: str
    difficulty: int
    max_requirements: int | None


TIERS: tuple[_Tier, ...] = (
    _Tier("tier_1", DIFFICULTY_EASY, 3),
    _Tier("tier_2", DIFFICULTY_EASY, 4),
    _Tier("tier_3", DIFFICULTY_EASY, None),
    _Tier("tier_4", DIFFICULTY_MEDIUM, 2),
)


@dataclass(frozen=True)
class ShortHorizonSelection:
    tier: str
    task_set: TaskSet
    candidates: tuple[ShortHorizonCandidate, ...]


def classify_reference_call(
    url: str,
    body_has_page_index: bool,
) -> tuple[bool, bool] | None:
    path, query = _split_url(url)
    if path is None:
        return None
    if path == "spotify/auth/token" or path.startswith("spotify/auth/token/"):
        return False, False
    paged = body_has_page_index or _query_has_page(query)
    return True, paged


def reference_shape(calls: Sequence[tuple[str, bool]]) -> tuple[int, int]:
    agent_calls = 0
    paged_calls = 0
    for url, body_has_page_index in calls:
        classified = classify_reference_call(url, body_has_page_index)
        if classified is None:
            continue
        include, paged = classified
        if not include:
            continue
        agent_calls += 1
        if paged:
            paged_calls += 1
    return agent_calls, paged_calls


def selection_audit(
    candidates: Sequence[ShortHorizonCandidate],
) -> dict[str, object]:
    _unique(candidates)
    return {
        "selection_rule": SELECTION_RULE,
        "selection_seed": SELECTION_SEED,
        "split": SPLIT,
        "task_count_requested": TASK_COUNT,
        "structural_limits": {
            "max_paged_calls": MAX_PAGED_CALLS,
            "max_reference_agent_calls": MAX_REFERENCE_AGENT_CALLS,
            "max_tasks_per_scenario": MAX_TASKS_PER_SCENARIO,
            "minimum_scenario_count": _minimum_scenarios(TASK_COUNT),
        },
        "intended_horizon": {
            "expected_productive_turns": "3-10",
            "execute_max_model_turns": 12,
            "execute_max_tokens": 192,
            "max_model_len": 32768,
        },
        "tiers": {
            tier.name: _summary(_pool(candidates, tier, structural=True))
            for tier in TIERS
        },
        "requirement_tiers_without_structural_filter": {
            tier.name: _summary(_pool(candidates, tier, structural=False))
            for tier in TIERS
        },
        "structural_short": _summary(
            tuple(candidate for candidate in candidates if _structural(candidate))
        ),
        "easy": _summary(
            tuple(
                candidate
                for candidate in candidates
                if candidate.difficulty == DIFFICULTY_EASY
            )
        ),
        "selected_tier": None,
    }


def select_spotify_short_set(
    candidates: Sequence[ShortHorizonCandidate],
    *,
    appworld_version: str,
    split: str = SPLIT,
    selection_rule: str = SELECTION_RULE,
    selection_seed: int = SELECTION_SEED,
    task_count: int = TASK_COUNT,
) -> ShortHorizonSelection:
    if split != SPLIT:
        raise ShortHorizonSelectionError("split must be train")
    if selection_rule != SELECTION_RULE:
        raise ShortHorizonSelectionError("selection_rule does not match the short-horizon rule")
    _label(appworld_version, "appworld_version")
    _count(selection_seed, "selection_seed")
    if isinstance(task_count, bool) or not isinstance(task_count, int) or task_count < 1:
        raise ShortHorizonSelectionError("task_count must be a positive integer")
    _unique(candidates)
    audit = selection_audit(candidates)
    chosen_tier: _Tier | None = None
    chosen_pool: tuple[ShortHorizonCandidate, ...] = ()
    for tier in TIERS:
        pool = _pool(candidates, tier, structural=True)
        if _can_fill(pool, task_count):
            chosen_tier = tier
            chosen_pool = pool
            break
    if chosen_tier is None:
        raise ShortHorizonSelectionError(
            "short-horizon pool cannot fill the requested task count",
            audit,
        )
    ordered = _seeded_order(chosen_pool, selection_seed)
    selected: list[ShortHorizonCandidate] = []
    per_scenario: Counter[str] = Counter()
    for candidate in ordered:
        if per_scenario[candidate.scenario_id] >= MAX_TASKS_PER_SCENARIO:
            continue
        selected.append(candidate)
        per_scenario[candidate.scenario_id] += 1
        if len(selected) == task_count:
            break
    if len(selected) != task_count or len(per_scenario) < _minimum_scenarios(task_count):
        raise ShortHorizonSelectionError(
            "short-horizon pool cannot fill the requested task count",
            audit,
        )
    tasks = tuple((candidate.task_id, candidate.scenario_id) for candidate in selected)
    payload = canonical_task_set_bytes(
        appworld_version=appworld_version,
        split=split,
        selection_rule=selection_rule,
        selection_seed=selection_seed,
        tasks=tasks,
    )
    task_set = TaskSet(
        appworld_version=appworld_version,
        split=split,
        selection_rule=selection_rule,
        selection_seed=selection_seed,
        task_count=len(tasks),
        scenario_count=len(per_scenario),
        task_ids=tuple(task_id for task_id, _scenario_id in tasks),
        scenario_ids=tuple(scenario_id for _task_id, scenario_id in tasks),
        task_set_hash=task_set_hash_from_bytes(payload),
    )
    return ShortHorizonSelection(
        tier=chosen_tier.name,
        task_set=task_set,
        candidates=tuple(selected),
    )


def select_spotify_short_diagnostic(
    candidates: Sequence[ShortHorizonCandidate],
    *,
    appworld_version: str,
    split: str = SPLIT,
    selection_rule: str = DIAGNOSTIC_RULE,
    selection_seed: int = SELECTION_SEED,
) -> ShortHorizonSelection:
    if split != SPLIT:
        raise ShortHorizonSelectionError("split must be train")
    if selection_rule != DIAGNOSTIC_RULE:
        raise ShortHorizonSelectionError("selection_rule does not match the diagnostic rule")
    if selection_seed != SELECTION_SEED:
        raise ShortHorizonSelectionError("diagnostic selection_seed must be 17")
    _label(appworld_version, "appworld_version")
    _unique(candidates)
    selected = tuple(
        sorted(
            (candidate for candidate in candidates if _structural(candidate)),
            key=lambda candidate: candidate.task_id,
        )
    )
    if len(selected) != DIAGNOSTIC_TASK_COUNT:
        audit = selection_audit(candidates)
        audit["diagnostic_task_count"] = len(selected)
        raise ShortHorizonSelectionError(
            "short-horizon diagnostic pool is not 6 tasks",
            audit,
        )
    tasks = tuple((candidate.task_id, candidate.scenario_id) for candidate in selected)
    payload = canonical_task_set_bytes(
        appworld_version=appworld_version,
        split=split,
        selection_rule=selection_rule,
        selection_seed=selection_seed,
        tasks=tasks,
    )
    scenario_ids = tuple(scenario_id for _task_id, scenario_id in tasks)
    task_set = TaskSet(
        appworld_version=appworld_version,
        split=split,
        selection_rule=selection_rule,
        selection_seed=selection_seed,
        task_count=len(tasks),
        scenario_count=len(set(scenario_ids)),
        task_ids=tuple(task_id for task_id, _scenario_id in tasks),
        scenario_ids=scenario_ids,
        task_set_hash=task_set_hash_from_bytes(payload),
    )
    return ShortHorizonSelection(
        tier=DIAGNOSTIC_TIER,
        task_set=task_set,
        candidates=selected,
    )


def _pool(
    candidates: Sequence[ShortHorizonCandidate],
    tier: _Tier,
    *,
    structural: bool,
) -> tuple[ShortHorizonCandidate, ...]:
    return tuple(
        candidate
        for candidate in candidates
        if _matches(candidate, tier, structural=structural)
    )


def _matches(
    candidate: ShortHorizonCandidate,
    tier: _Tier,
    *,
    structural: bool,
) -> bool:
    if candidate.difficulty != tier.difficulty:
        return False
    if (
        tier.max_requirements is not None
        and candidate.requirement_count > tier.max_requirements
    ):
        return False
    if structural and not _structural(candidate):
        return False
    return True


def _structural(candidate: ShortHorizonCandidate) -> bool:
    return (
        candidate.reference_paged_calls <= MAX_PAGED_CALLS
        and candidate.reference_agent_calls <= MAX_REFERENCE_AGENT_CALLS
    )


def _can_fill(pool: Sequence[ShortHorizonCandidate], task_count: int) -> bool:
    per_scenario: Counter[str] = Counter(candidate.scenario_id for candidate in pool)
    available = sum(
        min(MAX_TASKS_PER_SCENARIO, count) for count in per_scenario.values()
    )
    return (
        available >= task_count
        and len(per_scenario) >= _minimum_scenarios(task_count)
    )


def _minimum_scenarios(task_count: int) -> int:
    return (task_count + MAX_TASKS_PER_SCENARIO - 1) // MAX_TASKS_PER_SCENARIO


def _seeded_order(
    pool: Sequence[ShortHorizonCandidate],
    seed: int,
) -> list[ShortHorizonCandidate]:
    ordered = sorted(pool, key=lambda candidate: candidate.task_id)
    _shuffle(ordered, random.Random(seed))
    return ordered


def _shuffle(items: list[ShortHorizonCandidate], rng: random.Random) -> None:
    for index in range(len(items) - 1, 0, -1):
        swap = rng.randrange(index + 1)
        items[index], items[swap] = items[swap], items[index]


def _summary(pool: Sequence[ShortHorizonCandidate]) -> dict[str, object]:
    per_scenario: Counter[str] = Counter(candidate.scenario_id for candidate in pool)
    return {
        "task_count": len(pool),
        "scenario_count": len(per_scenario),
        "tasks_per_scenario": sorted(per_scenario.values(), reverse=True),
        "difficulty": _histogram(candidate.difficulty for candidate in pool),
        "requirement_counts": _histogram(
            candidate.requirement_count for candidate in pool
        ),
        "paged_calls": _histogram(
            candidate.reference_paged_calls for candidate in pool
        ),
        "reference_agent_calls": _histogram(
            candidate.reference_agent_calls for candidate in pool
        ),
    }


def _histogram(values: Sequence[int]) -> dict[str, int]:
    counts: Counter[int] = Counter(values)
    return {str(key): counts[key] for key in sorted(counts)}


def _unique(candidates: Sequence[ShortHorizonCandidate]) -> None:
    seen: set[str] = set()
    for candidate in candidates:
        if candidate.task_id in seen:
            raise ShortHorizonSelectionError("task set contains duplicate task ids")
        seen.add(candidate.task_id)


def _split_url(url: str) -> tuple[str | None, str]:
    if not isinstance(url, str) or url.strip() == "":
        return None, ""
    path, separator, query = url.partition("?")
    path = path.strip("/")
    if path != "spotify" and not path.startswith("spotify/"):
        return None, ""
    return path, query if separator else ""


def _query_has_page(query: str) -> bool:
    if query == "":
        return False
    keys = {part.split("=", 1)[0] for part in query.split("&") if part}
    return "page_index" in keys or "page" in keys


def _label(value: object, name: str) -> None:
    if not isinstance(value, str) or value.strip() == "" or value != value.strip():
        raise ShortHorizonSelectionError(f"{name} must be a non-empty string")


def _difficulty(value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value not in _DIFFICULTIES:
        raise ShortHorizonSelectionError("difficulty must be 1, 2, or 3")


def _count(value: object, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ShortHorizonSelectionError(f"{name} must be a non-negative integer")
