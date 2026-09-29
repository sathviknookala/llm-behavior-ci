"""Deterministic pre-execution task metadata for plan-quality features.

A ``TaskPlanSpec`` names what a plan for one task is allowed to reference
before any tool executes: the app.api surface the task exposes, the
subgoals a correct plan should cover, the entities or constraints a
correct plan should name, and the pairs of tools whose relative order
matters. None of this is an AppWorld evaluator outcome, an AppWorld task
instruction, or AppWorld API documentation text; it is a synthetic,
committable description of what "before tool execution" evidence a plan
can be graded against. Loading mirrors ``tasks/catalog.py``: a local JSON
mapping, no appworld import, no environment lookup, no data root search.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

_MAX_TEXT = 256
_TOOL_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*\.[A-Za-z_][A-Za-z0-9_]*$")
_SPEC_KEYS = frozenset(
    {
        "task_id",
        "available_tools",
        "subgoal_keywords",
        "required_entities",
        "dependency_pairs",
    }
)


class PlanSpecError(ValueError):
    pass


class PlanSpecUnavailable(PlanSpecError):
    pass


def _line(value: object, name: str) -> str:
    if not isinstance(value, str) or value == "" or value != value.strip():
        raise PlanSpecError(f"{name} must be a non-empty string")
    if len(value) > _MAX_TEXT or any(
        ord(character) < 32 or ord(character) == 127 for character in value
    ):
        raise PlanSpecError(
            f"{name} must be a single line of at most {_MAX_TEXT} characters"
        )
    return value


def _exact_keys(payload: Mapping[object, object], name: str) -> None:
    if not all(isinstance(key, str) for key in payload):
        raise PlanSpecError(f"{name} has a non-string field name")
    unknown = sorted(set(payload) - _SPEC_KEYS)
    if unknown:
        raise PlanSpecError(f"{name} contains unknown fields: {', '.join(unknown)}")
    missing = sorted(_SPEC_KEYS - set(payload))
    if missing:
        raise PlanSpecError(f"{name} is missing fields: {', '.join(missing)}")


@dataclass(frozen=True)
class TaskPlanSpec:
    """Pre-execution requirement, tool, and dependency metadata for one task.

    ``available_tools`` is the closed set of ``app.api`` identifiers a plan
    may legitimately reference; it must be non-empty because every AppWorld
    task exposes at least one API. ``subgoal_keywords`` and
    ``required_entities`` may be empty: a task can legitimately have no
    declared subgoal or entity beyond the single top-level instruction, and
    an empty declaration is read as "nothing beyond the instruction to
    check" rather than as missing data. ``dependency_pairs`` names
    ``(before, after)`` tool pairs where ``before`` must be referenced no
    later than ``after`` in a consistent plan; both names must already be
    members of ``available_tools``.
    """

    task_id: str
    available_tools: tuple[str, ...]
    subgoal_keywords: tuple[tuple[str, ...], ...]
    required_entities: tuple[str, ...]
    dependency_pairs: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        _line(self.task_id, "task_id")
        if not isinstance(self.available_tools, tuple) or not self.available_tools:
            raise PlanSpecError("available_tools must be a non-empty tuple")
        seen_tools: set[str] = set()
        for tool in self.available_tools:
            if not isinstance(tool, str) or not _TOOL_PATTERN.match(tool):
                raise PlanSpecError(
                    f"available_tools entries must match app.api: {tool!r}"
                )
            if tool in seen_tools:
                raise PlanSpecError("available_tools contains a duplicate entry")
            seen_tools.add(tool)
        if not isinstance(self.subgoal_keywords, tuple):
            raise PlanSpecError("subgoal_keywords must be a tuple")
        for group in self.subgoal_keywords:
            if not isinstance(group, tuple) or not group:
                raise PlanSpecError(
                    "each subgoal_keywords entry needs at least one phrase"
                )
            for phrase in group:
                _line(phrase, "subgoal_keywords phrase")
        if not isinstance(self.required_entities, tuple):
            raise PlanSpecError("required_entities must be a tuple")
        seen_entities: set[str] = set()
        for entity in self.required_entities:
            _line(entity, "required_entities entry")
            if entity in seen_entities:
                raise PlanSpecError("required_entities contains a duplicate entry")
            seen_entities.add(entity)
        if not isinstance(self.dependency_pairs, tuple):
            raise PlanSpecError("dependency_pairs must be a tuple")
        for pair in self.dependency_pairs:
            if not isinstance(pair, tuple) or len(pair) != 2:
                raise PlanSpecError("dependency_pairs entries must be 2-tuples")
            before, after = pair
            if before not in seen_tools or after not in seen_tools:
                raise PlanSpecError(
                    "dependency_pairs entries must reference available_tools"
                )
            if before == after:
                raise PlanSpecError(
                    "dependency_pairs cannot relate a tool to itself"
                )


def task_plan_spec_from_mapping(payload: object) -> TaskPlanSpec:
    if not isinstance(payload, dict):
        raise PlanSpecError("task plan spec must be an object")
    _exact_keys(payload, "task plan spec")
    task_id = payload["task_id"]
    if not isinstance(task_id, str):
        raise PlanSpecError("task_id must be a string")
    tools = payload["available_tools"]
    if not isinstance(tools, list):
        raise PlanSpecError("available_tools must be a list")
    subgoals = payload["subgoal_keywords"]
    if not isinstance(subgoals, list):
        raise PlanSpecError("subgoal_keywords must be a list")
    subgoal_groups: list[tuple[str, ...]] = []
    for group in subgoals:
        if not isinstance(group, list):
            raise PlanSpecError("each subgoal_keywords entry must be a list")
        subgoal_groups.append(tuple(str(phrase) for phrase in group))
    entities = payload["required_entities"]
    if not isinstance(entities, list):
        raise PlanSpecError("required_entities must be a list")
    dependencies = payload["dependency_pairs"]
    if not isinstance(dependencies, list):
        raise PlanSpecError("dependency_pairs must be a list")
    pairs: list[tuple[str, str]] = []
    for pair in dependencies:
        if not isinstance(pair, list) or len(pair) != 2:
            raise PlanSpecError("each dependency_pairs entry must be a 2-item list")
        before, after = pair
        pairs.append((str(before), str(after)))
    return TaskPlanSpec(
        task_id=task_id,
        available_tools=tuple(str(tool) for tool in tools),
        subgoal_keywords=tuple(subgoal_groups),
        required_entities=tuple(str(entity) for entity in entities),
        dependency_pairs=tuple(pairs),
    )


def task_plan_specs_from_mapping(payload: object) -> tuple[TaskPlanSpec, ...]:
    if isinstance(payload, dict) and "entries" in payload:
        entries = payload["entries"]
    else:
        entries = payload
    if not isinstance(entries, list):
        raise PlanSpecError("task plan specs must be a list, or an object with entries")
    specs: list[TaskPlanSpec] = []
    seen: set[str] = set()
    for item in entries:
        spec = task_plan_spec_from_mapping(item)
        if spec.task_id in seen:
            raise PlanSpecError("task plan specs contains a duplicate task_id")
        seen.add(spec.task_id)
        specs.append(spec)
    return tuple(specs)


def load_task_plan_specs(path: Path) -> tuple[TaskPlanSpec, ...]:
    if not path.exists():
        raise PlanSpecUnavailable("task plan spec path does not exist")
    import json

    text = path.read_text(encoding="utf-8")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as error:
        raise PlanSpecError("task plan specs file is not valid JSON") from error
    return task_plan_specs_from_mapping(payload)


def index_by_task_id(
    specs: Sequence[TaskPlanSpec],
) -> Mapping[str, TaskPlanSpec]:
    indexed: dict[str, TaskPlanSpec] = {}
    for spec in specs:
        if spec.task_id in indexed:
            raise PlanSpecError("task plan specs contains a duplicate task_id")
        indexed[spec.task_id] = spec
    return indexed
