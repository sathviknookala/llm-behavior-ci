"""Versioned, deterministic plan-text feature extraction for the offline gate.

Every feature here is computable from information available strictly
before tool execution: the plan text the agent emitted in plan mode, and
``TaskPlanSpec`` metadata that describes a task's tool surface, subgoals,
required entities, and tool-ordering constraints. Nothing here reads an
evaluator outcome, a tool trace, or anything produced after the plan
episode terminates.

``PLAN_FEATURE_SCHEMA_VERSION`` names this feature vector; the protocol
lock and validation artifacts bind it, and a lock written under another
version is rejected. Changing a feature's definition, adding one, or
removing one is a new version. ``plan-features-v2`` reads a tool reference
as ``app.api`` or the action spelling ``apis.app.api``; v1 split
``apis.app.api`` into the invalid pair ``apis.app``.

Two pure functions produce the vector: ``structural_plan_features`` needs
only the plan text (character/line/token shape; kept as supplementary
evidence, per the offline-gate non-negotiable that structural signal alone
is not a sufficient plan representation). ``semantic_plan_features`` also
needs a ``TaskPlanSpec`` and is where task-grounded meaning lives:
requirement coverage, tool-reference validity, required-entity coverage,
and dependency-order consistency. Both are pure and depend only on their
arguments, so the same plan text and the same spec always produce the
same vector.
"""

from __future__ import annotations

import re
from typing import Mapping, Sequence

from llm_behavior_ci.tasks.plan_specs import TaskPlanSpec

PLAN_FEATURE_SCHEMA_VERSION = "plan-features-v2"

STRUCTURAL_PLAN_FEATURES = frozenset(
    {
        "char_count",
        "line_count",
        "numbered_step_count",
        "token_count",
        "empty_line_count",
        "mean_step_chars",
    }
)

SEMANTIC_PLAN_FEATURES = frozenset(
    {
        "requirement_coverage_fraction",
        "subgoal_covered_count",
        "subgoal_total_count",
        "entity_coverage_fraction",
        "entity_covered_count",
        "entity_total_count",
        "valid_tool_reference_count",
        "distinct_valid_tool_count",
        "invalid_tool_reference_count",
        "tool_reference_fraction",
        "invalid_tool_reference_fraction",
        "available_tool_count",
        "dependency_violation_count",
        "dependency_applicable_count",
        "dependency_consistency_fraction",
    }
)

PLAN_TEXT_FEATURES = STRUCTURAL_PLAN_FEATURES | SEMANTIC_PLAN_FEATURES

SUBGOAL_FEATURES = frozenset(
    {"requirement_coverage_fraction", "subgoal_covered_count", "subgoal_total_count"}
)
ENTITY_FEATURES = frozenset(
    {"entity_coverage_fraction", "entity_covered_count", "entity_total_count"}
)
DEPENDENCY_FEATURES = frozenset(
    {
        "dependency_violation_count",
        "dependency_applicable_count",
        "dependency_consistency_fraction",
    }
)

_NUMBERED_STEP = re.compile(r"^\s*\d+\.")
_TOOL_CHAIN = re.compile(
    r"(?<![A-Za-z0-9_.])[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)+(?![A-Za-z0-9_])"
)
_ACTION_NAMESPACE = "apis"


def tool_references(line: str) -> tuple[tuple[str, bool], ...]:
    """Every dotted reference in ``line`` as ``(normalized, well_formed)``.

    ``app.api`` and the action spelling ``apis.app.api`` both normalize to
    lowercase ``app.api`` and are well formed. Any other dotted chain,
    such as ``a.b.c`` or ``apis.app``, is one malformed reference kept
    verbatim, so it counts as invalid rather than being split into a
    valid-looking pair.
    """

    found: list[tuple[str, bool]] = []
    for match in _TOOL_CHAIN.finditer(line):
        parts = match.group(0).lower().split(".")
        if len(parts) == 2 and parts[0] != _ACTION_NAMESPACE:
            found.append((".".join(parts), True))
        elif len(parts) == 3 and parts[0] == _ACTION_NAMESPACE:
            found.append((".".join(parts[1:]), True))
        else:
            found.append((".".join(parts), False))
    return tuple(found)


class PlanFeatureError(ValueError):
    """Raised when a plan feature cannot be computed from its inputs."""


def _non_empty_lines(plan_text: str) -> list[str]:
    return [line for line in plan_text.splitlines() if line.strip()]


def _normalize(text: str) -> str:
    """Case- and whitespace-insensitive form used for phrase containment.

    Collapsing runs of whitespace and lowercasing means a step written as
    ``"Create   an  Event"`` still covers a declared subgoal phrased
    ``"create an event"``: reformatting a plan must not change coverage,
    entity, or tool-reference features on its own.
    """

    return " ".join(text.split()).lower()


def structural_plan_features(plan_text: str) -> dict[str, float]:
    """Raw shape of the plan text: length, line, and token counts.

    Supplementary to ``semantic_plan_features``: two plans with unrelated
    content can share every one of these values, which is why they are not
    used alone to judge whether candidate planning behavior changed.
    """

    if not isinstance(plan_text, str):
        raise PlanFeatureError("plan_text must be a string")
    lines = plan_text.splitlines()
    non_empty = _non_empty_lines(plan_text)
    numbered = [line for line in non_empty if _NUMBERED_STEP.match(line)]
    tokens = plan_text.split()
    mean_step_chars = (
        float(sum(len(line) for line in numbered) / len(numbered))
        if numbered
        else 0.0
    )
    return {
        "char_count": float(len(plan_text)),
        "line_count": float(len(non_empty)),
        "numbered_step_count": float(len(numbered)),
        "token_count": float(len(tokens)),
        "empty_line_count": float(len(lines) - len(non_empty)),
        "mean_step_chars": mean_step_chars,
    }


def semantic_plan_features(
    plan_text: str,
    task_spec: TaskPlanSpec,
) -> dict[str, float]:
    """Task-grounded plan features, given the task's pre-execution spec.

    Degradation rules, applied explicitly rather than left to silently
    produce a bogus number:

    - Zero declared subgoals or required entities means nothing beyond the
      instruction is being checked, so coverage is vacuously ``1.0``.
    - Zero applicable dependency pairs (neither tool named in the plan, or
      no pairs declared) means there is nothing to violate, so consistency
      is vacuously ``1.0``.
    - Zero tool-like references in the plan text means ``tool_reference_
      fraction`` and ``invalid_tool_reference_fraction`` are both ``0.0``:
      a plan naming no tool at all gets no credit and no penalty.

    A task with no ``TaskPlanSpec`` at all is not this function's problem
    to paper over: the caller (``lifecycle.offline_gate``) raises before
    reaching here, since a missing spec is missing data, not an empty one.
    """

    if not isinstance(plan_text, str):
        raise PlanFeatureError("plan_text must be a string")
    if not isinstance(task_spec, TaskPlanSpec):
        raise PlanFeatureError("task_spec must be a TaskPlanSpec")

    non_empty = _non_empty_lines(plan_text)
    available = {tool.lower() for tool in task_spec.available_tools}

    valid_mentions: list[tuple[int, str]] = []
    invalid_mentions: list[tuple[int, str]] = []
    for step_index, line in enumerate(non_empty):
        for token, well_formed in tool_references(line):
            if well_formed and token in available:
                valid_mentions.append((step_index, token))
            else:
                invalid_mentions.append((step_index, token))

    total_mentions = len(valid_mentions) + len(invalid_mentions)
    distinct_valid = {tool for _step_index, tool in valid_mentions}
    tool_reference_fraction = (
        len(valid_mentions) / total_mentions if total_mentions else 0.0
    )
    invalid_tool_reference_fraction = (
        len(invalid_mentions) / total_mentions if total_mentions else 0.0
    )

    normalized_text = _normalize(plan_text)
    subgoal_total = len(task_spec.subgoal_keywords)
    subgoal_covered = sum(
        1
        for group in task_spec.subgoal_keywords
        if any(_normalize(phrase) in normalized_text for phrase in group)
    )
    requirement_coverage_fraction = (
        subgoal_covered / subgoal_total if subgoal_total else 1.0
    )

    entity_total = len(task_spec.required_entities)
    entity_covered = sum(
        1
        for entity in task_spec.required_entities
        if _normalize(entity) in normalized_text
    )
    entity_coverage_fraction = (
        entity_covered / entity_total if entity_total else 1.0
    )

    first_index: dict[str, int] = {}
    for step_index, tool in valid_mentions:
        if tool not in first_index:
            first_index[tool] = step_index

    applicable = 0
    violations = 0
    for before, after in task_spec.dependency_pairs:
        before_key = before.lower()
        after_key = after.lower()
        if before_key in first_index and after_key in first_index:
            applicable += 1
            if first_index[after_key] < first_index[before_key]:
                violations += 1
    dependency_consistency_fraction = (
        1.0 - (violations / applicable) if applicable else 1.0
    )

    return {
        "requirement_coverage_fraction": requirement_coverage_fraction,
        "subgoal_covered_count": float(subgoal_covered),
        "subgoal_total_count": float(subgoal_total),
        "entity_coverage_fraction": entity_coverage_fraction,
        "entity_covered_count": float(entity_covered),
        "entity_total_count": float(entity_total),
        "valid_tool_reference_count": float(len(valid_mentions)),
        "distinct_valid_tool_count": float(len(distinct_valid)),
        "invalid_tool_reference_count": float(len(invalid_mentions)),
        "tool_reference_fraction": tool_reference_fraction,
        "invalid_tool_reference_fraction": invalid_tool_reference_fraction,
        "available_tool_count": float(len(task_spec.available_tools)),
        "dependency_violation_count": float(violations),
        "dependency_applicable_count": float(applicable),
        "dependency_consistency_fraction": dependency_consistency_fraction,
    }


def require_semantic_coverage(
    task_ids: Sequence[str],
    specs: Mapping[str, TaskPlanSpec],
    features: Sequence[str],
) -> None:
    """Refuse a gate whose semantic features would be vacuous or missing.

    Every task needs a spec when any semantic feature is requested.
    Subgoal features need declared subgoals, entity features need declared
    entities, and dependency features need declared pairs, on every task:
    the vacuous ``1.0`` the extractor returns for an empty declaration is
    not evidence that a plan covered anything. The error names counts, not
    task ids.
    """

    requested = set(features) & SEMANTIC_PLAN_FEATURES
    if not requested:
        return
    missing = [task_id for task_id in task_ids if task_id not in specs]
    if missing:
        raise PlanFeatureError(
            f"semantic plan features need a task_plan_spec for every gate task; "
            f"{len(missing)} of {len(task_ids)} are missing"
        )
    checks = (
        (SUBGOAL_FEATURES, "subgoal_keywords", lambda spec: spec.subgoal_keywords),
        (ENTITY_FEATURES, "required_entities", lambda spec: spec.required_entities),
        (DEPENDENCY_FEATURES, "dependency_pairs", lambda spec: spec.dependency_pairs),
    )
    for group, name, read in checks:
        if not requested & group:
            continue
        empty = sum(1 for task_id in task_ids if not read(specs[task_id]))
        if empty:
            raise PlanFeatureError(
                f"{', '.join(sorted(requested & group))} need non-empty {name}; "
                f"{empty} of {len(task_ids)} gate tasks declare none"
            )


def extract_plan_features(
    plan_text: str,
    task_spec: TaskPlanSpec | None = None,
) -> Mapping[str, float]:
    """Full versioned feature vector: structural, plus semantic when a spec is given."""

    features = dict(structural_plan_features(plan_text))
    if task_spec is not None:
        features.update(semantic_plan_features(plan_text, task_spec))
    return features
