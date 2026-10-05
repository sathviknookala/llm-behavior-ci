"""Run the Spotify capability pilot: twenty execute episodes, one at a time.

Requires --configuration, --task-set, --base-url, --store, and --output.
The configuration must be the committed Spotify capability profile:
execute horizon 20, execute token cap 192, temperature 0, seed 17, and
``tool_access_profile`` ``spotify_capability_v1``. ``git_commit`` must be
HEAD and the tracked source and config tree must be clean.

Prints one public JSON object after each episode. Writes one public
aggregate to --output. An episode-local runtime failure, including a
context-length rejection, is recorded for that task and the remaining
tasks still run. The record keeps the steps and controller counts already
collected. Execute mode attempts the native evaluator on a world that is
already open; if that attempt is missing or fails, the evaluator outcome
stays unavailable and is not treated as a success or a failure. An
end-to-end success is an episode that finishes without a runtime failure
and whose evaluator reports success. The denominator is the full task
count. A pass over every task writes ``run_status`` ``completed`` even
when some tasks are runtime failures. A task-set, provenance, manifest,
configuration, store, or runtime-initialization failure still aborts,
writes ``run_status`` ``aborted``, and leaves every unfinished task
``not_attempted``. Task ids, instructions, model text, API arguments,
trajectories, and evaluator internals stay in the local store. When the
local manifest omits ``appworld_setup_profile``, the matching committed
task metadata supplies it.

The aggregate keeps the existing capability metrics and adds controller
counts: workflow-envelope rejections, repeat-gate blocks, completion-gate
blocks, stall events, model generations, executed tool actions, productive
action-turn-cap hits, and workflow-format retries. A workflow JSON object
with a parseable action is not a parser error when a controller policy
blocks it. ``episodes_hitting_execute_turn_cap`` still counts stored model
steps. ``productive_action_turn_cap_hits`` counts episodes whose generations
that spent an execute turn reached ``execute_max_model_turns``. For
``plan_progress_v2``, the first two consecutive malformed envelopes in an
attempt do not spend that budget.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from collections.abc import Callable, Sequence
from dataclasses import replace
from datetime import datetime
from pathlib import Path

from llm_behavior_ci.config import (
    AgentConfiguration,
    ConfigError,
    EpisodeIdentity,
    ModelConfiguration,
    RunConfiguration,
    RunIdentity,
    TaskConfiguration,
    new_episode_identity,
    new_run_identity,
    run_configuration_hash,
)
from llm_behavior_ci.records import (
    EXECUTE_TERMINATIONS,
    EpisodeResult,
    LocalTaskRef,
    ModelStep,
    RecordedError,
    ToolStep,
)
from llm_behavior_ci.runtime.actions import ActionRejected, parse_model_output
from llm_behavior_ci.runtime.workflow import (
    WorkflowTraceAccounting,
    account_workflow_trace,
    workflow_output_has_parseable_action,
)
from llm_behavior_ci.runtime.clock import wall_now
from llm_behavior_ci.runtime.episode import (
    EpisodeRejected,
    build_runtime,
    run_episode,
    runtime_failure_message,
)
from llm_behavior_ci.runtime.provenance import ProvenanceError, enforce_committed_provenance
from llm_behavior_ci.storage import EpisodeStore, StorageError
from llm_behavior_ci.tasks.selection import SelectionError, TaskSet, verify_task_set
from llm_behavior_ci.tasks.short_horizon import DIAGNOSTIC_RULE, DIAGNOSTIC_TASK_COUNT

_PUBLIC_TASK_FIELDS = (
    "appworld_version",
    "split",
    "selection_rule",
    "selection_seed",
    "task_count",
    "task_set_hash",
)
_OPTIONAL_PUBLIC_TASK_FIELDS = ("appworld_setup_profile",)
_REPO_ROOT = Path(__file__).resolve().parents[2]
_PILOT_PROFILE = "spotify_capability_v1"
_PILOT_TASK_COUNT = 20
_PILOT_TURN_CAP = 20
_PILOT_TOKEN_CAP = 192
_PILOT_SEED = 17
_DIFFICULTY_EASY = 1
_DIFFICULTY_MEDIUM = 2
_TERMINAL_STATUSES = (
    "evaluator_success",
    "evaluator_failure",
    "runtime_failure",
    "not_attempted",
)
_EXPERIMENT_FAILURES = (
    ProvenanceError,
    SelectionError,
    ConfigError,
    EpisodeRejected,
    StorageError,
)


def _refuse_results_store(path: Path) -> None:
    resolved = path.resolve()
    for parent in (resolved, *resolved.parents):
        if parent.name == "results":
            print("store path must not be under results/", file=sys.stderr)
            raise SystemExit(2)


def _load_json(path: Path) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise SystemExit(2) from error


def _git_commit() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        print("git rev-parse HEAD failed", file=sys.stderr)
        raise SystemExit(2)
    revision = completed.stdout.strip()
    if len(revision) != 40:
        print("git rev-parse HEAD failed", file=sys.stderr)
        raise SystemExit(2)
    return revision


def _public_task_fields(payload: dict[str, object]) -> dict[str, object]:
    fields = {field: payload[field] for field in _PUBLIC_TASK_FIELDS}
    for field in _OPTIONAL_PUBLIC_TASK_FIELDS:
        if field in payload and payload[field] is not None:
            fields[field] = payload[field]
    return fields


def _task_identity(task: TaskConfiguration) -> dict[str, object]:
    document = task.to_dict()
    document.pop("appworld_setup_profile", None)
    return document


def _matching_committed_task(task: TaskConfiguration) -> TaskConfiguration | None:
    identity = _task_identity(task)
    tasks_dir = _REPO_ROOT / "configs" / "tasks"
    if not tasks_dir.is_dir():
        return None
    for path in sorted(tasks_dir.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict):
            continue
        try:
            committed = TaskConfiguration.from_dict(_public_task_fields(payload))
        except (KeyError, TypeError, ConfigError):
            continue
        if _task_identity(committed) == identity:
            return committed
    return None


def _adopt_committed_setup_profile(task: TaskConfiguration) -> TaskConfiguration | None:
    committed = _matching_committed_task(task)
    if committed is None:
        return None
    if task.appworld_setup_profile is None:
        return replace(task, appworld_setup_profile=committed.appworld_setup_profile)
    if task.appworld_setup_profile != committed.appworld_setup_profile:
        return None
    return task


def _load_task_set(payload: object) -> tuple[TaskConfiguration, TaskSet]:
    if not isinstance(payload, dict):
        raise SelectionError("task set must be an object")
    task = TaskConfiguration.from_dict(_public_task_fields(payload))
    task_ids = tuple(payload["task_ids"])
    scenario_ids = tuple(payload["scenario_ids"])
    task_set = TaskSet(
        appworld_version=str(payload["appworld_version"]),
        split=str(payload["split"]),
        selection_rule=str(payload["selection_rule"]),
        selection_seed=int(payload["selection_seed"]),
        task_count=int(payload["task_count"]),
        scenario_count=int(payload["scenario_count"]),
        task_ids=task_ids,
        scenario_ids=scenario_ids,
        task_set_hash=str(payload["task_set_hash"]),
    )
    verify_task_set(task, task_set)
    return task, task_set


def _difficulty_by_task(payload: object) -> dict[str, int]:
    if not isinstance(payload, dict):
        return {}
    raw = payload.get("difficulty_by_task")
    if not isinstance(raw, dict):
        return {}
    parsed: dict[str, int] = {}
    for key, value in raw.items():
        if not isinstance(key, str) or isinstance(value, bool) or not isinstance(value, int):
            continue
        parsed[key] = value
    return parsed


def _persistence_callbacks(
    store: EpisodeStore,
    task_id: str,
    holder: dict[str, str] | None = None,
) -> tuple[
    Callable[[EpisodeIdentity, RunIdentity], None],
    Callable[[ModelStep | ToolStep], None],
]:
    current: dict[str, str] = {} if holder is None else holder

    def on_start(identity: EpisodeIdentity, run: RunIdentity) -> None:
        store.start_episode(identity, run, task_id)
        current["episode_id"] = identity.episode_id

    def on_step(step: ModelStep | ToolStep) -> None:
        episode_id = current.get("episode_id")
        if episode_id is None:
            raise StorageError("append_step failed")
        store.append_step(episode_id, step)

    return on_start, on_step


def _public_summary(
    *,
    configuration: RunConfiguration,
    task_index: int,
    episode: EpisodeResult,
) -> dict[str, object]:
    outcome = episode.evaluator_outcome
    error_type, error_message = _public_runtime_failure(episode)
    wall_seconds = (episode.ended_at - episode.started_at).total_seconds()
    model_latency_seconds = sum(step.latency_seconds for step in episode.model_steps)
    successful_tool_calls = sum(1 for step in episode.tool_steps if step.error is None)
    error_tool_calls = sum(1 for step in episode.tool_steps if step.error is not None)
    return {
        "configuration_hash": run_configuration_hash(configuration),
        "task_set_hash": configuration.task.task_set_hash,
        "task_index": task_index,
        "terminal_status": _terminal_status(episode),
        "evaluator_status": _evaluator_status(episode),
        "termination_reason": episode.termination_reason,
        "runtime_error_type": error_type,
        "runtime_error_message": error_message,
        "status": episode.status,
        "model_step_count": len(episode.model_steps),
        "tool_step_count": len(episode.tool_steps),
        "successful_tool_call_count": successful_tool_calls,
        "error_tool_call_count": error_tool_calls,
        "model_latency_seconds": model_latency_seconds,
        "wall_seconds": wall_seconds,
        "evaluator_success": None if outcome is None else outcome.success,
        "passed_requirements": None if outcome is None else outcome.passed_requirements,
        "total_requirements": None if outcome is None else outcome.total_requirements,
    }


def _percentile(values: Sequence[int], percent: int) -> int | float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (percent / 100) * (len(ordered) - 1)
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[low]
    weight = position - low
    interpolated = ordered[low] * (1 - weight) + ordered[high] * weight
    rounded = round(interpolated, 6)
    if rounded == int(rounded):
        return int(rounded)
    return rounded


def _distribution(values: Sequence[int], percents: Sequence[int]) -> dict[str, object]:
    if not values:
        return {f"p{percent}": None for percent in percents} | {"max": None}
    return {f"p{percent}": _percentile(values, percent) for percent in percents} | {
        "max": max(values)
    }


def _parser_errors(
    configuration: RunConfiguration,
    episode: EpisodeResult,
) -> int:
    workflow = configuration.agent.workflow
    if workflow is not None and workflow.policy in {"plan_progress_v1", "plan_progress_v2"}:
        return sum(
            0
            if workflow_output_has_parseable_action(step.output_text)
            else 1
            for step in episode.model_steps
        )
    count = 0
    for step in episode.model_steps:
        text = step.output_text
        if text == "STOP" or text.startswith("STOP\n"):
            continue
        try:
            parse_model_output(text)
        except ActionRejected:
            count += 1
    return count


def _repeated_tool_calls(episode: EpisodeResult) -> int:
    previous: str | None = None
    repeats = 0
    for step in episode.tool_steps:
        if previous is not None and step.action == previous:
            repeats += 1
        previous = step.action
    return repeats


def _success_difficulty(episode: EpisodeResult, manifest_difficulty: int | None) -> int | None:
    outcome = episode.evaluator_outcome
    if episode.termination_reason == "runtime_error" or outcome is None or not outcome.success:
        return None
    if outcome.difficulty is not None:
        return outcome.difficulty
    return manifest_difficulty


def _workflow_accounting(
    configuration: RunConfiguration,
    episode: EpisodeResult,
) -> WorkflowTraceAccounting:
    generations = len(episode.model_steps)
    executed = len(episode.tool_steps)
    workflow = configuration.agent.workflow
    if workflow is None:
        return WorkflowTraceAccounting(
            model_generation_count=generations,
            workflow_envelope_rejection_count=0,
            repeat_gate_block_count=0,
            completion_gate_block_count=0,
            stall_event_count=0,
            executed_tool_action_count=executed,
            productive_model_turns=generations,
            workflow_format_retry_count=0,
        )
    return account_workflow_trace(
        [(step.index, step.output_text) for step in episode.model_steps],
        [
            (step.index, step.action, step.error is not None)
            for step in episode.tool_steps
        ],
        workflow,
    )


def _aggregate(
    *,
    configuration: RunConfiguration,
    episodes: Sequence[EpisodeResult],
    manifest_difficulties: Sequence[int | None],
) -> dict[str, object]:
    token_cap = configuration.agent.sampling.generation_max_tokens("execute")
    turn_cap = configuration.agent.execute_turn_limit
    successes = 0
    easy = 0
    medium = 0
    other = 0
    parser_errors = 0
    recoverable_errors = 0
    repeated = 0
    token_cap_hits = 0
    turn_cap_hits = 0
    productive_turn_cap_hits = 0
    envelope_rejections = 0
    repeat_blocks = 0
    completion_blocks = 0
    stall_events = 0
    model_generations = 0
    executed_actions = 0
    format_retries = 0
    generated_tokens: list[int] = []
    termination_counts = {reason: 0 for reason in sorted(EXECUTE_TERMINATIONS)}
    for episode, manifest_difficulty in zip(episodes, manifest_difficulties, strict=True):
        outcome = episode.evaluator_outcome
        if (
            episode.termination_reason != "runtime_error"
            and outcome is not None
            and outcome.success
        ):
            successes += 1
        label = _success_difficulty(episode, manifest_difficulty)
        if label == _DIFFICULTY_EASY:
            easy += 1
        elif label == _DIFFICULTY_MEDIUM:
            medium += 1
        elif label is not None:
            other += 1
        parser_errors += _parser_errors(configuration, episode)
        repeated += _repeated_tool_calls(episode)
        workflow_counts = _workflow_accounting(configuration, episode)
        envelope_rejections += workflow_counts.workflow_envelope_rejection_count
        repeat_blocks += workflow_counts.repeat_gate_block_count
        completion_blocks += workflow_counts.completion_gate_block_count
        stall_events += workflow_counts.stall_event_count
        model_generations += workflow_counts.model_generation_count
        executed_actions += workflow_counts.executed_tool_action_count
        format_retries += workflow_counts.workflow_format_retry_count
        if len(episode.model_steps) >= turn_cap:
            turn_cap_hits += 1
        if workflow_counts.productive_model_turns >= turn_cap:
            productive_turn_cap_hits += 1
        for step in episode.model_steps:
            generated_tokens.append(step.generated_token_count)
            if step.generated_token_count >= token_cap:
                token_cap_hits += 1
        for step in episode.tool_steps:
            if step.error is not None and step.error.recoverable:
                recoverable_errors += 1
        termination_counts[episode.termination_reason] = (
            termination_counts.get(episode.termination_reason, 0) + 1
        )
    task_count = len(episodes)
    return {
        "task_count": task_count,
        "configuration_hash": run_configuration_hash(configuration),
        "task_set_hash": configuration.task.task_set_hash,
        "git_commit": configuration.git_commit,
        "mode": "execute",
        "concurrency": 1,
        "evaluator_success_count": successes,
        "evaluator_success_rate": None if task_count == 0 else successes / task_count,
        "complete_task_termination_count": termination_counts.get("appworld_completed", 0),
        "model_turns": _distribution(
            [len(episode.model_steps) for episode in episodes],
            (50, 90, 95),
        ),
        "tool_calls": _distribution(
            [len(episode.tool_steps) for episode in episodes],
            (50, 90, 95),
        ),
        "generated_tokens": _distribution(generated_tokens, (50, 90, 95, 99)),
        "generations_hitting_execute_token_cap": token_cap_hits,
        "execute_token_cap": token_cap,
        "episodes_hitting_execute_turn_cap": turn_cap_hits,
        "execute_turn_cap": turn_cap,
        "productive_action_turn_cap_hits": productive_turn_cap_hits,
        "workflow_envelope_rejection_count": envelope_rejections,
        "repeat_gate_block_count": repeat_blocks,
        "completion_gate_block_count": completion_blocks,
        "stall_event_count": stall_events,
        "model_generation_count": model_generations,
        "executed_tool_action_count": executed_actions,
        "workflow_format_retry_count": format_retries,
        "parser_error_count": parser_errors,
        "recoverable_tool_error_count": recoverable_errors,
        "repeated_tool_call_count": repeated,
        "termination_reason_counts": termination_counts,
        "easy_success_count": easy,
        "medium_success_count": medium,
        "other_success_count": other,
    }


def _terminal_status(episode: EpisodeResult | None) -> str:
    if episode is None:
        return "not_attempted"
    if episode.termination_reason == "runtime_error":
        return "runtime_failure"
    outcome = episode.evaluator_outcome
    if outcome is not None and outcome.success:
        return "evaluator_success"
    return "evaluator_failure"


def _evaluator_status(episode: EpisodeResult | None) -> str:
    if episode is None or episode.evaluator_outcome is None:
        return "unavailable"
    if episode.evaluator_outcome.success:
        return "success"
    return "failure"


def _public_runtime_failure(episode: EpisodeResult | None) -> tuple[str | None, str | None]:
    if episode is None or episode.termination_reason != "runtime_error":
        return None, None
    message = (
        episode.episode_errors[0].message
        if episode.episode_errors
        else "runtime_error"
    )
    if message.startswith("context_length_exceeded"):
        return "context_length_exceeded", message
    if "HTTP 400" in message:
        return "model_http_400", message
    return "runtime_error", message


def _ratio(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


def _result_slots(
    episodes: Sequence[EpisodeResult],
    manifest_difficulties: Sequence[int | None],
    *,
    expected: int,
    task_ids: Sequence[str],
    difficulties: dict[str, int],
) -> tuple[list[EpisodeResult | None], list[int | None]]:
    slots: list[EpisodeResult | None] = list(episodes)
    labels: list[int | None] = list(manifest_difficulties)
    while len(slots) < expected:
        index = len(slots)
        task_id = task_ids[index] if index < len(task_ids) else None
        slots.append(None)
        labels.append(None if task_id is None else difficulties.get(task_id))
    return slots, labels


def _annotate_terminal(
    payload: dict[str, object],
    *,
    slots: Sequence[EpisodeResult | None],
    run_status: str,
    failure_reason: str | None,
    failed_episode: int | None,
) -> dict[str, object]:
    counts = {status: 0 for status in _TERMINAL_STATUSES}
    unavailable = 0
    scored_success = 0
    scored_failure = 0
    for episode in slots:
        counts[_terminal_status(episode)] += 1
        if _evaluator_status(episode) == "unavailable":
            unavailable += 1
        if (
            episode is not None
            and episode.termination_reason != "runtime_error"
            and episode.evaluator_outcome is not None
        ):
            if episode.evaluator_outcome.success:
                scored_success += 1
            else:
                scored_failure += 1
    expected = len(slots)
    end_to_end = counts["evaluator_success"]
    scored = scored_success + scored_failure
    attempted = expected - counts["not_attempted"]
    payload["run_status"] = run_status
    payload["pilot_status"] = run_status
    payload["task_count"] = expected
    payload["episodes_expected"] = expected
    payload["episodes_attempted"] = attempted
    payload["episodes_completed"] = counts["evaluator_success"] + counts["evaluator_failure"]
    payload["attempted_task_count"] = attempted
    payload["not_attempted_count"] = counts["not_attempted"]
    payload["runtime_failure_count"] = counts["runtime_failure"]
    payload["evaluator_failure_count"] = counts["evaluator_failure"]
    payload["evaluator_success_count"] = end_to_end
    payload["evaluator_success_rate"] = _ratio(end_to_end, expected)
    payload["end_to_end_success_count"] = end_to_end
    payload["end_to_end_success_rate"] = _ratio(end_to_end, expected)
    payload["evaluator_only_success_count"] = scored_success
    payload["evaluator_only_success_rate"] = _ratio(scored_success, scored)
    payload["evaluator_unavailable_count"] = unavailable
    payload["terminal_status_counts"] = counts
    if run_status == "aborted":
        payload["failure_type"] = "experiment_error"
        payload["failure_reason"] = failure_reason or "experiment_error"
        payload["failed_episode"] = failed_episode
    return payload


def _episode_row(
    configuration: RunConfiguration,
    index: int,
    episode: EpisodeResult | None,
    difficulty: int | None,
) -> dict[str, object]:
    if episode is None:
        return {
            "task_index": index,
            "difficulty": difficulty,
            "terminal_status": "not_attempted",
            "evaluator_status": "unavailable",
            "evaluator_success": None,
            "passed_requirements": None,
            "total_requirements": None,
            "requirement_fraction": None,
            "model_step_count": 0,
            "tool_step_count": 0,
            "termination_reason": None,
            "runtime_error_type": None,
            "runtime_error_message": None,
            "parser_error_count": 0,
            "workflow_envelope_rejection_count": 0,
            "repeat_gate_block_count": 0,
            "completion_gate_block_count": 0,
            "stall_event_count": 0,
        }
    outcome = episode.evaluator_outcome
    episode_passed = None if outcome is None else outcome.passed_requirements
    episode_total = None if outcome is None else outcome.total_requirements
    if outcome is not None and outcome.difficulty is not None:
        difficulty = outcome.difficulty
    error_type, error_message = _public_runtime_failure(episode)
    accounting = _workflow_accounting(configuration, episode)
    return {
        "task_index": index,
        "difficulty": difficulty,
        "terminal_status": _terminal_status(episode),
        "evaluator_status": _evaluator_status(episode),
        "evaluator_success": None if outcome is None else outcome.success,
        "passed_requirements": episode_passed,
        "total_requirements": episode_total,
        "requirement_fraction": (
            None
            if episode_passed is None or episode_total is None
            else _requirement_fraction(episode_passed, episode_total)
        ),
        "model_step_count": len(episode.model_steps),
        "tool_step_count": len(episode.tool_steps),
        "termination_reason": episode.termination_reason,
        "runtime_error_type": error_type,
        "runtime_error_message": error_message,
        "parser_error_count": _parser_errors(configuration, episode),
        "workflow_envelope_rejection_count": accounting.workflow_envelope_rejection_count,
        "repeat_gate_block_count": accounting.repeat_gate_block_count,
        "completion_gate_block_count": accounting.completion_gate_block_count,
        "stall_event_count": accounting.stall_event_count,
    }


def _with_diagnostic_fields(
    payload: dict[str, object],
    *,
    configuration: RunConfiguration,
    slots: Sequence[EpisodeResult | None],
    manifest_difficulties: Sequence[int | None],
) -> dict[str, object]:
    if configuration.task.selection_rule != DIAGNOSTIC_RULE:
        return payload
    real = [episode for episode in slots if episode is not None]
    model_turns = [len(episode.model_steps) for episode in real]
    tool_calls = [len(episode.tool_steps) for episode in real]
    passed = 0
    total = 0
    for episode in real:
        outcome = episode.evaluator_outcome
        if outcome is not None:
            passed += outcome.passed_requirements
            total += outcome.total_requirements
    rows = [
        _episode_row(configuration, index, episode, difficulty)
        for index, (episode, difficulty) in enumerate(
            zip(slots, manifest_difficulties, strict=True)
        )
    ]
    payload["selection_rule"] = DIAGNOSTIC_RULE
    payload["model_repository"] = configuration.model.model.repository
    payload["model_revision"] = configuration.model.model.revision
    payload["requirements_passed"] = passed
    payload["requirements_total"] = total
    payload["requirement_pass_fraction"] = _requirement_fraction(passed, total)
    payload["model_turn_mean"] = _mean(model_turns)
    payload["model_turn_median"] = _median(model_turns)
    payload["tool_call_mean"] = _mean(tool_calls)
    payload["tool_call_median"] = _median(tool_calls)
    payload["episodes"] = rows
    return payload


def _required_task_count(task: TaskConfiguration) -> int:
    if task.selection_rule == DIAGNOSTIC_RULE:
        return DIAGNOSTIC_TASK_COUNT
    return _PILOT_TASK_COUNT


def _mean(values: Sequence[int]) -> int | float | None:
    if not values:
        return None
    value = sum(values) / len(values)
    rounded = round(value, 6)
    if rounded == int(rounded):
        return int(rounded)
    return rounded


def _median(values: Sequence[int]) -> int | float | None:
    if not values:
        return None
    ordered = sorted(values)
    midpoint = len(ordered) // 2
    if len(ordered) % 2 == 1:
        return ordered[midpoint]
    return _mean((ordered[midpoint - 1], ordered[midpoint]))


def _requirement_fraction(passed: int, total: int) -> int | float | None:
    if total == 0:
        return None
    value = passed / total
    rounded = round(value, 6)
    if rounded == int(rounded):
        return int(rounded)
    return rounded


def _require_pilot_configuration(agent: AgentConfiguration) -> None:
    if agent.tool_access_profile != _PILOT_PROFILE:
        print("capability pilot requires the Spotify tool access profile", file=sys.stderr)
        raise SystemExit(2)
    if agent.execute_max_model_turns != _PILOT_TURN_CAP:
        print("capability pilot requires 20 execute turns", file=sys.stderr)
        raise SystemExit(2)
    if agent.sampling.execute_max_tokens != _PILOT_TOKEN_CAP:
        print("capability pilot requires 192 execute tokens", file=sys.stderr)
        raise SystemExit(2)
    if agent.sampling.temperature != 0.0:
        print("capability pilot requires temperature 0", file=sys.stderr)
        raise SystemExit(2)
    if agent.sampling.seed != _PILOT_SEED:
        print("capability pilot requires seed 17", file=sys.stderr)
        raise SystemExit(2)


def _coerce_exit_code(error: BaseException) -> int:
    if type(error) is not SystemExit:
        return 1
    code = getattr(error, "code", None)
    if code is None:
        return 2
    if isinstance(code, bool):
        return 1
    if isinstance(code, int):
        return code
    if isinstance(code, str):
        stripped = code.strip()
        negative = stripped.startswith("-")
        digits = stripped[1:] if negative else stripped
        if digits.isdigit():
            value = int(digits)
            return -value if negative else value
        return 1
    return 1


def _announce_failure(error: BaseException | str) -> None:
    text = error if isinstance(error, str) else runtime_failure_message(error)
    try:
        print("capability pilot failed", file=sys.stderr)
        if text.strip() != "":
            print(text, file=sys.stderr)
    except Exception:
        try:
            print("capability pilot failed", file=sys.stderr)
        except Exception:
            return


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _runtime_failure_result(
    *,
    identity: EpisodeIdentity,
    run: RunIdentity,
    configuration: RunConfiguration,
    task_id: str,
    scenario_id: str | None,
    started_at: datetime,
    ended_at: datetime,
    model_steps: tuple[ModelStep, ...],
    tool_steps: tuple[ToolStep, ...],
    message: str,
) -> EpisodeResult:
    return EpisodeResult(
        episode=identity,
        run=run,
        task=LocalTaskRef(
            task_id=task_id,
            scenario_id=scenario_id,
            split=configuration.task.split,
        ),
        mode="execute",
        execution_seed=configuration.agent.sampling.seed,
        status="failed",
        started_at=started_at,
        ended_at=ended_at,
        model_steps=model_steps,
        tool_steps=tool_steps,
        plan_text=None,
        evaluator_outcome=None,
        termination_reason="runtime_error",
        episode_errors=(
            RecordedError(
                source="runtime",
                recoverable=False,
                message=message,
                step_index=None,
            ),
        ),
        role=None,
    )


def _close_open_runtime_failure(
    store: EpisodeStore,
    episode_id: str,
    *,
    configuration: RunConfiguration,
    scenario_id: str | None,
    message: str,
) -> EpisodeResult | None:
    try:
        opened = store.load_open_episode(episode_id)
    except StorageError:
        return None
    model_steps = tuple(step for step in opened.steps if isinstance(step, ModelStep))
    tool_steps = tuple(step for step in opened.steps if isinstance(step, ToolStep))
    started_at = min((step.started_at for step in opened.steps), default=wall_now())
    ended_at = wall_now()
    if ended_at < started_at:
        ended_at = started_at
    episode = _runtime_failure_result(
        identity=opened.identity,
        run=opened.run,
        configuration=configuration,
        task_id=opened.task_id,
        scenario_id=scenario_id,
        started_at=started_at,
        ended_at=ended_at,
        model_steps=model_steps,
        tool_steps=tool_steps,
        message=message,
    )
    try:
        store.finish_episode(episode)
    except StorageError:
        return None
    return episode


def main(argv: Sequence[str] | None = None) -> int:
    if argv is None and len(sys.argv) <= 1:
        return 2
    args_list = list(sys.argv[1:] if argv is None else argv)
    if not args_list:
        return 2

    parser = argparse.ArgumentParser(
        description=(
            "Run the 20-task Spotify capability pilot in execute mode. "
            "Prints one public JSON object per episode and writes an aggregate."
        )
    )
    parser.add_argument("--configuration", required=True)
    parser.add_argument("--task-set", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--store", required=True)
    parser.add_argument("--output", required=True)
    try:
        args = parser.parse_args(args_list)
    except SystemExit as error:
        return _coerce_exit_code(error)

    store_path = Path(args.store)
    output_path = Path(args.output)
    try:
        _refuse_results_store(store_path)
    except SystemExit as error:
        return _coerce_exit_code(error)
    if not store_path.parent.is_dir():
        print("store parent directory does not exist", file=sys.stderr)
        return 2

    configuration_path = Path(args.configuration)
    task_set_path = Path(args.task_set)
    for path in (configuration_path, task_set_path):
        if not path.is_file():
            print(f"missing file: {path}", file=sys.stderr)
            return 2

    appworld_root = os.environ.get("APPWORLD_ROOT")
    if appworld_root is None or appworld_root.strip() == "":
        print("APPWORLD_ROOT must be set to an existing directory", file=sys.stderr)
        return 2
    if not Path(appworld_root).is_dir():
        print("APPWORLD_ROOT must be set to an existing directory", file=sys.stderr)
        return 2

    store: EpisodeStore | None = None
    configuration: RunConfiguration | None = None
    episodes: list[EpisodeResult] = []
    manifest_difficulties: list[int | None] = []
    attempted_index: int | None = None
    open_holder: dict[str, str] = {}
    open_task_id: str | None = None
    open_scenario_id: str | None = None
    difficulties: dict[str, int] = {}
    task_ids: tuple[str, ...] = ()

    def publish(
        run_status: str,
        *,
        failure_reason: str | None = None,
        failed_episode: int | None = None,
    ) -> None:
        if configuration is None:
            return
        expected = _required_task_count(configuration.task)
        slots, labels = _result_slots(
            episodes,
            manifest_difficulties,
            expected=expected,
            task_ids=task_ids,
            difficulties=difficulties,
        )
        payload = _annotate_terminal(
            _aggregate(
                configuration=configuration,
                episodes=episodes,
                manifest_difficulties=manifest_difficulties,
            ),
            slots=slots,
            run_status=run_status,
            failure_reason=failure_reason,
            failed_episode=failed_episode,
        )
        _write_json(
            output_path,
            _with_diagnostic_fields(
                payload,
                configuration=configuration,
                slots=slots,
                manifest_difficulties=labels,
            ),
        )

    def discard_open_episode(message: str) -> None:
        episode_id = open_holder.get("episode_id")
        if (
            store is None
            or configuration is None
            or not episode_id
            or open_task_id is None
        ):
            return
        _close_open_runtime_failure(
            store,
            episode_id,
            configuration=configuration,
            scenario_id=open_scenario_id,
            message=message,
        )
        open_holder.clear()

    def capture_episode_local(
        error: BaseException,
        run: RunIdentity,
        task_id: str,
        scenario_id: str | None,
    ) -> EpisodeResult:
        message = runtime_failure_message(error)
        episode_id = open_holder.get("episode_id")
        if episode_id:
            if store is None or configuration is None:
                raise StorageError("failed episode could not be closed")
            closed = _close_open_runtime_failure(
                store,
                episode_id,
                configuration=configuration,
                scenario_id=scenario_id,
                message=message,
            )
            if closed is None:
                raise StorageError("failed episode could not be closed")
            open_holder.clear()
            return closed
        if store is None or configuration is None:
            raise StorageError("failed episode could not be closed")
        now = wall_now()
        identity = new_episode_identity(run)
        episode = _runtime_failure_result(
            identity=identity,
            run=run,
            configuration=configuration,
            task_id=task_id,
            scenario_id=scenario_id,
            started_at=now,
            ended_at=now,
            model_steps=(),
            tool_steps=(),
            message=message,
        )
        store.start_episode(identity, run, task_id)
        store.finish_episode(episode)
        return episode

    def abort_experiment(error: BaseException) -> None:
        try:
            reason = runtime_failure_message(error)
            discard_open_episode(reason)
            publish(
                "aborted",
                failure_reason=reason,
                failed_episode=attempted_index,
            )
        except Exception:
            return

    try:
        configuration_payload = _load_json(configuration_path)
        if not isinstance(configuration_payload, dict):
            print("configuration must be an object", file=sys.stderr)
            return 2
        if "model" not in configuration_payload or "agent" not in configuration_payload:
            print("configuration must contain model and agent", file=sys.stderr)
            return 2
        model = ModelConfiguration.from_dict(configuration_payload["model"])
        agent = AgentConfiguration.from_dict(configuration_payload["agent"])
        _require_pilot_configuration(agent)
        task_payload = _load_json(task_set_path)
        task, task_set = _load_task_set(task_payload)
        expected_tasks = _required_task_count(task)
        if task.task_count != expected_tasks or task_set.task_count != expected_tasks:
            print(f"capability pilot requires {expected_tasks} tasks", file=sys.stderr)
            return 2
        adopted = _adopt_committed_setup_profile(task)
        if adopted is None:
            print("task set does not match committed public metadata", file=sys.stderr)
            return 2
        task = adopted
        difficulties = _difficulty_by_task(task_payload)
        task_ids = tuple(task_set.task_ids)
        configuration = RunConfiguration(
            model=model,
            agent=agent,
            task=task,
            run_seed=task.selection_seed,
            git_commit=_git_commit(),
            protocol_hash=None,
        )
        enforce_committed_provenance(configuration)
        runtime = build_runtime(
            configuration,
            args.base_url,
            mode="execute",
        )
        store = EpisodeStore(store_path)
        for task_index, (task_id, scenario_id) in enumerate(
            zip(task_set.task_ids, task_set.scenario_ids, strict=True)
        ):
            attempted_index = task_index
            open_task_id = task_id
            open_scenario_id = scenario_id
            open_holder.clear()
            run = new_run_identity(configuration)
            on_start, on_step = _persistence_callbacks(store, task_id, open_holder)
            try:
                episode = run_episode(
                    task_id,
                    configuration,
                    "execute",
                    run=run,
                    runtime=runtime,
                    on_start=on_start,
                    on_step=on_step,
                    scenario_id=scenario_id,
                    evaluate_after_runtime_failure=True,
                )
            except _EXPERIMENT_FAILURES:
                raise
            except Exception as error:
                episode = capture_episode_local(error, run, task_id, scenario_id)
            else:
                store.finish_episode(episode)
                open_holder.clear()
            episodes.append(episode)
            manifest_difficulties.append(difficulties.get(task_id))
            summary = _public_summary(
                configuration=configuration,
                task_index=task_index,
                episode=episode,
            )
            print(json.dumps(summary, sort_keys=True), flush=True)
        publish("completed")
        return 0
    except SystemExit as error:
        nested = getattr(error, "code", None) if type(error) is SystemExit else None
        if type(error) is not SystemExit or (
            isinstance(nested, BaseException) and not isinstance(nested, SystemExit)
        ):
            reported = nested if isinstance(nested, BaseException) else error
            abort_experiment(reported)
            _announce_failure(reported)
            return 1
        try:
            discard_open_episode(runtime_failure_message(error))
        except Exception:
            pass
        return _coerce_exit_code(error)
    except Exception as error:
        abort_experiment(error)
        _announce_failure(error)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
