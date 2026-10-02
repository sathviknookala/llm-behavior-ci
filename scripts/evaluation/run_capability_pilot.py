"""Run the Spotify capability pilot: twenty execute episodes, one at a time.

Requires --configuration, --task-set, --base-url, --store, and --output.
The configuration must be the committed Spotify capability profile:
execute horizon 20, execute token cap 192, temperature 0, seed 17, and
``tool_access_profile`` ``spotify_capability_v1``. ``git_commit`` must be
HEAD and the tracked source and config tree must be clean.

Prints one public JSON object after each episode. Writes one public
aggregate to --output. Task ids, instructions, model text, API arguments,
trajectories, and evaluator internals stay in the local store. When the
local manifest omits ``appworld_setup_profile``, the matching committed
task metadata supplies it.
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
from pathlib import Path

from llm_behavior_ci.config import (
    AgentConfiguration,
    ConfigError,
    EpisodeIdentity,
    ModelConfiguration,
    RunConfiguration,
    RunIdentity,
    TaskConfiguration,
    new_run_identity,
    run_configuration_hash,
)
from llm_behavior_ci.records import EXECUTE_TERMINATIONS, EpisodeResult, ModelStep, ToolStep
from llm_behavior_ci.runtime.actions import ActionRejected, parse_model_output
from llm_behavior_ci.runtime.workflow import workflow_output_has_parseable_action
from llm_behavior_ci.runtime.episode import (
    EpisodeRejected,
    RuntimeUnavailable,
    build_runtime,
    run_episode,
)
from llm_behavior_ci.runtime.provenance import ProvenanceError, enforce_committed_provenance
from llm_behavior_ci.storage import EpisodeStore, StorageError
from llm_behavior_ci.tasks.selection import SelectionError, TaskSet, verify_task_set

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
) -> tuple[
    Callable[[EpisodeIdentity, RunIdentity], None],
    Callable[[ModelStep | ToolStep], None],
]:
    current: dict[str, str] = {}

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
    wall_seconds = (episode.ended_at - episode.started_at).total_seconds()
    model_latency_seconds = sum(step.latency_seconds for step in episode.model_steps)
    successful_tool_calls = sum(1 for step in episode.tool_steps if step.error is None)
    error_tool_calls = sum(1 for step in episode.tool_steps if step.error is not None)
    return {
        "configuration_hash": run_configuration_hash(configuration),
        "task_set_hash": configuration.task.task_set_hash,
        "task_index": task_index,
        "termination_reason": episode.termination_reason,
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
    if workflow is not None and workflow.policy == "plan_progress_v1":
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
    if outcome is None or not outcome.success:
        return None
    if outcome.difficulty is not None:
        return outcome.difficulty
    return manifest_difficulty


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
    generated_tokens: list[int] = []
    termination_counts = {reason: 0 for reason in sorted(EXECUTE_TERMINATIONS)}
    for episode, manifest_difficulty in zip(episodes, manifest_difficulties, strict=True):
        outcome = episode.evaluator_outcome
        if outcome is not None and outcome.success:
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
        if len(episode.model_steps) >= turn_cap:
            turn_cap_hits += 1
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
        "parser_error_count": parser_errors,
        "recoverable_tool_error_count": recoverable_errors,
        "repeated_tool_call_count": repeated,
        "termination_reason_counts": termination_counts,
        "easy_success_count": easy,
        "medium_success_count": medium,
        "other_success_count": other,
    }


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
        code = error.code
        if code is None:
            return 2
        return int(code)

    store_path = Path(args.store)
    output_path = Path(args.output)
    try:
        _refuse_results_store(store_path)
    except SystemExit as error:
        code = error.code
        return 2 if code is None else int(code)
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
        if task.task_count != _PILOT_TASK_COUNT or task_set.task_count != _PILOT_TASK_COUNT:
            print("capability pilot requires 20 tasks", file=sys.stderr)
            return 2
        adopted = _adopt_committed_setup_profile(task)
        if adopted is None:
            print("task set does not match committed public metadata", file=sys.stderr)
            return 2
        task = adopted
        difficulties = _difficulty_by_task(task_payload)
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
        episodes: list[EpisodeResult] = []
        manifest_difficulties: list[int | None] = []
        for task_index, (task_id, scenario_id) in enumerate(
            zip(task_set.task_ids, task_set.scenario_ids, strict=True)
        ):
            run = new_run_identity(configuration)
            on_start, on_step = _persistence_callbacks(store, task_id)
            episode = run_episode(
                task_id,
                configuration,
                "execute",
                run=run,
                runtime=runtime,
                on_start=on_start,
                on_step=on_step,
                scenario_id=scenario_id,
            )
            store.finish_episode(episode)
            episodes.append(episode)
            manifest_difficulties.append(difficulties.get(task_id))
            summary = _public_summary(
                configuration=configuration,
                task_index=task_index,
                episode=episode,
            )
            print(json.dumps(summary, sort_keys=True), flush=True)
        aggregate = _aggregate(
            configuration=configuration,
            episodes=episodes,
            manifest_difficulties=manifest_difficulties,
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(aggregate, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        return 0
    except SystemExit as error:
        code = error.code
        return 2 if code is None else int(code)
    except (ConfigError, SelectionError, ProvenanceError) as error:
        print(str(error) or "capability pilot failed", file=sys.stderr)
        return 1
    except (
        KeyError,
        TypeError,
        ValueError,
        EpisodeRejected,
        RuntimeUnavailable,
        StorageError,
        OSError,
    ):
        print("capability pilot failed", file=sys.stderr)
        return 1
    except Exception:
        print("capability pilot failed", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
