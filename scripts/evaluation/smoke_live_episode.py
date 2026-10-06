"""Run one execute-mode AppWorld episode against a live model.

Requires --configuration, --task-set, --task-index, and --store. A vLLM
configuration also requires --base-url. An Anthropic configuration reads
``ANTHROPIC_API_KEY`` and does not use a local server. APPWORLD_ROOT must
be set to an existing directory. Bare invocation exits 2. Store paths under
a directory named results are refused. Prints one public-safe JSON object.
Not a benchmark and not a gate. This development smoke stamps the current
HEAD and does not refuse a dirty tree. A capture, live gate, harm
measurement, or benchmark does.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

from llm_behavior_ci.config import (
    AgentConfiguration,
    AnthropicModelConfiguration,
    ConfigError,
    EpisodeIdentity,
    ModelConfiguration,
    RunConfiguration,
    RunIdentity,
    TaskConfiguration,
    load_model_configuration,
    new_run_identity,
    run_configuration_hash,
)
from llm_behavior_ci.records import ModelStep, ToolStep
from llm_behavior_ci.runtime.episode import (
    EpisodeRejected,
    RuntimeUnavailable,
    build_runtime,
    run_episode,
)
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
_REPO_ROOT = Path(__file__).resolve().parents[2]


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
    return {field: payload[field] for field in _PUBLIC_TASK_FIELDS}


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


def _matches_committed_public(task: TaskConfiguration) -> bool:
    tasks_dir = _REPO_ROOT / "configs" / "tasks"
    if not tasks_dir.is_dir():
        return False
    expected = task.to_dict()
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
        if committed.to_dict() == expected:
            return True
    return False


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
    episode,
) -> dict[str, object]:
    outcome = episode.evaluator_outcome
    wall_seconds = (episode.ended_at - episode.started_at).total_seconds()
    model_latency_seconds = sum(
        step.latency_seconds for step in episode.model_steps
    )
    successful_tool_calls = sum(
        1 for step in episode.tool_steps if step.error is None
    )
    error_tool_calls = sum(
        1 for step in episode.tool_steps if step.error is not None
    )
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
        "passed_requirements": (
            None if outcome is None else outcome.passed_requirements
        ),
        "total_requirements": (
            None if outcome is None else outcome.total_requirements
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    if argv is None and len(sys.argv) <= 1:
        return 2
    args_list = list(sys.argv[1:] if argv is None else argv)
    if not args_list:
        return 2

    parser = argparse.ArgumentParser(
        description=(
            "Run one execute-mode AppWorld episode against a live model. "
            "Prints one public-safe JSON object. Not a benchmark and not a gate."
        )
    )
    parser.add_argument("--configuration", required=True)
    parser.add_argument("--task-set", required=True)
    parser.add_argument("--task-index", required=True, type=int)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--store", required=True)
    try:
        args = parser.parse_args(args_list)
    except SystemExit as error:
        code = error.code
        if code is None:
            return 2
        return int(code)

    store_path = Path(args.store)
    try:
        _refuse_results_store(store_path)
    except SystemExit as error:
        code = error.code
        return 2 if code is None else int(code)

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
        model = load_model_configuration(configuration_payload["model"])
        agent = AgentConfiguration.from_dict(configuration_payload["agent"])
        if isinstance(model, ModelConfiguration):
            if not isinstance(args.base_url, str) or args.base_url.strip() == "":
                print("smoke episode requires --base-url", file=sys.stderr)
                return 2
            endpoint_url: str | None = args.base_url
        elif isinstance(model, AnthropicModelConfiguration):
            if args.base_url not in (None, ""):
                print("Anthropic smoke does not use --base-url", file=sys.stderr)
                return 2
            endpoint_url = None
        else:
            print("smoke episode requires a supported model provider", file=sys.stderr)
            return 2
        task, task_set = _load_task_set(_load_json(task_set_path))
        if not _matches_committed_public(task):
            print("task set does not match committed public metadata", file=sys.stderr)
            return 2
        if args.task_index < 0 or args.task_index >= task_set.task_count:
            print("task index out of range", file=sys.stderr)
            return 2
        configuration = RunConfiguration(
            model=model,
            agent=agent,
            task=task,
            run_seed=task.selection_seed,
            git_commit=_git_commit(),
            protocol_hash=None,
        )
        run = new_run_identity(configuration)
        runtime = build_runtime(configuration, endpoint_url, mode="execute")
        store = EpisodeStore(store_path)
        task_id = task_set.task_ids[args.task_index]
        scenario_id = task_set.scenario_ids[args.task_index]
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
        print(
            json.dumps(
                _public_summary(
                    configuration=configuration,
                    task_index=args.task_index,
                    episode=episode,
                ),
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        )
        return 0
    except SystemExit as error:
        code = error.code
        return 2 if code is None else int(code)
    except (
        ConfigError,
        SelectionError,
        KeyError,
        TypeError,
        ValueError,
        EpisodeRejected,
        RuntimeUnavailable,
        StorageError,
        OSError,
    ):
        print("smoke episode failed", file=sys.stderr)
        return 1
    except Exception:
        print("smoke episode failed", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
