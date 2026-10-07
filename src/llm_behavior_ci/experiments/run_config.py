"""Build a full ``RunConfiguration`` from a committed template and a local task set.

A model template under ``configs/models/`` holds the ``model`` and
``agent`` sections. A local task-set manifest holds the public
``TaskConfiguration`` fields plus the protected ``task_ids`` and
``scenario_ids``, and optionally per-task ``difficulty_by_task`` and
``required_apps_by_task``. The builder verifies the manifest's canonical
hash, adopts the committed ``appworld_setup_profile`` for the same public
task identity, and fills ``run_seed``, ``git_commit``, and
``protocol_hash``. Nothing here edits a hash by hand: the configuration
hash is recomputed from the result.

``preflight_tool_access`` checks the agent's tool-access and setup
profiles against the task metadata before any world opens. The Spotify
profiles are scoped to tasks whose non-infrastructure required apps are
exactly ``{spotify}``; a general template sets neither profile.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

from llm_behavior_ci.config import (
    AgentConfiguration,
    ConfigError,
    RunConfiguration,
    TaskConfiguration,
    canonical_configuration_json,
    load_model_configuration,
    run_configuration_hash,
)
from llm_behavior_ci.tasks.selection import (
    SelectionError,
    TaskSet,
    canonical_task_set_bytes,
    task_set_hash_from_bytes,
)

_PUBLIC_TASK_FIELDS = (
    "appworld_version",
    "split",
    "selection_rule",
    "selection_seed",
    "task_count",
    "task_set_hash",
)
_INFRASTRUCTURE_APPS = frozenset({"admin", "api_docs", "supervisor"})
_SPOTIFY_PROFILES = frozenset({"spotify_capability_v1"})
_SPOTIFY_SETUP_PROFILES = frozenset({"spotify_authenticated_v1"})
SPOTIFY_ONLY_SELECTION_RULES = frozenset(
    {"fixed_spotify_capability", "spotify_short_horizon_diagnostic_v1"}
)


class RunConfigError(ValueError):
    """A run configuration cannot be built or fails its preflight."""


@dataclass(frozen=True)
class LocalTaskManifest:
    """A verified local task set and its optional per-task metadata.

    ``difficulty_by_task`` and ``required_apps_by_task`` are split-available
    task metadata. They are never derived from an evaluator outcome.
    """

    task: TaskConfiguration
    task_set: TaskSet
    difficulty_by_task: tuple[tuple[str, int], ...] = ()
    required_apps_by_task: tuple[tuple[str, tuple[str, ...]], ...] = ()

    def difficulty(self, task_id: str) -> int | None:
        for name, value in self.difficulty_by_task:
            if name == task_id:
                return value
        return None

    def required_apps(self, task_id: str) -> tuple[str, ...] | None:
        for name, value in self.required_apps_by_task:
            if name == task_id:
                return value
        return None


def _load_json(path: Path) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RunConfigError(f"{path} is not readable JSON") from error


def task_set_from_payload(payload: object) -> TaskSet:
    """A ``TaskSet`` whose ``task_set_hash`` matches its canonical bytes."""

    if not isinstance(payload, Mapping):
        raise RunConfigError("task set must be an object")
    try:
        task_set = TaskSet(
            appworld_version=str(payload["appworld_version"]),
            split=str(payload["split"]),
            selection_rule=str(payload["selection_rule"]),
            selection_seed=int(payload["selection_seed"]),
            task_count=int(payload["task_count"]),
            scenario_count=int(payload["scenario_count"]),
            task_ids=tuple(payload["task_ids"]),
            scenario_ids=tuple(payload["scenario_ids"]),
            task_set_hash=str(payload["task_set_hash"]),
        )
    except (KeyError, TypeError, ValueError, SelectionError) as error:
        raise RunConfigError("task set is incomplete") from error
    digest = canonical_task_set_bytes(
        appworld_version=task_set.appworld_version,
        split=task_set.split,
        selection_rule=task_set.selection_rule,
        selection_seed=task_set.selection_seed,
        tasks=tuple(zip(task_set.task_ids, task_set.scenario_ids, strict=True)),
    )
    if task_set_hash_from_bytes(digest) != task_set.task_set_hash:
        raise RunConfigError("task_set_hash does not match the canonical digest")
    return task_set


def _committed_setup_profile(
    task: TaskConfiguration,
    committed_tasks_dir: Path | None,
) -> str | None:
    if committed_tasks_dir is None or not committed_tasks_dir.is_dir():
        return None
    identity = {field: getattr(task, field) for field in _PUBLIC_TASK_FIELDS}
    for path in sorted(committed_tasks_dir.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict):
            continue
        if all(payload.get(field) == value for field, value in identity.items()):
            profile = payload.get("appworld_setup_profile")
            return profile if isinstance(profile, str) else None
    return None


def load_local_task_manifest(
    path: Path,
    *,
    committed_tasks_dir: Path | None = None,
) -> LocalTaskManifest:
    """Load and verify a local task-set manifest.

    The setup profile comes from the manifest when present, otherwise from
    the committed public task metadata with the same identity. A manifest
    profile that disagrees with the committed one is refused.
    """

    payload = _load_json(path)
    task_set = task_set_from_payload(payload)
    assert isinstance(payload, Mapping)
    manifest_profile = payload.get("appworld_setup_profile")
    if manifest_profile is not None and not isinstance(manifest_profile, str):
        raise RunConfigError("appworld_setup_profile must be a string")
    try:
        task = TaskConfiguration(
            appworld_version=task_set.appworld_version,
            split=task_set.split,
            selection_rule=task_set.selection_rule,
            selection_seed=task_set.selection_seed,
            task_count=task_set.task_count,
            task_set_hash=task_set.task_set_hash,
        )
    except ConfigError as error:
        raise RunConfigError(str(error)) from error
    committed = _committed_setup_profile(task, committed_tasks_dir)
    if manifest_profile is not None and committed is not None and manifest_profile != committed:
        raise RunConfigError("task set setup profile disagrees with committed metadata")
    profile = manifest_profile if manifest_profile is not None else committed
    if profile is not None:
        task = replace(task, appworld_setup_profile=profile)
    difficulties: list[tuple[str, int]] = []
    raw_difficulty = payload.get("difficulty_by_task") or {}
    if not isinstance(raw_difficulty, Mapping):
        raise RunConfigError("difficulty_by_task must be an object")
    for task_id, value in sorted(raw_difficulty.items()):
        if isinstance(value, bool) or value not in (1, 2, 3):
            raise RunConfigError("difficulty_by_task values must be 1, 2, or 3")
        difficulties.append((str(task_id), int(value)))
    apps: list[tuple[str, tuple[str, ...]]] = []
    raw_apps = payload.get("required_apps_by_task") or {}
    if not isinstance(raw_apps, Mapping):
        raise RunConfigError("required_apps_by_task must be an object")
    for task_id, value in sorted(raw_apps.items()):
        if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
            raise RunConfigError("required_apps_by_task values must be string lists")
        apps.append((str(task_id), tuple(sorted(value))))
    known = set(task_set.task_ids)
    if any(name not in known for name, _ in difficulties) or any(
        name not in known for name, _ in apps
    ):
        raise RunConfigError("task metadata names a task outside the task set")
    return LocalTaskManifest(
        task=task,
        task_set=task_set,
        difficulty_by_task=tuple(difficulties),
        required_apps_by_task=tuple(apps),
    )


def build_run_configuration(
    template: Mapping[str, object],
    task: TaskConfiguration,
    *,
    run_seed: int,
    git_commit: str,
    protocol_hash: str | None = None,
) -> RunConfiguration:
    """Compose a template's model and agent with a task binding."""

    if not isinstance(template, Mapping):
        raise RunConfigError("template must be an object")
    unknown = sorted(set(template) - {"model", "agent"})
    if unknown:
        raise RunConfigError(
            f"template may hold only model and agent; found {', '.join(unknown)}"
        )
    try:
        return RunConfiguration(
            model=load_model_configuration(template["model"]),
            agent=AgentConfiguration.from_dict(template["agent"]),
            task=task,
            run_seed=run_seed,
            git_commit=git_commit,
            protocol_hash=protocol_hash,
        )
    except (KeyError, ConfigError) as error:
        raise RunConfigError(f"template does not build a run configuration: {error}") from error


def _task_apps(required: Sequence[str]) -> frozenset[str]:
    return frozenset(app for app in required if app and app not in _INFRASTRUCTURE_APPS)


def preflight_tool_access(
    configuration: RunConfiguration,
    manifest: LocalTaskManifest,
) -> None:
    """Refuse a Spotify-scoped profile on tasks that need another app.

    ``spotify_capability_v1`` and ``spotify_authenticated_v1`` are valid only
    when every task's non-infrastructure required apps are exactly
    ``{spotify}``. Those profiles need ``required_apps_by_task`` for every
    task, except on a set built by a ``SPOTIFY_ONLY_SELECTION_RULES``
    builder, which refuses any task outside ``{spotify}`` when the set is
    made. A configuration with neither profile passes. The error carries
    counts, never task ids.
    """

    access = configuration.agent.tool_access_profile
    setup = configuration.task.appworld_setup_profile
    scoped = access in _SPOTIFY_PROFILES or setup in _SPOTIFY_SETUP_PROFILES
    if access is not None and access not in _SPOTIFY_PROFILES:
        raise RunConfigError(f"unknown tool_access_profile: {access}")
    if setup is not None and setup not in _SPOTIFY_SETUP_PROFILES:
        raise RunConfigError(f"unknown appworld_setup_profile: {setup}")
    if not scoped:
        return
    if (
        manifest.task_set.selection_rule in SPOTIFY_ONLY_SELECTION_RULES
        and not manifest.required_apps_by_task
    ):
        return
    task_ids = manifest.task_set.task_ids
    missing = sum(1 for task_id in task_ids if manifest.required_apps(task_id) is None)
    if missing:
        raise RunConfigError(
            "a Spotify-scoped profile needs required_apps_by_task for every task; "
            f"{missing} of {len(task_ids)} are missing"
        )
    outside = sum(
        1
        for task_id in task_ids
        if _task_apps(manifest.required_apps(task_id) or ()) != frozenset({"spotify"})
    )
    if outside:
        raise RunConfigError(
            "the Spotify-scoped profile excludes apps these tasks require; "
            f"{outside} of {len(task_ids)} tasks need an app other than spotify"
        )


def configuration_document(configuration: RunConfiguration) -> dict[str, object]:
    """The canonical JSON document of a configuration, ready to write locally."""

    return json.loads(canonical_configuration_json(configuration))


def configuration_summary(configuration: RunConfiguration) -> dict[str, object]:
    """Public identity of a built configuration: hashes and bindings, no task ids."""

    return {
        "configuration_hash": run_configuration_hash(configuration),
        "task_set_hash": configuration.task.task_set_hash,
        "split": configuration.task.split,
        "run_seed": configuration.run_seed,
        "git_commit": configuration.git_commit,
        "protocol_hash": configuration.protocol_hash,
        "appworld_setup_profile": configuration.task.appworld_setup_profile,
        "tool_access_profile": configuration.agent.tool_access_profile,
    }
