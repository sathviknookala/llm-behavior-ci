"""Synthetic AppWorld catalog entries and loaders.

The catalog is a local JSON mapping of task ids, optional scenario ids,
splits, and optional difficulty. This module does not import appworld,
read environment variables, or search a data root.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from llm_behavior_ci.config import SPLITS

_MAX_TEXT = 256
_CATALOG_KEYS = frozenset({"appworld_version", "entries"})
_ENTRY_KEYS = frozenset({"task_id", "scenario_id", "split", "difficulty"})
_DIFFICULTIES = frozenset({1, 2, 3})


class CatalogError(ValueError):
    pass


class CatalogUnavailable(CatalogError):
    pass


def _line(value: object, name: str) -> str:
    if not isinstance(value, str) or value == "" or value != value.strip():
        raise CatalogError(f"{name} must be a non-empty string")
    if len(value) > _MAX_TEXT or any(
        ord(character) < 32 or ord(character) == 127 for character in value
    ):
        raise CatalogError(
            f"{name} must be a single line of at most {_MAX_TEXT} characters"
        )
    return value


def _exact_keys(payload: dict[object, object], allowed: frozenset[str], name: str) -> None:
    if not all(isinstance(key, str) for key in payload):
        raise CatalogError(f"{name} has a non-string field name")
    unknown = sorted(set(payload) - allowed)
    if unknown:
        joined = ", ".join(unknown)
        raise CatalogError(f"{name} contains unknown fields: {joined}")
    missing = sorted(allowed - set(payload))
    if missing:
        joined = ", ".join(missing)
        raise CatalogError(f"{name} is missing fields: {joined}")


@dataclass(frozen=True)
class CatalogEntry:
    task_id: str
    scenario_id: str | None
    split: str
    difficulty: int | None

    def __post_init__(self) -> None:
        _line(self.task_id, "task_id")
        if self.scenario_id is not None:
            _line(self.scenario_id, "scenario_id")
        if self.split not in SPLITS:
            choices = ", ".join(sorted(SPLITS))
            raise CatalogError(f"split must be one of: {choices}")
        if self.difficulty is not None and (
            isinstance(self.difficulty, bool) or self.difficulty not in _DIFFICULTIES
        ):
            raise CatalogError("difficulty must be 1, 2, 3, or null")


@dataclass(frozen=True)
class TaskCatalog:
    appworld_version: str
    entries: tuple[CatalogEntry, ...]

    def __post_init__(self) -> None:
        _line(self.appworld_version, "appworld_version")
        if not isinstance(self.entries, tuple):
            raise CatalogError("entries must be a tuple")
        seen: set[str] = set()
        for entry in self.entries:
            if not isinstance(entry, CatalogEntry):
                raise CatalogError("entries must contain catalog entries")
            if entry.task_id in seen:
                raise CatalogError("catalog contains duplicate task ids")
            seen.add(entry.task_id)


def catalog_from_mapping(payload: object) -> TaskCatalog:
    if not isinstance(payload, dict):
        raise CatalogError("catalog must be an object")
    _exact_keys(payload, _CATALOG_KEYS, "catalog")
    version = payload["appworld_version"]
    if not isinstance(version, str):
        raise CatalogError("appworld_version must be a string")
    entries_payload = payload["entries"]
    if not isinstance(entries_payload, list):
        raise CatalogError("entries must be a list")
    entries: list[CatalogEntry] = []
    for item in entries_payload:
        if not isinstance(item, dict):
            raise CatalogError("entry must be an object")
        _exact_keys(item, _ENTRY_KEYS, "entry")
        task_id = item["task_id"]
        scenario_id = item["scenario_id"]
        split = item["split"]
        difficulty = item["difficulty"]
        if not isinstance(task_id, str):
            raise CatalogError("task_id must be a string")
        if scenario_id is not None and not isinstance(scenario_id, str):
            raise CatalogError("scenario_id must be a string or null")
        if not isinstance(split, str):
            raise CatalogError("split must be a string")
        if difficulty is not None and (
            isinstance(difficulty, bool) or not isinstance(difficulty, int)
        ):
            raise CatalogError("difficulty must be an integer or null")
        entries.append(
            CatalogEntry(
                task_id=task_id,
                scenario_id=scenario_id,
                split=split,
                difficulty=difficulty,
            )
        )
    return TaskCatalog(appworld_version=version, entries=tuple(entries))


def load_appworld_catalog(path: Path) -> TaskCatalog:
    if not path.exists():
        raise CatalogUnavailable("catalog path does not exist")
    text = path.read_text(encoding="utf-8")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as error:
        raise CatalogError("catalog is not valid JSON") from error
    return catalog_from_mapping(payload)
