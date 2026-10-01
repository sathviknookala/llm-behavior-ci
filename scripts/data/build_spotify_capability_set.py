"""Build the fixed Spotify capability task set.

Requires --task-ids and --manifest. --task-ids is a local JSON list of the
twenty approved train task ids. Both paths must be ignored by git, because
resolved task ids stay local (docs/DATA.md). Opens each listed task, refuses
the set unless every task's required apps are exactly Spotify and its
difficulty is 1 or 2, then writes the local manifest to --manifest and the
public task metadata to configs/tasks/train_spotify_capability.json. The
public file is the hash, counts, split, rule, and seed only.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

from llm_behavior_ci.tasks.selection import (
    canonical_task_set_bytes,
    task_set_hash_from_bytes,
)

APPWORLD_VERSION = "0.1.3.post1"
SELECTION_RULE = "fixed_spotify_capability"
SELECTION_SEED = 17
TASK_COUNT = 20
PUBLIC_PATH = Path("configs/tasks/train_spotify_capability.json")

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _require_ignored(path: Path) -> None:
    completed = subprocess.run(
        ["git", "check-ignore", "-q", str(path.resolve())],
        cwd=_REPO_ROOT,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        print(f"path must be ignored by git: {path}", file=sys.stderr)
        raise SystemExit(2)


def _load_task_ids(path: Path) -> tuple[str, ...]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        print(f"cannot read task ids: {path}", file=sys.stderr)
        raise SystemExit(2) from error
    if not isinstance(payload, list) or not all(
        isinstance(task_id, str) and task_id for task_id in payload
    ):
        print("task ids must be a JSON list of non-empty strings", file=sys.stderr)
        raise SystemExit(2)
    if len(set(payload)) != len(payload):
        print("task ids must be unique", file=sys.stderr)
        raise SystemExit(2)
    if len(payload) != TASK_COUNT:
        print(f"task ids must list {TASK_COUNT} tasks", file=sys.stderr)
        raise SystemExit(2)
    return tuple(payload)


def _check_tasks(task_ids: Sequence[str]) -> tuple[list[tuple[str, str]], dict[str, int]]:
    from appworld import AppWorld
    from appworld.task import task_id_to_generator_id

    tasks: list[tuple[str, str]] = []
    difficulty_by_task: dict[str, int] = {}
    failures: list[str] = []

    for task_id in task_ids:
        world = AppWorld(
            task_id=task_id,
            ground_truth_mode="full",
        )
        try:
            ground_truth = world.task.ground_truth

            required_apps = set(ground_truth.required_apps)
            difficulty = int(ground_truth.metadata["difficulty"])
            rejected = False

            if required_apps != {"spotify"}:
                failures.append(
                    f"{task_id}: required_apps={sorted(required_apps)}"
                )
                rejected = True

            if difficulty not in {1, 2}:
                failures.append(f"{task_id}: difficulty={difficulty}")
                rejected = True

            if rejected:
                continue

            tasks.append((task_id, task_id_to_generator_id(task_id)))
            difficulty_by_task[task_id] = difficulty
        finally:
            world.close()

    if failures:
        raise RuntimeError("\n".join(failures))
    return tasks, difficulty_by_task


def main(argv: Sequence[str] | None = None) -> int:
    args_list = list(sys.argv[1:] if argv is None else argv)
    if not args_list:
        return 2

    parser = argparse.ArgumentParser(
        description="Build the fixed Spotify capability task set from a local id list."
    )
    parser.add_argument("--task-ids", required=True)
    parser.add_argument("--manifest", required=True)
    try:
        args = parser.parse_args(args_list)
    except SystemExit as error:
        code = error.code
        return 2 if code is None else int(code)

    task_ids_path = Path(args.task_ids)
    manifest_path = Path(args.manifest)
    _require_ignored(task_ids_path)
    _require_ignored(manifest_path)
    task_ids = _load_task_ids(task_ids_path)

    tasks, difficulty_by_task = _check_tasks(task_ids)

    canonical = canonical_task_set_bytes(
        appworld_version=APPWORLD_VERSION,
        split="train",
        selection_rule=SELECTION_RULE,
        selection_seed=SELECTION_SEED,
        tasks=tasks,
    )

    digest = task_set_hash_from_bytes(canonical)

    local_manifest = {
        "appworld_version": APPWORLD_VERSION,
        "split": "train",
        "selection_rule": SELECTION_RULE,
        "selection_seed": SELECTION_SEED,
        "task_count": len(tasks),
        "scenario_count": len(set(s for _, s in tasks)),
        "task_set_hash": digest,
        "task_ids": [t for t, _ in tasks],
        "scenario_ids": [s for _, s in tasks],
        "difficulty_by_task": difficulty_by_task,
    }

    public_config = {
        key: local_manifest[key]
        for key in (
            "appworld_version",
            "split",
            "selection_rule",
            "selection_seed",
            "task_count",
            "scenario_count",
            "task_set_hash",
        )
    }

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    PUBLIC_PATH.parent.mkdir(parents=True, exist_ok=True)

    manifest_path.write_text(
        json.dumps(local_manifest, indent=2) + "\n",
        encoding="utf-8",
    )

    PUBLIC_PATH.write_text(
        json.dumps(public_config, indent=2) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(public_config, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
