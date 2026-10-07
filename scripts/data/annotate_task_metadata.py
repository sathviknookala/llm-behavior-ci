"""Add difficulty and required-app labels to a local task-set manifest.

Requires --manifest, a git-ignored local manifest (task ids stay local,
docs/DATA.md). Opens each task once through AppWorld and writes
``difficulty_by_task`` and ``required_apps_by_task`` back into the same
file; the task set and its hash are unchanged and re-verified. Refuses
``test_normal`` and ``test_challenge``: the final-split labels are an
open gate, not part of this patch. Labels come from AppWorld task metadata,
never from an evaluator outcome. Bare invocation exits 2.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

from llm_behavior_ci.experiments.run_config import RunConfigError, load_local_task_manifest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_OPEN_SPLITS = frozenset({"train", "dev"})

Reader = Callable[[str], tuple[int, tuple[str, ...]]]


def _is_ignored(path: Path) -> bool:
    completed = subprocess.run(
        ["git", "check-ignore", "-q", str(path.resolve())],
        cwd=_REPO_ROOT,
        capture_output=True,
        check=False,
    )
    return completed.returncode == 0


def appworld_reader(task_id: str) -> tuple[int, tuple[str, ...]]:
    from appworld import AppWorld

    world = AppWorld(task_id=task_id, ground_truth_mode="full")
    try:
        ground_truth = world.task.ground_truth
        return (
            int(ground_truth.metadata["difficulty"]),
            tuple(sorted(str(app) for app in ground_truth.required_apps)),
        )
    finally:
        world.close()


def annotate(path: Path, reader: Reader) -> int:
    manifest = load_local_task_manifest(path)
    if manifest.task_set.split not in _OPEN_SPLITS:
        raise RunConfigError("task metadata annotation is closed on this split")
    payload = json.loads(path.read_text(encoding="utf-8"))
    difficulty: dict[str, int] = {}
    apps: dict[str, list[str]] = {}
    for task_id in manifest.task_set.task_ids:
        level, required = reader(task_id)
        difficulty[task_id] = level
        apps[task_id] = list(required)
    payload["difficulty_by_task"] = difficulty
    payload["required_apps_by_task"] = apps
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)
    load_local_task_manifest(path)
    return len(difficulty)


def main(
    argv: Sequence[str] | None = None,
    *,
    reader: Reader | None = None,
    require_ignored: bool = True,
) -> int:
    args_list = list(sys.argv[1:] if argv is None else argv)
    if not args_list:
        return 2
    parser = argparse.ArgumentParser(description="Annotate a local task manifest.")
    parser.add_argument("--manifest", required=True)
    try:
        args = parser.parse_args(args_list)
    except SystemExit as error:
        return 2 if error.code is None else int(error.code)
    path = Path(args.manifest)
    if require_ignored and not _is_ignored(path):
        print(f"manifest must be ignored by git: {path}", file=sys.stderr)
        return 2
    try:
        count = annotate(path, reader or appworld_reader)
    except (RunConfigError, OSError, KeyError, ValueError) as error:
        print(str(error) or "annotation failed", file=sys.stderr)
        return 1
    print(f"annotated {count} tasks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
