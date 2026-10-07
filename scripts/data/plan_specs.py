"""Author and validate local task plan specs for the Tier 1 plan-quality gate.

``skeleton --task-set MANIFEST --output PATH`` writes one entry per task
with ``available_tools`` filled from the task's AppWorld API surface (its
required apps plus ``supervisor``) and empty ``subgoal_keywords``,
``required_entities``, and ``dependency_pairs`` for a human to fill in.
It never reads the task instruction into the file and never reads an
evaluator outcome. Both paths must be git-ignored, since entries name
tasks.

``validate --specs PATH --task-set MANIFEST [--features NAME ...]`` loads
the specs, checks that every task in the set has exactly one spec and no
spec names another task, and runs the gate's semantic-coverage check for
the requested features. It prints counts only.

Bare invocation exits 2.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

from llm_behavior_ci.experiments.run_config import RunConfigError, load_local_task_manifest
from llm_behavior_ci.lifecycle.plan_features import PlanFeatureError, require_semantic_coverage
from llm_behavior_ci.tasks.plan_specs import (
    PlanSpecError,
    index_by_task_id,
    load_task_plan_specs,
    task_plan_spec_from_mapping,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_INFRASTRUCTURE = frozenset({"admin", "api_docs"})

ToolReader = Callable[[str, tuple[str, ...] | None], tuple[str, ...]]


def _is_ignored(path: Path) -> bool:
    completed = subprocess.run(
        ["git", "check-ignore", "-q", str(path.resolve())],
        cwd=_REPO_ROOT,
        capture_output=True,
        check=False,
    )
    return completed.returncode == 0


def appworld_tools(task_id: str, required_apps: tuple[str, ...] | None) -> tuple[str, ...]:
    from appworld import AppWorld

    world = AppWorld(task_id=task_id)
    try:
        documentation = getattr(world.task, "api_docs", {})
        apps = (
            set(required_apps) | {"supervisor"}
            if required_apps is not None
            else {str(app) for app in documentation} - _INFRASTRUCTURE
        )
        return tuple(
            sorted(
                f"{app}.{api}"
                for app, apis in documentation.items()
                if str(app) in apps
                for api in apis
            )
        )
    finally:
        world.close()


def skeleton(manifest_path: Path, output: Path, reader: ToolReader) -> int:
    manifest = load_local_task_manifest(manifest_path)
    entries = []
    for task_id in manifest.task_set.task_ids:
        tools = reader(task_id, manifest.required_apps(task_id))
        entry = {
            "task_id": task_id,
            "available_tools": list(tools),
            "subgoal_keywords": [],
            "required_entities": [],
            "dependency_pairs": [],
        }
        task_plan_spec_from_mapping(entry)
        entries.append(entry)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps({"entries": entries}, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return len(entries)


def validate(specs_path: Path, manifest_path: Path, features: Sequence[str]) -> dict[str, int]:
    manifest = load_local_task_manifest(manifest_path)
    specs = index_by_task_id(load_task_plan_specs(specs_path))
    task_ids = manifest.task_set.task_ids
    extra = [name for name in specs if name not in set(task_ids)]
    if extra:
        raise PlanSpecError(f"{len(extra)} specs name a task outside the task set")
    missing = [task_id for task_id in task_ids if task_id not in specs]
    if missing:
        raise PlanSpecError(f"{len(missing)} of {len(task_ids)} tasks have no spec")
    require_semantic_coverage(task_ids, specs, tuple(features))
    return {
        "tasks": len(task_ids),
        "with_subgoals": sum(1 for spec in specs.values() if spec.subgoal_keywords),
        "with_entities": sum(1 for spec in specs.values() if spec.required_entities),
        "with_dependencies": sum(1 for spec in specs.values() if spec.dependency_pairs),
    }


def main(
    argv: Sequence[str] | None = None,
    *,
    reader: ToolReader | None = None,
    require_ignored: bool = True,
) -> int:
    args_list = list(sys.argv[1:] if argv is None else argv)
    if not args_list:
        return 2
    parser = argparse.ArgumentParser(description="Author and validate plan specs.")
    commands = parser.add_subparsers(dest="command", required=True)
    make = commands.add_parser("skeleton")
    make.add_argument("--task-set", required=True)
    make.add_argument("--output", required=True)
    check = commands.add_parser("validate")
    check.add_argument("--specs", required=True)
    check.add_argument("--task-set", required=True)
    check.add_argument("--features", nargs="*", default=())
    try:
        args = parser.parse_args(args_list)
    except SystemExit as error:
        return 2 if error.code is None else int(error.code)
    try:
        if args.command == "skeleton":
            output = Path(args.output)
            if require_ignored and not _is_ignored(output):
                print(f"output must be ignored by git: {output}", file=sys.stderr)
                return 2
            count = skeleton(Path(args.task_set), output, reader or appworld_tools)
            print(f"wrote {count} plan spec skeletons")
            return 0
        summary = validate(Path(args.specs), Path(args.task_set), args.features)
    except (RunConfigError, PlanSpecError, PlanFeatureError, OSError) as error:
        print(str(error) or "plan spec command failed", file=sys.stderr)
        return 1
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
