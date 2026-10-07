"""Build the short-horizon Spotify train task set.

Reads AppWorld train ground truth only. A task is eligible when its required
apps are exactly Spotify. Difficulty and requirement count come from
metadata and the length of the evaluator test list. Reference-call shape
comes from the call log's paths and the presence of a page index. Requirement
text, instructions, and solution source are not read.

Writes ``configs/tasks/train_spotify_capability_short.json`` and the local
manifest only when ``spotify_capability_short_v1`` can fill 20 tasks.
``--diagnostic`` instead writes the 6-task native shape-limited set
``spotify_short_horizon_diagnostic_v1`` and does not widen the 20-task
rule. The public file is the hash, counts, split, rule, seed, and setup
profile. Resolved task ids stay in the gitignored manifest. When the
requested pool is unavailable, prints a count-only audit and writes nothing.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

from llm_behavior_ci.tasks.short_horizon import (
    DIAGNOSTIC_RULE,
    SELECTION_RULE,
    SELECTION_SEED,
    SPLIT,
    ShortHorizonCandidate,
    ShortHorizonSelection,
    ShortHorizonSelectionError,
    reference_shape,
    select_spotify_short_diagnostic,
    select_spotify_short_set,
)

APPWORLD_VERSION = "0.1.3.post1"
SETUP_PROFILE = "spotify_authenticated_v1"
PUBLIC_PATH = Path("configs/tasks/train_spotify_capability_short.json")
DIAGNOSTIC_PUBLIC_PATH = Path(
    "configs/tasks/train_spotify_short_horizon_diagnostic_6.json"
)
_REPO_ROOT = Path(__file__).resolve().parents[2]
_PUBLIC_KEYS = (
    "appworld_version",
    "split",
    "selection_rule",
    "selection_seed",
    "task_count",
    "scenario_count",
    "task_set_hash",
    "appworld_setup_profile",
)


def _require_ignored(path: Path, repository_root: Path) -> None:
    completed = subprocess.run(
        ["git", "check-ignore", "-q", str(path.resolve())],
        cwd=repository_root,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        print(f"path must be ignored by git: {path}", file=sys.stderr)
        raise SystemExit(2)


def _require_appworld_root() -> None:
    root = os.environ.get("APPWORLD_ROOT")
    if root is None or root.strip() == "" or not Path(root).is_dir():
        print("APPWORLD_ROOT must be set to an existing directory", file=sys.stderr)
        raise SystemExit(2)


def _has_page_index(body: object) -> bool:
    return isinstance(body, dict) and "page_index" in body


def _call_shape(calls: object) -> tuple[int, int]:
    if not isinstance(calls, list):
        raise ShortHorizonSelectionError("reference call log is unreadable")
    pairs: list[tuple[str, bool]] = []
    for call in calls:
        if not isinstance(call, dict):
            raise ShortHorizonSelectionError("reference call log is unreadable")
        url = call.get("url")
        if not isinstance(url, str):
            raise ShortHorizonSelectionError("reference call log is unreadable")
        pairs.append((url, _has_page_index(call.get("data"))))
    return reference_shape(pairs)


def load_train_spotify_candidates() -> tuple[ShortHorizonCandidate, ...]:
    from appworld.common.path_store import path_store
    from appworld.common.utils import read_json
    from appworld.task import load_task_ids, task_id_to_generator_id

    root = Path(path_store.data)
    candidates: list[ShortHorizonCandidate] = []
    for task_id in load_task_ids(SPLIT):
        ground_truth = root / "tasks" / task_id / "ground_truth"
        try:
            raw_apps = read_json(ground_truth / "required_apps.json")
        except (OSError, ValueError, TypeError) as error:
            raise ShortHorizonSelectionError(
                "spotify train candidate is missing ground-truth counts"
            ) from error
        if not isinstance(raw_apps, list):
            raise ShortHorizonSelectionError(
                "spotify train candidate is missing ground-truth counts"
            )
        if set(raw_apps) != {"spotify"}:
            continue
        try:
            metadata = read_json(ground_truth / "metadata.json")
            tests = read_json(ground_truth / "test_data.json")
            calls = read_json(ground_truth / "api_calls.json")
        except (OSError, ValueError, TypeError) as error:
            raise ShortHorizonSelectionError(
                "spotify train candidate is missing ground-truth counts"
            ) from error
        if not isinstance(metadata, dict) or not isinstance(tests, list):
            raise ShortHorizonSelectionError(
                "spotify train candidate is missing ground-truth counts"
            )
        agent_calls, paged_calls = _call_shape(calls)
        try:
            difficulty = metadata["difficulty"]
        except KeyError as error:
            raise ShortHorizonSelectionError(
                "spotify train candidate is missing ground-truth counts"
            ) from error
        candidates.append(
            ShortHorizonCandidate(
                task_id=task_id,
                scenario_id=task_id_to_generator_id(task_id),
                difficulty=difficulty,
                requirement_count=len(tests),
                reference_paged_calls=paged_calls,
                reference_agent_calls=agent_calls,
            )
        )
    return tuple(candidates)


def public_task_document(selection: ShortHorizonSelection) -> dict[str, object]:
    task_set = selection.task_set
    return {
        "appworld_version": task_set.appworld_version,
        "split": task_set.split,
        "selection_rule": task_set.selection_rule,
        "selection_seed": task_set.selection_seed,
        "task_count": task_set.task_count,
        "scenario_count": task_set.scenario_count,
        "task_set_hash": task_set.task_set_hash,
        "appworld_setup_profile": SETUP_PROFILE,
    }


def local_manifest(selection: ShortHorizonSelection) -> dict[str, object]:
    document = public_task_document(selection)
    by_id = {candidate.task_id: candidate for candidate in selection.candidates}
    document["selection_tier"] = selection.tier
    document["task_ids"] = list(selection.task_set.task_ids)
    document["scenario_ids"] = list(selection.task_set.scenario_ids)
    document["difficulty_by_task"] = {
        task_id: by_id[task_id].difficulty for task_id in selection.task_set.task_ids
    }
    document["requirement_count_by_task"] = {
        task_id: by_id[task_id].requirement_count
        for task_id in selection.task_set.task_ids
    }
    document["reference_paged_calls_by_task"] = {
        task_id: by_id[task_id].reference_paged_calls
        for task_id in selection.task_set.task_ids
    }
    document["reference_agent_calls_by_task"] = {
        task_id: by_id[task_id].reference_agent_calls
        for task_id in selection.task_set.task_ids
    }
    return document


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main(
    argv: Sequence[str] | None = None,
    loader: Callable[[], tuple[ShortHorizonCandidate, ...]] | None = None,
    *,
    repository_root: Path | None = None,
) -> int:
    """Select the set and write the ignored manifest and the public document.

    ``repository_root`` is the git work tree whose ignore rules the
    manifest path must satisfy; it defaults to this repository.
    """

    args_list = list(sys.argv[1:] if argv is None else argv)
    if not args_list:
        return 2
    parser = argparse.ArgumentParser(
        description=(
            "Select the fixed short-horizon Spotify train task set. "
            "Writes nothing when the pool cannot fill 20 tasks."
        )
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--public", default=None)
    parser.add_argument("--diagnostic", action="store_true")
    try:
        args = parser.parse_args(args_list)
    except SystemExit as error:
        code = error.code
        return 2 if code is None else int(code)

    manifest_path = Path(args.manifest)
    public_default = DIAGNOSTIC_PUBLIC_PATH if args.diagnostic else PUBLIC_PATH
    public_path = Path(args.public) if args.public is not None else public_default
    _require_ignored(
        manifest_path,
        _REPO_ROOT if repository_root is None else Path(repository_root),
    )
    if loader is None:
        _require_appworld_root()
        source = load_train_spotify_candidates
    else:
        source = loader
    try:
        candidates = source()
        if args.diagnostic:
            selection = select_spotify_short_diagnostic(
                candidates,
                appworld_version=APPWORLD_VERSION,
                split=SPLIT,
                selection_rule=DIAGNOSTIC_RULE,
                selection_seed=SELECTION_SEED,
            )
        else:
            selection = select_spotify_short_set(
                candidates,
                appworld_version=APPWORLD_VERSION,
                split=SPLIT,
                selection_rule=SELECTION_RULE,
                selection_seed=SELECTION_SEED,
            )
    except ShortHorizonSelectionError as error:
        if not error.audit:
            print(str(error), file=sys.stderr)
            return 1
        if args.diagnostic:
            print("short-horizon diagnostic pool is not 6 tasks", file=sys.stderr)
        else:
            print("short-horizon pool cannot fill 20 tasks", file=sys.stderr)
        print(json.dumps(error.audit, indent=2))
        return 3

    public = public_task_document(selection)
    if tuple(public) != _PUBLIC_KEYS:
        print("public task document has an unexpected field set", file=sys.stderr)
        return 2
    _write_json(manifest_path, local_manifest(selection))
    _write_json(public_path, public)
    print(json.dumps(public, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
