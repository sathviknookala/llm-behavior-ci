"""Build the fixed Spotify capability task set.

Opens each approved train task, refuses the set unless every task's
required apps are exactly Spotify and its difficulty is 1 or 2, then
writes a local manifest and the public task metadata. The local manifest
contains resolved task ids and stays out of git. The public file is the
hash, counts, split, rule, and seed only.
"""

from __future__ import annotations

import json
from pathlib import Path

from appworld import AppWorld
from appworld.task import task_id_to_generator_id

from llm_behavior_ci.tasks.selection import (
    canonical_task_set_bytes,
    task_set_hash_from_bytes,
)

APPWORLD_VERSION = "0.1.3.post1"
SELECTION_RULE = "fixed_spotify_capability"
SELECTION_SEED = 17

TASK_IDS = (
    "82e2fac_1",
    "82e2fac_2",
    "82e2fac_3",
    "287e338_1",
    "287e338_2",
    "287e338_3",
    "aa8502b_1",
    "aa8502b_2",
    "c901732_1",
    "c901732_2",
    "ccb4494_1",
    "ccb4494_2",
    "ccb4494_3",
    "e3d6c94_1",
    "e3d6c94_2",
    "e3d6c94_3",
    "692c77d_1",
    "692c77d_2",
    "ce359b5_1",
    "ce359b5_2",
)


def main() -> None:
    tasks: list[tuple[str, str]] = []
    difficulty_by_task: dict[str, int] = {}
    failures: list[str] = []

    for task_id in TASK_IDS:
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

            scenario_id = task_id_to_generator_id(task_id)

            tasks.append((task_id, scenario_id))
            difficulty_by_task[task_id] = difficulty
        finally:
            world.close()

    if failures:
        raise RuntimeError("\n".join(failures))

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

    Path("data/manifests").mkdir(parents=True, exist_ok=True)
    Path("configs/tasks").mkdir(parents=True, exist_ok=True)

    Path(
        "data/manifests/spotify_capability_20.json"
    ).write_text(
        json.dumps(local_manifest, indent=2) + "\n",
        encoding="utf-8",
    )

    Path(
        "configs/tasks/train_spotify_capability.json"
    ).write_text(
        json.dumps(public_config, indent=2) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(public_config, indent=2))


if __name__ == "__main__":
    main()
