"""Prepare or run a bounded, resumable hosted calibration on train or dev.

Requires --configuration, --task-set, --repetitions, --max-model-episodes,
--confidence-level, --resamples, --seed, --checkpoint, and --report, plus
either --prepare or --run. --modes defaults to execute,plan. --inventory-root,
repeated --store, and repeated --capture supply existing episodes. A scored
observation fills a slot only when its configuration hash matches. Another
hash stays inventory.

--prepare writes the checkpoint and a public report and does not call a
provider. --run resumes that checkpoint, executes at most
--max-model-episodes model episodes, and checkpoints after each slot.
Do-nothing episodes are not model episodes. An A/A pair is not started
unless both model episodes fit. The repetition count is a provisional
execution budget, not a power-based sample target. test_normal is refused.
Bare invocation exits 2.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from llm_behavior_ci.config import ConfigError, RunConfiguration, new_run_identity
from llm_behavior_ci.experiments.calibration import (
    CalibrationError,
    InventoryCounts,
    apply_executed,
    bind_observations,
    build_slots,
    calibration_report,
    checkpoint_identity,
    discover_sqlite_stores,
    load_checkpoint,
    observation_from_episode_dict,
    occupied_from_checkpoint,
    read_capture_file,
    read_sqlite_store,
    select_batch,
    sources_from_checkpoint,
    write_checkpoint,
)
from llm_behavior_ci.experiments.run_config import RunConfigError, load_local_task_manifest
from llm_behavior_ci.runtime.provenance import (
    ProvenanceError,
    enforce_committed_provenance,
    read_repository_state,
    relevant_dirty_paths,
)
from llm_behavior_ci.usage import UsageError, load_pricing


def main(argv: Sequence[str] | None = None) -> int:
    args_list = list(sys.argv[1:] if argv is None else argv)
    if not args_list:
        return 2
    parser = argparse.ArgumentParser(description="Prepare or run hosted calibration.")
    parser.add_argument("--configuration", required=True)
    parser.add_argument("--task-set", required=True)
    parser.add_argument("--repetitions", required=True, type=int)
    parser.add_argument("--max-model-episodes", required=True, type=int)
    parser.add_argument("--confidence-level", required=True, type=float)
    parser.add_argument("--resamples", required=True, type=int)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--modes", default="execute,plan")
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--inventory-root", default=None)
    parser.add_argument("--store", action="append", default=[])
    parser.add_argument("--capture", action="append", default=[])
    parser.add_argument("--episode-store", default=None)
    parser.add_argument("--pricing", default=None)
    parser.add_argument("--endpoint", default=None)
    try:
        args = parser.parse_args(args_list)
    except SystemExit as error:
        return 2 if error.code is None else int(error.code)
    if args.prepare == args.run:
        print("pass exactly one of --prepare or --run", file=sys.stderr)
        return 2
    try:
        configuration = RunConfiguration.from_dict(
            json.loads(Path(args.configuration).read_text(encoding="utf-8"))
        )
        manifest = load_local_task_manifest(Path(args.task_set))
        if manifest.task_set.task_set_hash != configuration.task.task_set_hash:
            raise CalibrationError("task set does not match the configuration")
        if manifest.task_set.split != configuration.task.split:
            raise CalibrationError("task set split does not match the configuration")
        modes = tuple(part.strip() for part in str(args.modes).split(",") if part.strip())
        checkpoint_path = Path(args.checkpoint)
        report_path = Path(args.report)
        _refuse_results(checkpoint_path)
        _refuse_results(report_path)
        identity = checkpoint_identity(
            configuration,
            manifest.task_set,
            repetitions=args.repetitions,
            aa_modes=modes,
        )
        observations, counts = _inventory(args)
        occupied: dict = {}
        occupied_sources: dict = {}
        if checkpoint_path.is_file():
            payload = load_checkpoint(checkpoint_path)
            occupied = occupied_from_checkpoint(payload, identity)
            occupied_sources = sources_from_checkpoint(payload)
        slots = build_slots(manifest.task_set, repetitions=args.repetitions, aa_modes=modes)
        bound = bind_observations(
            slots,
            observations,
            configuration_hash=str(identity["configuration_hash"]),
            task_scenarios={
                task_id: str(scenario_id)
                for task_id, scenario_id in zip(
                    manifest.task_set.task_ids,
                    manifest.task_set.scenario_ids,
                    strict=True,
                )
            },
            occupied=occupied,
            occupied_sources=occupied_sources,
        )
        provenance = _provenance(configuration)
        if args.run:
            if not provenance["run_allowed"]:
                raise ProvenanceError(
                    "configured git_commit does not match HEAD or the tree is dirty"
                )
            _require_appworld_root()
            bound = _run_batch(
                bound,
                configuration,
                checkpoint_path=checkpoint_path,
                identity=identity,
                max_model_episodes=args.max_model_episodes,
                endpoint=args.endpoint,
                episode_store=None if args.episode_store is None else Path(args.episode_store),
            )
        pricing = None if args.pricing is None else load_pricing(Path(args.pricing))
        report = calibration_report(
            configuration=configuration,
            task_set=manifest.task_set,
            observations=_with_bound(observations, bound),
            bound=bound,
            counts=counts,
            repetitions=args.repetitions,
            confidence_level=args.confidence_level,
            resamples=args.resamples,
            seed=args.seed,
            max_model_episodes=args.max_model_episodes,
            pricing=pricing,
            provenance=provenance,
        )
        write_checkpoint(checkpoint_path, identity=identity, bound=bound)
        _atomic_report(report_path, report)
    except (
        CalibrationError,
        ConfigError,
        RunConfigError,
        ProvenanceError,
        UsageError,
        OSError,
        json.JSONDecodeError,
    ) as error:
        print(str(error) or "calibration failed", file=sys.stderr)
        return 1
    print(json.dumps(_printed(report), sort_keys=True))
    return 0


def _inventory(args: argparse.Namespace) -> tuple[tuple, InventoryCounts]:
    paths: list[Path] = []
    if args.inventory_root is not None:
        paths.extend(discover_sqlite_stores(Path(args.inventory_root)))
    paths.extend(Path(item) for item in args.store)
    observations = []
    totals = InventoryCounts()
    seen: set[Path] = set()
    for path in paths:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        found, counts = read_sqlite_store(path)
        observations.extend(found)
        totals = _add_counts(totals, counts)
    for capture in args.capture:
        print(f"inventory_capture: {Path(capture).name}", file=sys.stderr, flush=True)
        found, counts = read_capture_file(
            Path(capture),
            on_progress=lambda count: print(
                f"inventory_records: {count}", file=sys.stderr, flush=True
            ),
        )
        observations.extend(found)
        totals = _add_counts(totals, counts)
    return tuple(observations), totals


def _add_counts(left: InventoryCounts, right: InventoryCounts) -> InventoryCounts:
    return InventoryCounts(
        stores_read=left.stores_read + right.stores_read,
        captures_read=left.captures_read + right.captures_read,
        open_episodes=left.open_episodes + right.open_episodes,
        unreadable_episodes=left.unreadable_episodes + right.unreadable_episodes,
        excluded_closed_split=left.excluded_closed_split + right.excluded_closed_split,
        scenario_conflicts=left.scenario_conflicts + right.scenario_conflicts,
    )


def _provenance(configuration: RunConfiguration) -> dict[str, bool]:
    state = read_repository_state()
    dirty = bool(relevant_dirty_paths(state.dirty_paths))
    head_matches = state.head == configuration.git_commit
    return {
        "head_matches": head_matches,
        "relevant_dirty": dirty,
        "run_allowed": head_matches and not dirty,
    }


def _require_appworld_root() -> None:
    import os

    root = os.environ.get("APPWORLD_ROOT", "").strip()
    if root == "" or not Path(root).is_dir():
        raise CalibrationError("APPWORLD_ROOT must be an existing directory")


def _run_batch(
    bound,
    configuration: RunConfiguration,
    *,
    checkpoint_path: Path,
    identity: dict[str, object],
    max_model_episodes: int,
    endpoint: str | None,
    episode_store: Path | None,
):
    from llm_behavior_ci.runtime.episode import (
        EpisodeRejected,
        RuntimeUnavailable,
        run_do_nothing_episode,
        run_episode,
        run_pair,
    )
    from llm_behavior_ci.runtime.factory import LiveRuntimeFactory, RuntimeFactoryError
    from llm_behavior_ci.storage import EpisodeStore

    enforce_committed_provenance(configuration)
    factory = LiveRuntimeFactory.from_endpoints(
        production=endpoint, reference=endpoint, candidate=endpoint
    )
    try:
        factory.preflight(
            {
                "production": configuration,
                "reference": configuration,
                "candidate": configuration,
            },
            require_distinct_endpoints=False,
        )
    except RuntimeFactoryError as error:
        raise CalibrationError(str(error)) from error
    store = None
    if episode_store is not None:
        _refuse_results(episode_store)
        episode_store.parent.mkdir(parents=True, exist_ok=True)
        store = EpisodeStore(episode_store)
    batch = select_batch(bound.slots, bound.filled, max_model_episodes=max_model_episodes)
    try:
        for slot in batch:
            try:
                episodes = _execute_slot(
                    slot,
                    configuration,
                    factory,
                    run_episode,
                    run_do_nothing_episode,
                    run_pair,
                )
            except (EpisodeRejected, RuntimeUnavailable, RuntimeFactoryError) as error:
                raise CalibrationError(str(error)) from error
            if store is not None:
                for episode in episodes:
                    _record_episode(store, episode)
            observations = tuple(_observation_for_slot(episode, slot) for episode in episodes)
            bound = apply_executed(bound, slot, observations)
            write_checkpoint(checkpoint_path, identity=identity, bound=bound)
    finally:
        if store is not None:
            store.close()
    return bound


def _execute_slot(slot, configuration, factory, run_episode, run_do_nothing_episode, run_pair):
    if slot.kind == "baseline_do_nothing":
        runtime = factory(configuration, mode="execute", role="production")
        return (
            run_do_nothing_episode(
                slot.task_id,
                configuration,
                run=new_run_identity(configuration),
                runtime=runtime,
                scenario_id=slot.scenario_id,
            ),
        )
    if slot.kind == "baseline_production":
        runtime = factory(configuration, mode="execute", role="production")
        return (
            run_episode(
                slot.task_id,
                configuration,
                "execute",
                run=new_run_identity(configuration),
                runtime=runtime,
                scenario_id=slot.scenario_id,
            ),
        )
    mode = "plan" if slot.kind == "aa_plan" else "execute"
    reference = factory(configuration, mode=mode, role="reference")
    candidate = factory(configuration, mode=mode, role="candidate")
    pair = run_pair(
        slot.task_id,
        configuration,
        configuration,
        reference_run=new_run_identity(configuration),
        candidate_run=new_run_identity(configuration),
        runtime=reference,
        candidate_runtime=candidate,
        mode=mode,
        scenario_id=slot.scenario_id,
    )
    return pair.reference, pair.candidate


def _observation_for_slot(episode, slot):
    observation = observation_from_episode_dict(
        episode.to_dict(),
        source_name="calibration",
        repetition=slot.repetition,
    )
    if slot.kind == "baseline_do_nothing":
        observation = replace_role(observation, "do_nothing", slot.scenario_id)
    elif slot.kind == "baseline_production":
        observation = replace_role(observation, "production", slot.scenario_id)
    else:
        observation = replace_role(observation, observation.role, slot.scenario_id)
    return observation


def replace_role(observation, role: str, scenario_id: str):
    from dataclasses import replace

    return replace(observation, role=role, scenario_id=scenario_id)


def _record_episode(store, episode) -> None:
    store.start_episode(episode.episode, episode.run, episode.task.task_id)
    steps = sorted((*episode.model_steps, *episode.tool_steps), key=lambda step: step.index)
    for step in steps:
        store.append_step(episode.episode.episode_id, step)
    store.finish_episode(episode)


def _with_bound(observations: tuple, bound) -> tuple:
    seen = {item.observation_id for item in observations}
    extra = []
    for group in bound.filled.values():
        for item in group:
            if item.observation_id not in seen:
                extra.append(item)
                seen.add(item.observation_id)
    return tuple(observations) + tuple(extra)


def _printed(report: dict[str, object]) -> dict[str, object]:
    completion = report["completion"]
    cost = report["cost"]
    batch = report["next_batch"]
    provenance = report["provenance"]
    assert isinstance(completion, dict)
    assert isinstance(cost, dict)
    assert isinstance(batch, dict)
    assert isinstance(provenance, dict)
    overlap = report["reference_baseline_overlap"]
    preserved = []
    if isinstance(overlap, list):
        for item in overlap:
            if isinstance(item, dict) and item.get("fills_target_slots") is False:
                preserved.append(
                    {
                        "configuration_hash": item.get("configuration_hash"),
                        "healthy_execute_aa_references": item.get("healthy_execute_aa_references"),
                        "overlapping_episodes_if_counted_in_both": item.get(
                            "overlapping_episodes_if_counted_in_both"
                        ),
                        "supports_baseline_success": item.get("supports_baseline_success"),
                    }
                )
    return {
        "record": report["record"],
        "split": report["split"],
        "configuration_hash": report["configuration_hash"],
        "task_set_hash": report["task_set_hash"],
        "qualification_complete": report["qualification_complete"],
        "sampling_status": report["sampling"]["allocation_status"]
        if isinstance(report["sampling"], dict)
        else None,
        "model_episodes_remaining": completion["model_episodes_remaining"],
        "model_episodes_filled": completion["model_episodes_filled"],
        "completion_fraction": completion["fraction"],
        "next_batch_model_episodes": batch["model_episodes"],
        "cost_status": cost["status"],
        "next_batch_estimated": cost["next_batch_estimated"],
        "remaining_execute_estimated": cost["remaining_execute_estimated"],
        "run_allowed": provenance["run_allowed"],
        "preserved_overlap": preserved,
    }


def _atomic_report(path: Path, report: dict[str, object]) -> None:
    _refuse_results(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
    temporary.replace(path)


def _refuse_results(path: Path) -> None:
    if any(parent.name == "results" for parent in (path, *path.parents)):
        raise CalibrationError("calibration output cannot be written as a public result")


if __name__ == "__main__":
    raise SystemExit(main())
