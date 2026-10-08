"""Resumable hosted calibration on a fixed train or dev task set.

The repetition count is a caller-supplied execution budget. It is not a
power-based sample target and not a protocol threshold. A partial sample
stays preliminary. An arm is qualified only when every one of its slots
has a scored observation. ``test_normal`` and ``test_challenge`` are refused.

One episode fills one slot of the configuration hash that produced it.
A same-hash execute pair fills an A/A slot. Its reference episode is a
healthy production draw and can support a baseline success estimate, but
the exclusive assignment does not put that episode in both analyses.
Counting it in both is an overlap of that many episodes. A different
configuration hash is inventory only and is never relabeled onto the
target.

Cost is priced only from a caller-supplied table. Cached input is a subset
of input tokens. Reasoning tokens are not added on top of output tokens.
A projection from another configuration names that hash and is not a
measured bill for the target.
"""

from __future__ import annotations

import json
import sqlite3
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from statistics import fmean, stdev

from llm_behavior_ci.config import (
    RunConfiguration,
    hosted_provider,
    run_configuration_hash,
)
from llm_behavior_ci.records import assert_public_payload
from llm_behavior_ci.stats.bootstrap import clustered_paired_bootstrap
from llm_behavior_ci.tasks.selection import TaskSet
from llm_behavior_ci.usage import PricingTable, UsageError, token_cost

CHECKPOINT_VERSION = "calibration-checkpoint-v1"
_OPEN_SPLITS = frozenset({"train", "dev"})
_CLOSED_SPLITS = frozenset({"test_normal", "test_challenge"})
_INFRASTRUCTURE = frozenset({"runtime_error", "timeout", "cancelled"})
_BASELINE_ROLES = frozenset({"unpaired", "reference", "production"})
_AA_KINDS = frozenset({"aa_execute", "aa_plan"})
_PILOT_EVIDENCE = (
    "production_outcomes_on_at_least_two_scenarios",
    "repeated_production_draws_of_the_same_tasks",
    "aligned_do_nothing_outcomes",
    "caller_supplied_harm_margin_and_alpha",
)


class CalibrationError(ValueError):
    pass


@dataclass(frozen=True)
class ScoredObservation:
    observation_id: str
    configuration_hash: str
    task_set_hash: str | None
    split: str
    task_id: str
    scenario_id: str | None
    mode: str
    role: str
    repetition: int | None
    pair_id: str | None
    success: bool | None
    requirement_fraction: float | None
    termination_reason: str
    latency_seconds: float
    input_tokens: int | None
    output_tokens: int | None
    cache_read_tokens: int | None
    reasoning_tokens: int | None
    request_count: int
    provider: str | None
    model_id: str | None
    source_name: str

    def to_dict(self) -> dict[str, object]:
        return {
            "observation_id": self.observation_id,
            "configuration_hash": self.configuration_hash,
            "task_set_hash": self.task_set_hash,
            "split": self.split,
            "task_id": self.task_id,
            "scenario_id": self.scenario_id,
            "mode": self.mode,
            "role": self.role,
            "repetition": self.repetition,
            "pair_id": self.pair_id,
            "success": self.success,
            "requirement_fraction": self.requirement_fraction,
            "termination_reason": self.termination_reason,
            "latency_seconds": self.latency_seconds,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_read_tokens": self.cache_read_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "request_count": self.request_count,
            "provider": self.provider,
            "model_id": self.model_id,
            "source_name": self.source_name,
        }

    @classmethod
    def from_dict(cls, payload: object) -> ScoredObservation:
        if not isinstance(payload, Mapping):
            raise CalibrationError("observation must be an object")
        try:
            return cls(
                observation_id=str(payload["observation_id"]),
                configuration_hash=str(payload["configuration_hash"]),
                task_set_hash=(
                    None
                    if payload.get("task_set_hash") is None
                    else str(payload["task_set_hash"])
                ),
                split=str(payload["split"]),
                task_id=str(payload["task_id"]),
                scenario_id=(
                    None
                    if payload.get("scenario_id") is None
                    else str(payload["scenario_id"])
                ),
                mode=str(payload["mode"]),
                role=str(payload["role"]),
                repetition=(
                    None
                    if payload.get("repetition") is None
                    else int(payload["repetition"])
                ),
                pair_id=(
                    None if payload.get("pair_id") is None else str(payload["pair_id"])
                ),
                success=payload.get("success"),
                requirement_fraction=(
                    None
                    if payload.get("requirement_fraction") is None
                    else float(payload["requirement_fraction"])
                ),
                termination_reason=str(payload["termination_reason"]),
                latency_seconds=float(payload["latency_seconds"]),
                input_tokens=_optional_int(payload.get("input_tokens")),
                output_tokens=_optional_int(payload.get("output_tokens")),
                cache_read_tokens=_optional_int(payload.get("cache_read_tokens")),
                reasoning_tokens=_optional_int(payload.get("reasoning_tokens")),
                request_count=int(payload["request_count"]),
                provider=(
                    None if payload.get("provider") is None else str(payload["provider"])
                ),
                model_id=(
                    None if payload.get("model_id") is None else str(payload["model_id"])
                ),
                source_name=str(payload["source_name"]),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise CalibrationError("observation is incomplete") from error


@dataclass(frozen=True)
class CalibrationSlot:
    slot_id: str
    kind: str
    repetition: int
    task_index: int
    task_id: str
    scenario_id: str
    model_episodes: int


@dataclass(frozen=True)
class InventoryCounts:
    stores_read: int = 0
    captures_read: int = 0
    open_episodes: int = 0
    unreadable_episodes: int = 0
    excluded_closed_split: int = 0
    scenario_conflicts: int = 0


@dataclass(frozen=True)
class BoundCalibration:
    slots: tuple[CalibrationSlot, ...]
    filled: dict[str, tuple[ScoredObservation, ...]]
    sources: dict[str, str]
    surplus_aa_pairs: int
    scenario_conflicts: int


def build_slots(
    task_set: TaskSet,
    *,
    repetitions: int,
    aa_modes: Sequence[str],
) -> tuple[CalibrationSlot, ...]:
    """Baseline slots first, then A/A, in manifest order."""

    _positive(repetitions, "repetitions")
    _refuse_closed(task_set.split)
    modes = _aa_modes(aa_modes)
    if any(scenario is None or scenario == "" for scenario in task_set.scenario_ids):
        raise CalibrationError("calibration requires a scenario id on every task")
    slots: list[CalibrationSlot] = []
    for repetition in range(repetitions):
        for index, (task_id, scenario_id) in enumerate(
            zip(task_set.task_ids, task_set.scenario_ids, strict=True)
        ):
            slots.append(
                _slot("baseline_production", repetition, index, task_id, str(scenario_id), 1)
            )
            slots.append(
                _slot("baseline_do_nothing", repetition, index, task_id, str(scenario_id), 0)
            )
    for mode in modes:
        kind = "aa_execute" if mode == "execute" else "aa_plan"
        for repetition in range(repetitions):
            for index, (task_id, scenario_id) in enumerate(
                zip(task_set.task_ids, task_set.scenario_ids, strict=True)
            ):
                slots.append(
                    _slot(kind, repetition, index, task_id, str(scenario_id), 2)
                )
    return tuple(slots)


def bind_observations(
    slots: Sequence[CalibrationSlot],
    observations: Sequence[ScoredObservation],
    *,
    configuration_hash: str,
    task_scenarios: Mapping[str, str],
    occupied: Mapping[str, tuple[ScoredObservation, ...]] | None = None,
    occupied_sources: Mapping[str, str] | None = None,
) -> BoundCalibration:
    """Fill empty slots from scored observations of this hash. Each episode is used once."""

    filled: dict[str, tuple[ScoredObservation, ...]] = {
        slot_id: group for slot_id, group in (occupied or {}).items()
    }
    sources = {
        slot_id: (occupied_sources or {}).get(slot_id, "checkpoint") for slot_id in filled
    }
    used = {item.observation_id for group in filled.values() for item in group}
    eligible: list[ScoredObservation] = []
    conflicts = 0
    for observation in observations:
        adopted = _adopt(observation, configuration_hash, task_scenarios)
        if adopted is None:
            if (
                observation.configuration_hash == configuration_hash
                and observation.task_id in task_scenarios
                and observation.scenario_id not in (None, task_scenarios[observation.task_id])
            ):
                conflicts += 1
            continue
        if adopted.observation_id in used:
            continue
        eligible.append(adopted)
    by_slot = {(slot.kind, slot.task_id, slot.repetition): slot for slot in slots}
    surplus = 0
    groups: dict[str, list[ScoredObservation]] = defaultdict(list)
    for observation in eligible:
        if observation.pair_id:
            groups[observation.pair_id].append(observation)
    pairs: list[tuple[ScoredObservation, ScoredObservation]] = []
    for members in groups.values():
        complete = _complete_aa_pair(members)
        if complete is not None:
            pairs.append(complete)
    pairs.sort(key=lambda item: (item[0].source_name, item[0].pair_id or "", item[0].observation_id))
    for reference, candidate in pairs:
        kind = "aa_execute" if reference.mode == "execute" else "aa_plan"
        if not any(slot.kind == kind for slot in slots):
            continue
        slot = _take_slot(by_slot, filled, kind, reference.task_id, reference.repetition)
        if slot is None:
            surplus += 1
            used.add(reference.observation_id)
            used.add(candidate.observation_id)
            continue
        filled[slot.slot_id] = (reference, candidate)
        sources[slot.slot_id] = "reused"
        used.add(reference.observation_id)
        used.add(candidate.observation_id)
    for observation in _sorted_baseline(eligible, used):
        slot = _take_slot(
            by_slot, filled, "baseline_production", observation.task_id, observation.repetition
        )
        if slot is None:
            continue
        filled[slot.slot_id] = (observation,)
        sources[slot.slot_id] = "reused"
        used.add(observation.observation_id)
    for observation in _sorted_nothing(eligible, used):
        slot = _take_slot(
            by_slot, filled, "baseline_do_nothing", observation.task_id, observation.repetition
        )
        if slot is None:
            continue
        filled[slot.slot_id] = (observation,)
        sources[slot.slot_id] = "reused"
        used.add(observation.observation_id)
    return BoundCalibration(
        slots=tuple(slots),
        filled=filled,
        sources=sources,
        surplus_aa_pairs=surplus,
        scenario_conflicts=conflicts,
    )


def select_batch(
    slots: Sequence[CalibrationSlot],
    filled: Mapping[str, object],
    *,
    max_model_episodes: int,
) -> tuple[CalibrationSlot, ...]:
    """Take the next unfilled slots without exceeding the model-episode cap.

    A slot that needs more model episodes than remain is not started, so an
    A/A pair is never split. Do-nothing slots are included only while the
    walk has not stopped on a model slot.
    """

    _positive(max_model_episodes, "max_model_episodes")
    chosen: list[CalibrationSlot] = []
    spent = 0
    for slot in slots:
        if slot.slot_id in filled:
            continue
        if spent + slot.model_episodes > max_model_episodes:
            if slot.model_episodes > 0:
                break
            continue
        chosen.append(slot)
        spent += slot.model_episodes
    return tuple(chosen)


def episode_token_cost(
    observation: ScoredObservation,
    pricing: PricingTable,
) -> float | None:
    """Price one episode with ``usage.token_cost``. Unknown counts leave it unset."""

    if observation.provider is None or observation.model_id is None:
        return None
    entry = pricing.entry_for(observation.provider, observation.model_id)
    if entry is None:
        return None
    rates = dict(entry.per_million_tokens)
    if "input_tokens" not in rates or "output_tokens" not in rates:
        return None
    try:
        return token_cost(
            observation.provider,
            rates,
            input_tokens=observation.input_tokens,
            output_tokens=observation.output_tokens,
            cache_read_tokens=observation.cache_read_tokens,
            cache_write_tokens=None,
        )
    except UsageError:
        return None


def calibration_report(
    *,
    configuration: RunConfiguration,
    task_set: TaskSet,
    observations: Sequence[ScoredObservation],
    bound: BoundCalibration,
    counts: InventoryCounts,
    repetitions: int,
    confidence_level: float,
    resamples: int,
    seed: int,
    max_model_episodes: int,
    pricing: PricingTable | None,
    provenance: Mapping[str, bool],
) -> dict[str, object]:
    """Public coverage, inventory, overlap, and estimates. No task or scenario ids."""

    _reporting(confidence_level, resamples, seed)
    target = run_configuration_hash(configuration)
    scenarios = {
        task_id: str(scenario_id)
        for task_id, scenario_id in zip(task_set.task_ids, task_set.scenario_ids, strict=True)
    }
    coverage = _coverage(bound.slots, bound.filled)
    batch = select_batch(bound.slots, bound.filled, max_model_episodes=max_model_episodes)
    document: dict[str, object] = {
        "record": "hosted_calibration_v1",
        "split": task_set.split,
        "configuration_hash": target,
        "task_set_hash": task_set.task_set_hash,
        "task_count": task_set.task_count,
        "scenario_count": task_set.scenario_count,
        "repetitions": repetitions,
        "sampling": {
            "repetitions": repetitions,
            "allocation_status": "provisional",
            "protocol_threshold": False,
            "power_based_target": "unmeasured",
            "pilot_evidence_needed": list(_PILOT_EVIDENCE),
        },
        "provenance": {
            "head_matches": bool(provenance.get("head_matches")),
            "relevant_dirty": bool(provenance.get("relevant_dirty")),
            "run_allowed": bool(provenance.get("run_allowed")),
        },
        "completion": coverage,
        "arms": _arm_estimates(
            bound,
            confidence_level=confidence_level,
            resamples=resamples,
            seed=seed,
        ),
        "surplus_aa_pairs": bound.surplus_aa_pairs,
        "scenario_conflicts": counts.scenario_conflicts + bound.scenario_conflicts,
        "reference_baseline_overlap": _overlap_report(
            observations,
            scenarios,
            repetitions=repetitions,
            target_hash=target,
        ),
        "inventory": _inventory(
            observations,
            target_hash=target,
            confidence_level=confidence_level,
            resamples=resamples,
            seed=seed + 20,
        ),
        "inventory_counts": {
            "stores_read": counts.stores_read,
            "captures_read": counts.captures_read,
            "open_episodes": counts.open_episodes,
            "unreadable_episodes": counts.unreadable_episodes,
            "excluded_closed_split": counts.excluded_closed_split,
            "finished_observations": len(observations),
        },
        "next_batch": _batch_summary(batch, max_model_episodes),
        "cost": _cost_report(observations, configuration, coverage, batch, pricing),
    }
    document["qualification_complete"] = all(
        isinstance(item, dict) and item.get("status") == "qualified"
        for item in document["arms"].values()
    )
    assert_public_payload(document)
    return document


def load_checkpoint(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise CalibrationError("checkpoint does not exist")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise CalibrationError("checkpoint is not readable JSON") from error
    if not isinstance(payload, dict) or payload.get("version") != CHECKPOINT_VERSION:
        raise CalibrationError("checkpoint version is not supported")
    return payload


def checkpoint_identity(
    configuration: RunConfiguration,
    task_set: TaskSet,
    *,
    repetitions: int,
    aa_modes: Sequence[str],
) -> dict[str, object]:
    return {
        "configuration_hash": run_configuration_hash(configuration),
        "task_set_hash": task_set.task_set_hash,
        "split": task_set.split,
        "repetitions": repetitions,
        "aa_modes": list(_aa_modes(aa_modes)),
    }


def occupied_from_checkpoint(
    payload: Mapping[str, object],
    identity: Mapping[str, object],
) -> dict[str, tuple[ScoredObservation, ...]]:
    stored = payload.get("identity")
    if stored != identity:
        raise CalibrationError("checkpoint identity does not match this calibration")
    raw_slots = payload.get("slots")
    if not isinstance(raw_slots, dict):
        raise CalibrationError("checkpoint slots are invalid")
    occupied: dict[str, tuple[ScoredObservation, ...]] = {}
    for slot_id, record in raw_slots.items():
        if not isinstance(slot_id, str) or not isinstance(record, dict):
            raise CalibrationError("checkpoint slot is invalid")
        if record.get("status") != "filled":
            continue
        raw_observations = record.get("observations")
        if not isinstance(raw_observations, list) or not raw_observations:
            raise CalibrationError("checkpoint slot is invalid")
        occupied[slot_id] = tuple(ScoredObservation.from_dict(item) for item in raw_observations)
    return occupied


def sources_from_checkpoint(payload: Mapping[str, object]) -> dict[str, str]:
    raw_slots = payload.get("slots")
    if not isinstance(raw_slots, dict):
        raise CalibrationError("checkpoint slots are invalid")
    sources: dict[str, str] = {}
    for slot_id, record in raw_slots.items():
        if not isinstance(slot_id, str) or not isinstance(record, dict):
            raise CalibrationError("checkpoint slot is invalid")
        if record.get("status") != "filled":
            continue
        source = record.get("source")
        sources[slot_id] = source if isinstance(source, str) and source else "checkpoint"
    return sources


def write_checkpoint(
    path: Path,
    *,
    identity: Mapping[str, object],
    bound: BoundCalibration,
) -> None:
    _refuse_results(path)
    slots: dict[str, object] = {}
    for slot_id, observations in sorted(bound.filled.items()):
        slots[slot_id] = {
            "status": "filled",
            "source": bound.sources.get(slot_id, "reused"),
            "observations": [item.to_dict() for item in observations],
        }
    _atomic_json(
        path,
        {"version": CHECKPOINT_VERSION, "identity": dict(identity), "slots": slots},
    )


def apply_executed(
    bound: BoundCalibration,
    slot: CalibrationSlot,
    observations: Sequence[ScoredObservation],
) -> BoundCalibration:
    """Record one finished slot. The observations must match the slot."""

    if slot.slot_id in bound.filled:
        raise CalibrationError("slot is already filled")
    if slot.kind == "baseline_do_nothing":
        if len(observations) != 1 or observations[0].role != "do_nothing":
            raise CalibrationError("do-nothing slot needs one do-nothing observation")
        if observations[0].success is None:
            raise CalibrationError("do-nothing slot needs an evaluator outcome")
    elif slot.kind == "baseline_production":
        if len(observations) != 1 or observations[0].success is None:
            raise CalibrationError("baseline slot needs one scored observation")
        if observations[0].role == "do_nothing":
            raise CalibrationError("baseline slot cannot be do-nothing")
    elif slot.kind in _AA_KINDS:
        if len(observations) != 2:
            raise CalibrationError("A/A slot needs two observations")
        if {item.role for item in observations} != {"reference", "candidate"}:
            raise CalibrationError("A/A slot needs reference and candidate")
        mode = "execute" if slot.kind == "aa_execute" else "plan"
        if any(item.mode != mode for item in observations):
            raise CalibrationError("A/A observations must match the slot mode")
        if mode == "execute" and any(item.success is None for item in observations):
            raise CalibrationError("execute A/A slot needs evaluator outcomes")
    else:
        raise CalibrationError("slot kind is unknown")
    if any(item.task_id != slot.task_id for item in observations):
        raise CalibrationError("observation task does not match the slot")
    filled = dict(bound.filled)
    sources = dict(bound.sources)
    filled[slot.slot_id] = tuple(observations)
    sources[slot.slot_id] = "executed"
    return replace(bound, filled=filled, sources=sources)


def read_sqlite_store(path: Path) -> tuple[tuple[ScoredObservation, ...], InventoryCounts]:
    observations: list[ScoredObservation] = []
    open_episodes = 0
    unreadable = 0
    excluded = 0
    try:
        connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    except sqlite3.Error as error:
        raise CalibrationError(f"store is not readable: {path.name}") from error
    try:
        try:
            rows = connection.execute("SELECT state, result_json FROM episodes").fetchall()
        except sqlite3.Error as error:
            raise CalibrationError(f"store is not readable: {path.name}") from error
        for state, blob in rows:
            if state != "finished" or not blob:
                open_episodes += 1
                continue
            try:
                payload = json.loads(blob)
                observation = observation_from_episode_dict(payload, source_name=path.name)
            except (CalibrationError, json.JSONDecodeError, TypeError, ValueError):
                unreadable += 1
                continue
            if observation.split in _CLOSED_SPLITS:
                excluded += 1
                continue
            observations.append(observation)
    finally:
        connection.close()
    return tuple(observations), InventoryCounts(
        stores_read=1,
        open_episodes=open_episodes,
        unreadable_episodes=unreadable,
        excluded_closed_split=excluded,
    )


def read_capture_file(
    path: Path,
    *,
    on_progress: Callable[[int], None] | None = None,
) -> tuple[tuple[ScoredObservation, ...], InventoryCounts]:
    observations: list[ScoredObservation] = []
    unreadable = 0
    excluded = 0
    seen = 0
    for record in _iter_capture_records(path):
        seen += 1
        if on_progress is not None and seen % 100 == 0:
            on_progress(seen)
        try:
            schedule = record.get("schedule")
            repetition = None
            if isinstance(schedule, Mapping) and schedule.get("repetition") is not None:
                repetition = int(schedule["repetition"])
            pair = record.get("pair")
            if not isinstance(pair, Mapping):
                raise CalibrationError("capture record has no pair")
            for role in ("reference", "candidate"):
                episode = pair.get(role)
                if not isinstance(episode, Mapping):
                    raise CalibrationError("capture side is missing")
                observation = observation_from_episode_dict(
                    episode,
                    source_name=path.name,
                    repetition=repetition,
                    role=role,
                )
                if observation.split in _CLOSED_SPLITS:
                    excluded += 1
                    continue
                observations.append(observation)
        except (CalibrationError, TypeError, ValueError):
            unreadable += 1
    if on_progress is not None:
        on_progress(seen)
    return tuple(observations), InventoryCounts(
        captures_read=1,
        unreadable_episodes=unreadable,
        excluded_closed_split=excluded,
    )


def observation_from_episode_dict(
    payload: Mapping[str, object],
    *,
    source_name: str,
    repetition: int | None = None,
    role: str | None = None,
) -> ScoredObservation:
    run = payload.get("run")
    task = payload.get("task")
    episode = payload.get("episode")
    if not isinstance(run, Mapping) or not isinstance(task, Mapping) or not isinstance(episode, Mapping):
        raise CalibrationError("episode is incomplete")
    outcome = payload.get("evaluator_outcome")
    success = None
    fraction = None
    if isinstance(outcome, Mapping):
        if isinstance(outcome.get("success"), bool):
            success = outcome["success"]
        raw_fraction = outcome.get("requirement_fraction")
        if isinstance(raw_fraction, (int, float)) and not isinstance(raw_fraction, bool):
            fraction = float(raw_fraction)
    calls = payload.get("provider_calls")
    call_rows = calls if isinstance(calls, list) else []
    recorded_role = payload.get("role")
    if role is None:
        role = "unpaired" if not isinstance(recorded_role, str) or recorded_role == "" else recorded_role
    termination = payload.get("termination_reason")
    if not isinstance(termination, str) or termination == "":
        raise CalibrationError("episode termination is missing")
    if termination == "do_nothing":
        role = "do_nothing"
    provider, model_id = _provider(call_rows)
    observation_id = str(episode.get("episode_id") or "")
    if observation_id == "":
        raise CalibrationError("episode id is missing")
    return ScoredObservation(
        observation_id=observation_id,
        configuration_hash=str(run.get("configuration_hash") or ""),
        task_set_hash=None if run.get("task_set_hash") is None else str(run.get("task_set_hash")),
        split=str(task.get("split") or ""),
        task_id=str(task.get("task_id") or ""),
        scenario_id=None if task.get("scenario_id") in (None, "") else str(task.get("scenario_id")),
        mode=str(payload.get("mode") or ""),
        role=role,
        repetition=repetition,
        pair_id=None if episode.get("pair_id") in (None, "") else str(episode.get("pair_id")),
        success=success,
        requirement_fraction=fraction,
        termination_reason=termination,
        latency_seconds=_latency(payload, call_rows),
        input_tokens=_token_sum(call_rows, "input_tokens"),
        output_tokens=_token_sum(call_rows, "output_tokens"),
        cache_read_tokens=_token_sum(call_rows, "cache_read_tokens"),
        reasoning_tokens=_token_sum(call_rows, "reasoning_tokens"),
        request_count=len(call_rows),
        provider=provider,
        model_id=model_id,
        source_name=source_name,
    )


def discover_sqlite_stores(root: Path) -> tuple[Path, ...]:
    if not root.is_dir():
        raise CalibrationError("inventory root is not a directory")
    return tuple(sorted(path for path in root.rglob("*.sqlite") if path.is_file()))


def _adopt(
    observation: ScoredObservation,
    configuration_hash: str,
    task_scenarios: Mapping[str, str],
) -> ScoredObservation | None:
    if observation.configuration_hash != configuration_hash:
        return None
    if observation.split not in _OPEN_SPLITS:
        return None
    if observation.observation_id == "":
        return None
    expected = task_scenarios.get(observation.task_id)
    if expected is None:
        return None
    if observation.scenario_id not in (None, expected):
        return None
    if observation.termination_reason in _INFRASTRUCTURE:
        return None
    role = "do_nothing" if observation.termination_reason == "do_nothing" else observation.role
    return replace(observation, scenario_id=expected, role=role)


def _sorted_baseline(
    eligible: Sequence[ScoredObservation], used: set[str]
) -> list[ScoredObservation]:
    chosen = [
        observation
        for observation in eligible
        if observation.observation_id not in used
        and observation.mode == "execute"
        and observation.success is not None
        and observation.role in _BASELINE_ROLES
    ]
    chosen.sort(key=lambda item: (item.source_name, item.observation_id))
    return chosen


def _sorted_nothing(
    eligible: Sequence[ScoredObservation], used: set[str]
) -> list[ScoredObservation]:
    chosen = [
        observation
        for observation in eligible
        if observation.observation_id not in used
        and observation.role == "do_nothing"
        and observation.success is not None
    ]
    chosen.sort(key=lambda item: (item.source_name, item.observation_id))
    return chosen


def _slot(
    kind: str,
    repetition: int,
    task_index: int,
    task_id: str,
    scenario_id: str,
    model_episodes: int,
) -> CalibrationSlot:
    return CalibrationSlot(
        slot_id=f"{kind}:{repetition}:{task_index}",
        kind=kind,
        repetition=repetition,
        task_index=task_index,
        task_id=task_id,
        scenario_id=scenario_id,
        model_episodes=model_episodes,
    )


def _complete_aa_pair(
    members: Sequence[ScoredObservation],
) -> tuple[ScoredObservation, ScoredObservation] | None:
    reference = [item for item in members if item.role == "reference"]
    candidate = [item for item in members if item.role == "candidate"]
    if len(reference) != 1 or len(candidate) != 1:
        return None
    left, right = reference[0], candidate[0]
    if left.task_id != right.task_id or left.mode != right.mode:
        return None
    if left.configuration_hash != right.configuration_hash:
        return None
    if left.mode == "execute" and (left.success is None or right.success is None):
        return None
    if left.mode == "plan" and (
        left.termination_reason != "plan_emitted" or right.termination_reason != "plan_emitted"
    ):
        return None
    if left.mode not in {"plan", "execute"}:
        return None
    return left, right


def _take_slot(
    by_slot: Mapping[tuple[str, str, int], CalibrationSlot],
    filled: Mapping[str, object],
    kind: str,
    task_id: str,
    preferred: int | None,
) -> CalibrationSlot | None:
    repetitions = sorted(
        repetition
        for slot_kind, slot_task, repetition in by_slot
        if slot_kind == kind and slot_task == task_id
    )
    order = list(repetitions)
    if preferred is not None and preferred in order:
        order.remove(preferred)
        order.insert(0, preferred)
    for repetition in order:
        if preferred is not None and repetition != preferred and preferred in repetitions:
            if repetition < preferred:
                continue
        slot = by_slot[(kind, task_id, repetition)]
        if slot.slot_id in filled:
            continue
        if preferred is not None and repetition != preferred:
            return slot
        if preferred is None or repetition == preferred:
            return slot
    if preferred is None:
        return None
    for repetition in repetitions:
        slot = by_slot[(kind, task_id, repetition)]
        if slot.slot_id not in filled:
            return slot
    return None


def _coverage(
    slots: Sequence[CalibrationSlot],
    filled: Mapping[str, tuple[ScoredObservation, ...]],
) -> dict[str, object]:
    arms: dict[str, object] = {}
    for kind in ("baseline_production", "baseline_do_nothing", "aa_execute", "aa_plan"):
        required = [slot for slot in slots if slot.kind == kind]
        done = [slot for slot in required if slot.slot_id in filled]
        arms[kind] = {
            "required": len(required),
            "filled": len(done),
            "fraction": (len(done) / len(required)) if required else 1.0,
            "scenarios_filled": len({slot.scenario_id for slot in done}),
            "model_episodes_filled": sum(slot.model_episodes for slot in done),
            "model_episodes_required": sum(slot.model_episodes for slot in required),
        }
    required_model = sum(slot.model_episodes for slot in slots)
    filled_model = sum(slot.model_episodes for slot in slots if slot.slot_id in filled)
    return {
        "arms": arms,
        "model_episodes_required": required_model,
        "model_episodes_filled": filled_model,
        "model_episodes_remaining": required_model - filled_model,
        "fraction": (filled_model / required_model) if required_model else 1.0,
    }


def _arm_estimates(
    bound: BoundCalibration,
    *,
    confidence_level: float,
    resamples: int,
    seed: int,
) -> dict[str, object]:
    required: dict[str, int] = defaultdict(int)
    for slot in bound.slots:
        required[slot.kind] += 1
    plan_status = _coverage_status(len(_slots_of(bound, "aa_plan")), required["aa_plan"])
    return {
        "baseline_production": _metric_arm(
            _slots_of(bound, "baseline_production"),
            required["baseline_production"],
            confidence_level=confidence_level,
            resamples=resamples,
            seed=seed,
            fraction_seed=seed + 1,
        ),
        "baseline_do_nothing": _metric_arm(
            _slots_of(bound, "baseline_do_nothing"),
            required["baseline_do_nothing"],
            confidence_level=confidence_level,
            resamples=resamples,
            seed=seed + 2,
            fraction_seed=seed + 3,
        ),
        "baseline_difference": _difference_arm(
            bound, confidence_level=confidence_level, resamples=resamples, seed=seed + 4
        ),
        "aa_execute": _aa_arm(
            _slots_of(bound, "aa_execute"),
            required["aa_execute"],
            confidence_level=confidence_level,
            resamples=resamples,
            seed=seed + 6,
        ),
        "aa_plan": _classified(
            plan_status,
            {"pairs": len(_slots_of(bound, "aa_plan")), "required_pairs": required["aa_plan"]},
        ),
    }


def _slots_of(
    bound: BoundCalibration, kind: str
) -> list[tuple[CalibrationSlot, tuple[ScoredObservation, ...]]]:
    return [
        (slot, bound.filled[slot.slot_id])
        for slot in bound.slots
        if slot.kind == kind and slot.slot_id in bound.filled
    ]


def _coverage_status(filled: int, required: int) -> str:
    if required == 0:
        return "qualified"
    if filled <= 0:
        return "unmeasured"
    if filled < required:
        return "preliminary"
    return "qualified"


def _metric_arm(
    filled_slots: Sequence[tuple[CalibrationSlot, tuple[ScoredObservation, ...]]],
    required: int,
    *,
    confidence_level: float,
    resamples: int,
    seed: int,
    fraction_seed: int,
) -> dict[str, object]:
    values, clusters = _success_series(filled_slots)
    fractions, fraction_clusters = _fraction_series(filled_slots)
    body = {
        "filled": len(filled_slots),
        "required": required,
        "success": _interval(values, clusters, confidence_level, resamples, seed),
        "requirement_fraction": _interval(
            fractions, fraction_clusters, confidence_level, resamples, fraction_seed
        ),
    }
    return _classified(_coverage_status(len(filled_slots), required), body)


def _classified(status: str, body: dict[str, object]) -> dict[str, object]:
    return {
        "status": status,
        "preliminary": body if status == "preliminary" else None,
        "qualified": body if status == "qualified" else None,
    }


def _difference_arm(
    bound: BoundCalibration,
    *,
    confidence_level: float,
    resamples: int,
    seed: int,
) -> dict[str, object]:
    required = [slot for slot in bound.slots if slot.kind == "baseline_production"]
    paired_values: list[float] = []
    nothing_values: list[float] = []
    clusters: list[str] = []
    fraction_values: list[float] = []
    fraction_base: list[float] = []
    fraction_clusters: list[str] = []
    for slot in required:
        partner_id = f"baseline_do_nothing:{slot.repetition}:{slot.task_index}"
        if slot.slot_id not in bound.filled or partner_id not in bound.filled:
            continue
        left = bound.filled[slot.slot_id][0]
        right = bound.filled[partner_id][0]
        if left.success is None or right.success is None:
            continue
        paired_values.append(1.0 if left.success else 0.0)
        nothing_values.append(1.0 if right.success else 0.0)
        clusters.append(slot.scenario_id)
        if left.requirement_fraction is not None and right.requirement_fraction is not None:
            fraction_values.append(left.requirement_fraction)
            fraction_base.append(right.requirement_fraction)
            fraction_clusters.append(slot.scenario_id)
    success = None
    if paired_values:
        success = _bootstrap_dict(
            clustered_paired_bootstrap(
                paired_values,
                nothing_values,
                clusters,
                confidence_level=confidence_level,
                resamples=resamples,
                seed=seed,
            ),
            episodes=len(paired_values),
            scenarios=len(set(clusters)),
        )
    fraction = None
    if fraction_values:
        fraction = _bootstrap_dict(
            clustered_paired_bootstrap(
                fraction_values,
                fraction_base,
                fraction_clusters,
                confidence_level=confidence_level,
                resamples=resamples,
                seed=seed + 1,
            ),
            episodes=len(fraction_values),
            scenarios=len(set(fraction_clusters)),
        )
    return _classified(
        _coverage_status(len(paired_values), len(required)),
        {
            "pairs": len(paired_values),
            "required_pairs": len(required),
            "success_difference": success,
            "requirement_fraction_difference": fraction,
        },
    )


def _aa_arm(
    filled_slots: Sequence[tuple[CalibrationSlot, tuple[ScoredObservation, ...]]],
    required: int,
    *,
    confidence_level: float,
    resamples: int,
    seed: int,
) -> dict[str, object]:
    disagreements: list[float] = []
    clusters: list[str] = []
    fractions: list[float] = []
    fraction_base: list[float] = []
    fraction_clusters: list[str] = []
    for slot, observations in filled_slots:
        reference = next(item for item in observations if item.role == "reference")
        candidate = next(item for item in observations if item.role == "candidate")
        if reference.success is None or candidate.success is None:
            continue
        disagreements.append(0.0 if reference.success == candidate.success else 1.0)
        clusters.append(slot.scenario_id)
        if (
            reference.requirement_fraction is not None
            and candidate.requirement_fraction is not None
        ):
            fractions.append(candidate.requirement_fraction)
            fraction_base.append(reference.requirement_fraction)
            fraction_clusters.append(slot.scenario_id)
    fraction = None
    if fractions:
        fraction = _bootstrap_dict(
            clustered_paired_bootstrap(
                fractions,
                fraction_base,
                fraction_clusters,
                confidence_level=confidence_level,
                resamples=resamples,
                seed=seed + 1,
            ),
            episodes=len(fractions),
            scenarios=len(set(fraction_clusters)),
        )
    return _classified(
        _coverage_status(len(filled_slots), required),
        {
            "pairs": len(disagreements),
            "required_pairs": required,
            "disagreement": _interval(disagreements, clusters, confidence_level, resamples, seed),
            "requirement_fraction_difference": fraction,
        },
    )


def _success_series(
    filled_slots: Sequence[tuple[CalibrationSlot, tuple[ScoredObservation, ...]]],
) -> tuple[list[float], list[str]]:
    values: list[float] = []
    clusters: list[str] = []
    for slot, observations in filled_slots:
        observation = observations[0]
        if observation.success is None:
            continue
        values.append(1.0 if observation.success else 0.0)
        clusters.append(slot.scenario_id)
    return values, clusters


def _fraction_series(
    filled_slots: Sequence[tuple[CalibrationSlot, tuple[ScoredObservation, ...]]],
) -> tuple[list[float], list[str]]:
    values: list[float] = []
    clusters: list[str] = []
    for slot, observations in filled_slots:
        observation = observations[0]
        if observation.requirement_fraction is None:
            continue
        values.append(observation.requirement_fraction)
        clusters.append(slot.scenario_id)
    return values, clusters


def _interval(
    values: Sequence[float],
    clusters: Sequence[str],
    confidence_level: float,
    resamples: int,
    seed: int,
) -> dict[str, object] | None:
    if not values:
        return None
    return _bootstrap_dict(
        clustered_paired_bootstrap(
            values,
            [0.0] * len(values),
            clusters,
            confidence_level=confidence_level,
            resamples=resamples,
            seed=seed,
        ),
        episodes=len(values),
        scenarios=len(set(clusters)),
    )


def _bootstrap_dict(result: object, *, episodes: int, scenarios: int) -> dict[str, object]:
    return {
        "estimate": result.mean_difference,
        "confidence_low": result.confidence_low,
        "confidence_high": result.confidence_high,
        "confidence_level": result.confidence_level,
        "resamples": result.resamples,
        "episodes": episodes,
        "scenarios": scenarios,
        "method": "clustered_paired_bootstrap",
    }


def _overlap_report(
    observations: Sequence[ScoredObservation],
    task_scenarios: Mapping[str, str],
    *,
    repetitions: int,
    target_hash: str,
) -> list[dict[str, object]]:
    grouped: dict[str, list[ScoredObservation]] = defaultdict(list)
    for observation in observations:
        if observation.task_id not in task_scenarios or observation.split not in _OPEN_SPLITS:
            continue
        expected = task_scenarios[observation.task_id]
        if observation.scenario_id not in (None, expected):
            continue
        if observation.termination_reason in _INFRASTRUCTURE:
            continue
        grouped[observation.configuration_hash].append(
            replace(observation, scenario_id=expected)
        )
    rows = [
        _overlap_one(configuration_hash, members, repetitions=repetitions, target_hash=target_hash)
        for configuration_hash, members in sorted(grouped.items())
        if configuration_hash == target_hash or _execute_references(members)
    ]
    if target_hash not in grouped:
        rows.insert(
            0,
            _overlap_one(target_hash, (), repetitions=repetitions, target_hash=target_hash),
        )
    return rows


def _execute_references(members: Sequence[ScoredObservation]) -> bool:
    return any(reference.mode == "execute" for reference, _candidate in _aa_groups(members))


def _overlap_one(
    configuration_hash: str,
    members: Sequence[ScoredObservation],
    *,
    repetitions: int,
    target_hash: str,
) -> dict[str, object]:
    pairs = [pair for pair in _aa_groups(members) if pair[0].mode == "execute"]
    reference_ids = {reference.observation_id for reference, _candidate in pairs}
    leftovers = [
        item
        for item in members
        if item.mode == "execute"
        and item.success is not None
        and item.role in _BASELINE_ROLES
        and item.observation_id not in reference_ids
        and item.termination_reason != "do_nothing"
    ]
    refs_by_task: dict[str, int] = defaultdict(int)
    scenarios: set[str] = set()
    for reference, _candidate in pairs:
        refs_by_task[reference.task_id] += 1
        if reference.scenario_id:
            scenarios.add(reference.scenario_id)
    base_by_task: dict[str, int] = defaultdict(int)
    for item in leftovers:
        base_by_task[item.task_id] += 1
    additional = 0
    for task_id, count in refs_by_task.items():
        free = repetitions - min(base_by_task[task_id], repetitions)
        additional += min(count, free)
    do_nothing = sum(1 for item in members if item.role == "do_nothing" and item.success is not None)
    return {
        "configuration_hash": configuration_hash,
        "relabeled_onto_target": False,
        "fills_target_slots": configuration_hash == target_hash,
        "healthy_execute_aa_references": len(reference_ids),
        "reference_scenarios": len(scenarios),
        "exclusive_policy_overlap": 0,
        "other_baseline_episodes": len(leftovers),
        "additional_baseline_slots_if_references_also_counted": additional,
        "overlapping_episodes_if_counted_in_both": additional,
        "references_beyond_repetition_budget": len(reference_ids) - additional,
        "supports_baseline_success": len(reference_ids) > 0,
        "supports_do_nothing_contrast": False,
        "do_nothing_episodes": do_nothing,
    }


def _inventory(
    observations: Sequence[ScoredObservation],
    *,
    target_hash: str,
    confidence_level: float,
    resamples: int,
    seed: int,
) -> list[dict[str, object]]:
    grouped: dict[str, list[ScoredObservation]] = defaultdict(list)
    for observation in observations:
        if observation.split in _OPEN_SPLITS:
            grouped[observation.configuration_hash].append(observation)
    return [
        _inventory_one(
            configuration_hash,
            members,
            target=configuration_hash == target_hash,
            confidence_level=confidence_level,
            resamples=resamples,
            seed=seed + index,
        )
        for index, (configuration_hash, members) in enumerate(sorted(grouped.items()))
    ]


def _inventory_one(
    configuration_hash: str,
    members: Sequence[ScoredObservation],
    *,
    target: bool,
    confidence_level: float,
    resamples: int,
    seed: int,
) -> dict[str, object]:
    scored = [
        item
        for item in members
        if item.mode == "execute"
        and item.success is not None
        and item.termination_reason not in _INFRASTRUCTURE
        and item.scenario_id
    ]
    successes = [1.0 if item.success else 0.0 for item in scored]
    clusters = [str(item.scenario_id) for item in scored]
    fractions = [item.requirement_fraction for item in scored if item.requirement_fraction is not None]
    fraction_clusters = [
        str(item.scenario_id) for item in scored if item.requirement_fraction is not None
    ]
    latencies = [item.latency_seconds for item in members if item.request_count or item.latency_seconds]
    pairs = _aa_groups(members)
    disagreements = []
    disagreement_clusters = []
    for reference, candidate in pairs:
        if reference.mode != "execute" or reference.success is None or candidate.success is None:
            continue
        if not reference.scenario_id:
            continue
        disagreements.append(0.0 if reference.success == candidate.success else 1.0)
        disagreement_clusters.append(reference.scenario_id)
    splits = sorted({item.split for item in members})
    return {
        "configuration_hash": configuration_hash,
        "estimate_class": "target_inventory" if target else "other_configuration",
        "reusable_for_target": target,
        "split": splits[0] if len(splits) == 1 else "mixed",
        "episodes": len(members),
        "tasks": len({item.task_id for item in members}),
        "scenarios": len({item.scenario_id for item in members if item.scenario_id}),
        "scenario_missing": sum(1 for item in members if not item.scenario_id),
        "scored_execute": len(scored),
        "aa_execute_pairs": sum(1 for left, _right in pairs if left.mode == "execute"),
        "aa_plan_pairs": sum(1 for left, _right in pairs if left.mode == "plan"),
        "success": _interval(successes, clusters, confidence_level, resamples, seed),
        "requirement_fraction": _interval(
            [float(item) for item in fractions],
            fraction_clusters,
            confidence_level,
            resamples,
            seed + 1,
        ),
        "requirement_fraction_sd": _sample_sd([float(item) for item in fractions]),
        "aa_disagreement": _interval(
            disagreements, disagreement_clusters, confidence_level, resamples, seed + 2
        ),
        "latency_seconds_mean": fmean(latencies) if latencies else None,
        "latency_seconds_sd": _sample_sd(latencies),
        "usage": _usage_totals(members),
    }


def _aa_groups(
    members: Sequence[ScoredObservation],
) -> list[tuple[ScoredObservation, ScoredObservation]]:
    grouped: dict[str, list[ScoredObservation]] = defaultdict(list)
    for item in members:
        if item.pair_id:
            grouped[item.pair_id].append(item)
    pairs = []
    for group in grouped.values():
        complete = _complete_aa_pair(group)
        if complete is not None:
            pairs.append(complete)
    return pairs


def _usage_totals(members: Sequence[ScoredObservation]) -> dict[str, object]:
    def total(name: str) -> int | None:
        values = [getattr(item, name) for item in members if item.request_count]
        if not values or any(value is None for value in values):
            return None
        return int(sum(values))

    return {
        "requests": sum(item.request_count for item in members),
        "input_tokens": total("input_tokens"),
        "output_tokens": total("output_tokens"),
        "cache_read_tokens": total("cache_read_tokens"),
        "reasoning_tokens": total("reasoning_tokens"),
    }


def _mixed_candidate_ids(observations: Sequence[ScoredObservation]) -> set[str]:
    grouped: dict[str, list[ScoredObservation]] = defaultdict(list)
    for item in observations:
        if item.pair_id:
            grouped[item.pair_id].append(item)
    excluded: set[str] = set()
    for group in grouped.values():
        if len({item.configuration_hash for item in group}) < 2:
            continue
        for item in group:
            if item.role == "candidate":
                excluded.add(item.observation_id)
    return excluded


def _cost_report(
    observations: Sequence[ScoredObservation],
    configuration: RunConfiguration,
    coverage: Mapping[str, object],
    batch: Sequence[CalibrationSlot],
    pricing: PricingTable | None,
) -> dict[str, object]:
    provider = hosted_provider(configuration.model)
    model_id = getattr(configuration.model, "model_id", None)
    target = run_configuration_hash(configuration)
    excluded = _mixed_candidate_ids(observations)
    matching = [
        item
        for item in observations
        if item.mode == "execute"
        and item.provider == provider
        and item.model_id == model_id
        and item.request_count > 0
        and item.observation_id not in excluded
    ]
    target_rows = [item for item in matching if item.configuration_hash == target]
    prior_rows = target_rows or [item for item in matching if item.configuration_hash != target]
    costs: list[float] = []
    if pricing is not None:
        for item in prior_rows:
            cost = episode_token_cost(item, pricing)
            if cost is not None:
                costs.append(cost)
    per_episode = fmean(costs) if costs else None
    batch_execute = sum(slot.model_episodes for slot in batch if slot.kind != "aa_plan")
    batch_plan = sum(slot.model_episodes for slot in batch if slot.kind == "aa_plan")
    status = "unmeasured"
    if per_episode is not None and target_rows:
        status = "projected_from_same_configuration"
    elif per_episode is not None:
        status = "projected_from_other_configuration"
    hashes = sorted({item.configuration_hash for item in prior_rows}) if costs else []
    return {
        "status": status,
        "pricing_version": None if pricing is None else pricing.pricing_version,
        "currency": None if pricing is None else pricing.currency,
        "target_priced_episodes": len(target_rows),
        "prior_execute_episodes": len(costs),
        "prior_configuration_hashes": hashes,
        "per_execute_episode": per_episode,
        "per_execute_episode_sd": _sample_sd(costs),
        "next_batch_execute_episodes": batch_execute,
        "next_batch_plan_episodes": batch_plan,
        "next_batch_estimated": None
        if per_episode is None or batch_plan
        else per_episode * batch_execute,
        "remaining_execute_episodes": _remaining_execute(coverage),
        "remaining_execute_estimated": None
        if per_episode is None
        else per_episode * _remaining_execute(coverage),
        "remaining_plan_episodes": _remaining_plan(coverage),
        "remaining_plan_estimated": None,
        "plan_estimate": "unmeasured",
    }


def _remaining_execute(coverage: Mapping[str, object]) -> int:
    arms = coverage["arms"]
    assert isinstance(arms, Mapping)
    total = 0
    for kind in ("baseline_production", "aa_execute"):
        arm = arms[kind]
        assert isinstance(arm, Mapping)
        total += int(arm["model_episodes_required"]) - int(arm["model_episodes_filled"])
    return total


def _remaining_plan(coverage: Mapping[str, object]) -> int:
    arms = coverage["arms"]
    assert isinstance(arms, Mapping)
    arm = arms["aa_plan"]
    assert isinstance(arm, Mapping)
    return int(arm["model_episodes_required"]) - int(arm["model_episodes_filled"])


def _batch_summary(batch: Sequence[CalibrationSlot], max_model_episodes: int) -> dict[str, object]:
    counts: dict[str, int] = defaultdict(int)
    for slot in batch:
        counts[slot.kind] += 1
    return {
        "max_model_episodes": max_model_episodes,
        "model_episodes": sum(slot.model_episodes for slot in batch),
        "baseline_production": counts["baseline_production"],
        "baseline_do_nothing": counts["baseline_do_nothing"],
        "aa_execute_pairs": counts["aa_execute"],
        "aa_plan_pairs": counts["aa_plan"],
    }


def _sample_sd(values: Sequence[float]) -> float | None:
    if len(values) < 2:
        return None
    return float(stdev(values))


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise CalibrationError("token count is invalid")
    return value


def _positive(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise CalibrationError(f"{name} must be a positive integer")


def _reporting(confidence_level: float, resamples: int, seed: int) -> None:
    if (
        isinstance(confidence_level, bool)
        or not isinstance(confidence_level, (int, float))
        or not 0.0 < float(confidence_level) < 1.0
    ):
        raise CalibrationError("confidence_level must be between zero and one")
    _positive(resamples, "resamples")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise CalibrationError("seed must be an int")


def _refuse_closed(split: str) -> None:
    if split in _CLOSED_SPLITS:
        raise CalibrationError("calibration is closed on this split")
    if split not in _OPEN_SPLITS:
        raise CalibrationError("calibration split must be train or dev")


def _aa_modes(modes: Sequence[str]) -> tuple[str, ...]:
    if isinstance(modes, str) or not modes:
        raise CalibrationError("modes must be plan, execute, or both")
    chosen = [mode for mode in ("execute", "plan") if mode in modes]
    if len(chosen) != len({mode for mode in modes}):
        raise CalibrationError("mode must be plan or execute")
    if any(mode not in {"plan", "execute"} for mode in modes):
        raise CalibrationError("mode must be plan or execute")
    return tuple(chosen)


def _provider(calls: Sequence[object]) -> tuple[str | None, str | None]:
    if not calls or not isinstance(calls[0], Mapping):
        return None, None
    provider = calls[0].get("provider")
    model_id = calls[0].get("model_id")
    return (
        provider if isinstance(provider, str) else None,
        model_id if isinstance(model_id, str) else None,
    )


def _token_sum(calls: Sequence[object], name: str) -> int | None:
    if not calls:
        return None
    total = 0
    for call in calls:
        if not isinstance(call, Mapping) or call.get(name) is None:
            return None
        value = call[name]
        if isinstance(value, bool) or not isinstance(value, int):
            return None
        total += value
    return total


def _latency(payload: Mapping[str, object], calls: Sequence[object]) -> float:
    if calls:
        total = 0.0
        for call in calls:
            if isinstance(call, Mapping) and isinstance(call.get("latency_seconds"), (int, float)):
                total += float(call["latency_seconds"])
        return total
    steps = payload.get("model_steps")
    if isinstance(steps, list) and steps:
        total = 0.0
        for step in steps:
            if isinstance(step, Mapping) and isinstance(step.get("latency_seconds"), (int, float)):
                total += float(step["latency_seconds"])
        return total
    try:
        started = datetime.fromisoformat(str(payload.get("started_at")))
        ended = datetime.fromisoformat(str(payload.get("ended_at")))
    except ValueError:
        return 0.0
    return max(0.0, (ended - started).total_seconds())


def _iter_capture_records(path: Path):
    decoder = json.JSONDecoder()
    marker = '"records"'
    try:
        handle = path.open(encoding="utf-8")
    except OSError as error:
        raise CalibrationError(f"capture is not readable: {path.name}") from error
    with handle:
        buffer = ""
        while True:
            chunk = handle.read(1 << 20)
            if not chunk:
                raise CalibrationError("capture has no records")
            buffer += chunk
            index = buffer.find(marker)
            if index < 0:
                continue
            rest = buffer[index + len(marker) :].lstrip()
            if rest == "" or rest == ":" or (rest.startswith(":") and rest[1:].strip() == ""):
                continue
            if rest.startswith(":"):
                rest = rest[1:].lstrip()
            if not rest.startswith("["):
                raise CalibrationError("capture records are incomplete")
            buffer = rest[1:]
            break
        while True:
            buffer = buffer.lstrip()
            if buffer.startswith("]"):
                return
            if buffer.startswith(","):
                buffer = buffer[1:]
                continue
            try:
                record, end = decoder.raw_decode(buffer)
            except json.JSONDecodeError:
                chunk = handle.read(1 << 20)
                if not chunk:
                    raise CalibrationError("capture records are incomplete") from None
                buffer += chunk
                continue
            if isinstance(record, dict):
                yield record
            buffer = buffer[end:]


def _refuse_results(path: Path) -> None:
    if any(parent.name == "results" for parent in (path, *path.parents)):
        raise CalibrationError("calibration output cannot be written as a public result")


def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
