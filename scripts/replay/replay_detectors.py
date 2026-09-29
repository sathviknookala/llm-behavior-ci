"""Replay configured detectors on a frozen observation stream.

Requires --observations, --schedule, --factories, and --output for a
scalar signal stream. --distributional-counts together with
--distributional-settings, --schedule, and --output replays a
tool_selection/task_mix counts stream instead. Does not start an agent.
Paths under a directory named results are refused.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path

from llm_behavior_ci.config import DistributionalMonitorSettings, MonitorSettings
from llm_behavior_ci.experiments.replay import (
    ReplayError,
    ReplaySchedule,
    distributional_detector_factories,
    monitoring_detector_factories,
    replay_detectors,
    replay_distributional_detectors,
    replay_result_public_dict,
)
from llm_behavior_ci.lifecycle.monitoring import FrozenReference
from llm_behavior_ci.records import MonitorObservation, RecordError, assert_public_payload


def _under_results(path: Path) -> bool:
    for parent in path.resolve().parents:
        if parent.name == "results":
            return True
    return False


def _load_json(path: Path) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ReplayError("input document is not readable JSON") from error


def _refuse_results(*paths: Path) -> None:
    for path in paths:
        if _under_results(path):
            raise ReplayError("paths under a results directory are refused")


def _timestamp(value: object, name: str) -> datetime:
    if not isinstance(value, str):
        raise ReplayError(f"{name} must be an ISO-8601 timestamp")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        raise ReplayError(f"{name} must be an ISO-8601 timestamp") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ReplayError(f"{name} must be timezone-aware")
    return parsed


def _load_observations(payload: object) -> tuple[MonitorObservation, ...]:
    if not isinstance(payload, list):
        raise ReplayError("observations document must be a list")
    try:
        return tuple(MonitorObservation.from_dict(item) for item in payload)
    except (RecordError, TypeError, ValueError) as error:
        raise ReplayError("observations document is invalid") from error


def _load_schedule(payload: object) -> ReplaySchedule:
    if not isinstance(payload, Mapping):
        raise ReplayError("schedule document must be an object")
    try:
        scenario_keys = payload["scenario_keys"]
        arrival_times = payload["arrival_times"]
        if not isinstance(scenario_keys, list) or not isinstance(arrival_times, list):
            raise ReplayError("scenario_keys and arrival_times must be lists")
        return ReplaySchedule(
            outcome_delay_seconds=float(payload["outcome_delay_seconds"]),
            horizon_episodes=int(payload["horizon_episodes"]),
            onset_index=int(payload["onset_index"]),
            scenario_keys=tuple(str(item) for item in scenario_keys),
            arrival_times=tuple(
                _timestamp(item, "arrival_times") for item in arrival_times
            ),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ReplayError("schedule document is invalid") from error


def _load_reference(payload: object) -> FrozenReference:
    if not isinstance(payload, Mapping):
        raise ReplayError("reference must be an object")
    try:
        baselines_raw = payload["baselines"]
        if not isinstance(baselines_raw, list):
            raise ReplayError("baselines must be a list of pairs")
        baselines: list[tuple[str, float]] = []
        for item in baselines_raw:
            if not isinstance(item, (list, tuple)) or len(item) != 2:
                raise ReplayError("baselines must be a list of pairs")
            baselines.append((str(item[0]), float(item[1])))
        return FrozenReference(
            configuration_hash=str(payload["configuration_hash"]),
            baselines=tuple(baselines),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ReplayError("reference document is invalid") from error


def _load_counts(payload: object) -> tuple[dict[str, int], ...]:
    if not isinstance(payload, list) or not payload:
        raise ReplayError("distributional counts document must be a non-empty list")
    try:
        return tuple(
            {str(name): int(count) for name, count in item.items()}
            for item in payload
        )
    except (AttributeError, TypeError, ValueError) as error:
        raise ReplayError("distributional counts document is invalid") from error


def _load_distributional_settings(payload: object) -> DistributionalMonitorSettings:
    try:
        return DistributionalMonitorSettings.from_dict(payload)
    except (KeyError, TypeError, ValueError) as error:
        raise ReplayError("distributional settings document is invalid") from error


def _load_factories_spec(payload: object):
    if not isinstance(payload, Mapping):
        raise ReplayError("factory spec must be an object")
    try:
        corrections_raw = payload["corrections"]
        if not isinstance(corrections_raw, list):
            raise ReplayError("corrections must be a list")
        reference_sample_raw = payload["reference_sample"]
        if not isinstance(reference_sample_raw, list):
            raise ReplayError("reference_sample must be a list")
        settings = MonitorSettings.from_dict(payload["settings"])
        reference = _load_reference(payload["reference"])
        return monitoring_detector_factories(
            settings,
            reference,
            signal=str(payload["signal"]),
            alpha=float(payload["alpha"]),
            window_episodes=int(payload["window_episodes"]),
            reference_sample=tuple(float(item) for item in reference_sample_raw),
            harm_margin=float(payload["harm_margin"]),
            corrections=tuple(str(item) for item in corrections_raw),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ReplayError("factory spec is invalid") from error


def main(argv: Sequence[str] | None = None) -> int:
    args_list = list(sys.argv[1:] if argv is None else argv)
    if not args_list:
        return 2
    parser = argparse.ArgumentParser(description="Replay detectors on a frozen stream.")
    parser.add_argument("--observations")
    parser.add_argument("--schedule")
    parser.add_argument("--factories")
    parser.add_argument("--distributional-counts")
    parser.add_argument("--distributional-settings")
    parser.add_argument("--output")
    try:
        args = parser.parse_args(args_list)
    except SystemExit as error:
        code = error.code
        if code is None:
            return 2
        return int(code)
    if args.distributional_counts is not None:
        if (
            args.distributional_settings is None
            or args.schedule is None
            or args.output is None
        ):
            return 2
    elif (
        args.observations is None
        or args.schedule is None
        or args.factories is None
        or args.output is None
    ):
        return 2
    try:
        schedule_path = Path(args.schedule)
        output_path = Path(args.output)
        if args.distributional_counts is not None:
            counts_path = Path(args.distributional_counts)
            distributional_settings_path = Path(args.distributional_settings)
            _refuse_results(
                counts_path,
                distributional_settings_path,
                schedule_path,
                output_path,
            )
            counts = _load_counts(_load_json(counts_path))
            schedule = _load_schedule(_load_json(schedule_path))
            settings = _load_distributional_settings(
                _load_json(distributional_settings_path)
            )
            factories = distributional_detector_factories(settings)
            results = replay_distributional_detectors(
                counts, factories, schedule=schedule
            )
        else:
            observations_path = Path(args.observations)
            factories_path = Path(args.factories)
            _refuse_results(
                observations_path,
                schedule_path,
                factories_path,
                output_path,
            )
            observations = _load_observations(_load_json(observations_path))
            schedule = _load_schedule(_load_json(schedule_path))
            factories = _load_factories_spec(_load_json(factories_path))
            results = replay_detectors(observations, factories, schedule=schedule)
        document = {
            "visibility": "public",
            "stream_hash": next(iter(results.values())).stream_hash,
            "methods": {
                name: replay_result_public_dict(result)
                for name, result in results.items()
            },
        }
        assert_public_payload(document)
        output_path.write_text(
            json.dumps(document, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        return 0
    except ReplayError as error:
        print(str(error) or "replay failed", file=sys.stderr)
        return 1
    except RecordError as error:
        print(str(error) or "public payload rejected", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
