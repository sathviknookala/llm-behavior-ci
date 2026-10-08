"""Rehearse the scheduled monitor stream on a dev task set.

Runs ``experiments.schedule.run_scheduled_monitor``, the loop the frozen
benchmark uses, with a healthy prefix served by --base and the faulted
configuration (``apply_fault(base, --fault)``) from onset. Requires
--base, --fault, --task-set (a local dev manifest), --schedule,
--monitor-settings, --frozen-reference, and --state (a local JSON
checkpoint, resumable). --distributional-monitors is optional. A vLLM
configuration needs --reference-endpoint (healthy) and
--candidate-endpoint (faulted); hosted configurations take none and read
their keys from the environment. Refuses a non-dev task set and any path
under results/. Prints counters, healthy-prefix alarms, and post-onset
delay; the output is rehearsal evidence for calibration, not a result.
Bare invocation exits 2.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path

from llm_behavior_ci.config import (
    ConfigError,
    DistributionalMonitorSettings,
    MonitorSettings,
    RunConfiguration,
    run_configuration_hash,
)
from llm_behavior_ci.experiments.faults import FaultError, apply_fault, load_fault
from llm_behavior_ci.experiments.run_config import RunConfigError, load_local_task_manifest
from llm_behavior_ci.experiments.schedule import (
    BenchmarkSchedule,
    ScheduleError,
    SimulatedClock,
    plan_arrivals,
    run_scheduled_monitor,
    schedule_hash,
)
from llm_behavior_ci.lifecycle.monitoring import (
    FrozenReference,
    MonitorRejected,
    ProductionMonitor,
    build_distributional_monitors,
    monitoring_period_id,
)
from llm_behavior_ci.runtime.episode import EpisodeRejected
from llm_behavior_ci.runtime.factory import LiveRuntimeFactory
from llm_behavior_ci.runtime.provenance import ProvenanceError, enforce_committed_provenance


def _under_results(path: Path) -> bool:
    return any(parent.name == "results" for parent in (path.resolve(), *path.resolve().parents))


def _load_json(path: Path) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ConfigError(f"{path} is not readable JSON") from error


def _frozen_reference(payload: object) -> FrozenReference:
    if not isinstance(payload, dict):
        raise ConfigError("frozen reference must be an object")
    pairs = tuple((str(name), float(value)) for name, value in payload.get("baselines", []))
    slices = tuple(
        (str(name), tuple((str(signal), float(value)) for signal, value in values))
        for name, values in payload.get("slice_baselines", [])
    )
    return FrozenReference(
        configuration_hash=str(payload.get("configuration_hash", "")),
        baselines=pairs,
        slice_baselines=slices,
        source=payload.get("source"),
    )


def main(argv: Sequence[str] | None = None) -> int:
    args_list = list(sys.argv[1:] if argv is None else argv)
    if not args_list:
        return 2
    parser = argparse.ArgumentParser(description="Rehearse the scheduled monitor on dev.")
    for name in (
        "--base",
        "--fault",
        "--task-set",
        "--schedule",
        "--monitor-settings",
        "--frozen-reference",
        "--state",
    ):
        parser.add_argument(name, required=True)
    parser.add_argument("--distributional-monitors", default=None)
    parser.add_argument("--reference-endpoint", default=None)
    parser.add_argument("--candidate-endpoint", default=None)
    try:
        args = parser.parse_args(args_list)
    except SystemExit as error:
        return 2 if error.code is None else int(error.code)
    state_path = Path(args.state)
    if _under_results(state_path):
        print("state must not be under results/", file=sys.stderr)
        return 2
    try:
        base = RunConfiguration.from_dict(_load_json(Path(args.base)))
        faulted = apply_fault(base, load_fault(Path(args.fault)))
        manifest = load_local_task_manifest(Path(args.task_set))
        if manifest.task_set.split != "dev":
            raise ConfigError("rehearsal runs on a dev task set only")
        schedule = BenchmarkSchedule.from_dict(_load_json(Path(args.schedule)))
        arrivals = plan_arrivals(manifest.task_set, schedule)
        enforce_committed_provenance(base)
        monitor_settings = MonitorSettings.from_dict(_load_json(Path(args.monitor_settings)))
        frozen = _frozen_reference(_load_json(Path(args.frozen_reference)))
        distributional_settings: tuple[DistributionalMonitorSettings, ...] = ()
        if args.distributional_monitors is not None:
            distributional_settings = tuple(
                DistributionalMonitorSettings.from_dict(item)
                for item in _load_json(Path(args.distributional_monitors))
            )
        factory = LiveRuntimeFactory.from_endpoints(
            reference=args.reference_endpoint,
            candidate=args.candidate_endpoint,
        )
        factory.preflight({"reference": base, "candidate": faulted})
        clock = SimulatedClock(schedule.clock_start)
        monitor = ProductionMonitor(
            monitor_settings,
            frozen,
            clock=clock,
            period_id=monitoring_period_id(run_configuration_hash(base), frozen.configuration_hash),
        )
        distributional = build_distributional_monitors(
            distributional_settings,
            reference_configuration_hash=frozen.configuration_hash,
            clock=clock,
            dedup_seconds=0.0,
        )
        state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.is_file() else {}

        def persist() -> None:
            state_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = state_path.with_suffix(state_path.suffix + ".tmp")
            temporary.write_text(json.dumps(state, sort_keys=True), encoding="utf-8")
            os.replace(temporary, state_path)

        def runtime_for(configuration: RunConfiguration) -> object:
            role = "reference" if configuration == base else "candidate"
            return factory(configuration, mode="execute", role=role)

        result = run_scheduled_monitor(
            schedule=schedule,
            arrivals=arrivals,
            healthy=base,
            faulted=faulted,
            runtime_for=runtime_for,
            monitor=monitor,
            distributional=distributional,
            clock=clock,
            state=state,
            persist=persist,
            difficulty_for=manifest.difficulty,
        )
        persist()
    except (
        ConfigError,
        FaultError,
        RunConfigError,
        ScheduleError,
        MonitorRejected,
        ProvenanceError,
        EpisodeRejected,
    ) as error:
        print(str(error) or "rehearsal failed", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "record": "dev_stream_rehearsal",
                "status": result.status,
                "schedule_hash": schedule_hash(schedule),
                "healthy_prefix_alarms": result.healthy_prefix_alarms,
                "post_onset_delay_episodes": result.post_onset_delay_episodes,
                "counters": result.counters.to_dict(),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
