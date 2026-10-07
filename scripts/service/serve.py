from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from llm_behavior_ci.config import (
    CanarySettings,
    ConfigError,
    DistributionalMonitorSettings,
    MonitorSettings,
    RunConfiguration,
    run_configuration_hash,
)
from llm_behavior_ci.experiments.protocol import (
    ProtocolError,
    TaskSelectionAllowance,
    bind_train_task_selection,
    task_selection_allowance_from_dict,
)
from llm_behavior_ci.experiments.run_config import (
    LocalTaskManifest,
    RunConfigError,
    load_local_task_manifest,
)
from llm_behavior_ci.lifecycle.monitoring import (
    FrozenReference,
    MonitorRejected,
    ProductionMonitor,
    TaskMetadata,
    build_distributional_monitors,
    monitoring_period_id,
)
from llm_behavior_ci.records import EpisodeResult
from llm_behavior_ci.runtime.factory import (
    ConfigurationRoutedRuntimeFactory,
    RuntimeFactoryError,
)
from llm_behavior_ci.service import (
    ConfigurationRegistry,
    ServiceDependencies,
    create_app,
)
from llm_behavior_ci.storage import EpisodeStore, StorageError


def _load_json(path: Path) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SystemExit(2) from error


def _refuse_results_store(path: Path) -> None:
    resolved = path.resolve()
    for parent in (resolved, *resolved.parents):
        if parent.name == "results":
            print("store path must not be under results/", file=sys.stderr)
            raise SystemExit(2)


def _baseline_pairs(raw: object, label: str) -> tuple[tuple[str, float], ...]:
    if not isinstance(raw, list):
        raise ConfigError(f"{label} must be a list")
    pairs: list[tuple[str, float]] = []
    for item in raw:
        if (
            not isinstance(item, (list, tuple))
            or len(item) != 2
            or not isinstance(item[0], str)
        ):
            raise ConfigError("each baseline must be [signal, estimate]")
        pairs.append((item[0], float(item[1])))
    return tuple(pairs)


def _frozen_reference(payload: object) -> FrozenReference:
    if not isinstance(payload, dict):
        raise ConfigError("frozen reference must be an object")
    configuration_hash = payload.get("configuration_hash")
    if not isinstance(configuration_hash, str) or configuration_hash == "":
        raise ConfigError("frozen reference configuration_hash is required")
    baselines = _baseline_pairs(payload.get("baselines"), "frozen reference baselines")
    raw_slices = payload.get("slice_baselines", [])
    if not isinstance(raw_slices, list):
        raise ConfigError("frozen reference slice_baselines must be a list")
    slices: list[tuple[str, tuple[tuple[str, float], ...]]] = []
    for item in raw_slices:
        if not isinstance(item, (list, tuple)) or len(item) != 2 or not isinstance(item[0], str):
            raise ConfigError("each slice baseline must be [slice, baselines]")
        slices.append((item[0], _baseline_pairs(item[1], "slice baselines")))
    source = payload.get("source")
    if source is not None and not isinstance(source, str):
        raise ConfigError("frozen reference source must be a string")
    try:
        return FrozenReference(
            configuration_hash=configuration_hash,
            baselines=baselines,
            slice_baselines=tuple(slices),
            source=source,
        )
    except MonitorRejected as error:
        raise ConfigError(str(error)) from error


def _runtime_factory_for(
    registry: ConfigurationRegistry,
    *,
    production_base_url: str | None,
    candidate_base_url: str | None,
) -> ConfigurationRoutedRuntimeFactory:
    """Route each registered configuration to its server, or to its provider.

    A hosted configuration takes no base URL; a vLLM one requires its own.
    Raises ``RuntimeFactoryError`` before the service starts when a URL or
    provider key is missing.
    """

    entries: dict[str, tuple[RunConfiguration, str | None]] = {
        "production": (registry.production, production_base_url),
    }
    if registry.candidate is not None:
        entries["candidate"] = (registry.candidate, candidate_base_url)
    return ConfigurationRoutedRuntimeFactory.for_configurations(entries)


def _train_to_dev_allowance(
    payload: object,
    registry: ConfigurationRegistry,
) -> TaskSelectionAllowance:
    """Validate a ``task-selection-allowance-v1`` document for dev serving.

    The allowance restores a train gate's task selection on dev serving
    configurations, so admission can rebuild and check the gate's own
    hashes. It must name ``task.split`` and ``task.task_set_hash`` with
    train values, there must be a candidate, and production and candidate
    must share one dev task binding. Every other hashed leaf is left to
    ``authorize_gated_candidate``, which refuses any difference from the
    gate.
    """

    try:
        allowance = task_selection_allowance_from_dict(payload)
    except ProtocolError as error:
        raise ConfigError(str(error)) from error
    required = {"task.split", "task.task_set_hash"}
    if not required <= allowance.allowed_leaves:
        raise ConfigError(
            "task selection allowance must name task.split and task.task_set_hash"
        )
    if allowance.train_values["task.split"] != "train":
        raise ConfigError("task selection allowance must restore the train split")
    if registry.candidate is None:
        raise ConfigError("task selection allowance requires --candidate-config")
    for label, configuration in (
        ("production", registry.production),
        ("candidate", registry.candidate),
    ):
        if configuration.task.split != "dev":
            raise ConfigError(
                f"task selection allowance serves dev only; {label} split is "
                f"{configuration.task.split}"
            )
        try:
            bind_train_task_selection(configuration, allowance)
        except ProtocolError as error:
            raise ConfigError(f"{label}: {error}") from error
    if registry.production.task != registry.candidate.task:
        raise ConfigError(
            "production and candidate must share one dev task binding"
        )
    return allowance


def _metadata_for_factory(
    manifest: LocalTaskManifest | None = None,
) -> Callable[[EpisodeResult], TaskMetadata]:
    """Completion-ordered monitor metadata, labeled by difficulty when known.

    With a local task manifest that records difficulty, every episode
    carries ``difficulty`` and a ``task_mix`` label ``difficulty:<n>``, the
    one per-task axis the split policy releases.
    """

    counter = {"index": 0}

    def metadata_for(episode: EpisodeResult) -> TaskMetadata:
        index = counter["index"]
        counter["index"] = index + 1
        difficulty = None if manifest is None else manifest.difficulty(episode.task.task_id)
        return TaskMetadata(
            signal="task_success",
            completion_index=index,
            difficulty=difficulty,
            task_mix=None if difficulty is None else f"difficulty:{difficulty}",
        )

    return metadata_for


def _build_dependencies(args: argparse.Namespace) -> ServiceDependencies:
    """Build the exact ``ServiceDependencies`` a live ``serve.py`` process runs.

    Pure with respect to the network: it reads the caller's config files
    and builds the monitor, distributional monitors, and runtime factory,
    but never starts ``uvicorn``. ``main`` calls this before creating the
    app so a caller (a test, or another entry point) can build the same
    dependencies and wrap them in ``create_app``/``TestClient`` directly.
    """

    production = RunConfiguration.from_dict(
        _load_json(Path(args.production_config))
    )
    candidate = None
    if args.candidate_config is not None:
        candidate = RunConfiguration.from_dict(
            _load_json(Path(args.candidate_config))
        )
    canary_settings = CanarySettings.from_dict(_load_json(Path(args.canary_settings)))
    monitor_settings = MonitorSettings.from_dict(
        _load_json(Path(args.monitor_settings))
    )
    frozen = _frozen_reference(_load_json(Path(args.frozen_reference)))
    if frozen.configuration_hash != run_configuration_hash(production):
        raise ConfigError(
            "frozen reference hash must match production configuration"
        )
    if monitor_settings.reference_configuration_hash != frozen.configuration_hash:
        raise ConfigError(
            "monitor settings reference hash must match frozen reference"
        )
    registry = ConfigurationRegistry(
        production=production,
        candidate=candidate,
    )
    allowance = None
    allowance_path = getattr(args, "task_selection_allowance", None)
    if allowance_path is not None:
        allowance = _train_to_dev_allowance(
            _load_json(Path(allowance_path)), registry
        )
    try:
        runtime_factory = _runtime_factory_for(
            registry,
            production_base_url=getattr(args, "production_base_url", None),
            candidate_base_url=getattr(args, "candidate_base_url", None),
        )
    except RuntimeFactoryError as error:
        raise ConfigError(str(error)) from error
    manifest = None
    task_metadata_path = getattr(args, "task_metadata", None)
    if task_metadata_path is not None:
        try:
            manifest = load_local_task_manifest(Path(task_metadata_path))
        except RunConfigError as error:
            raise ConfigError(str(error)) from error
        if manifest.task.task_set_hash != production.task.task_set_hash:
            raise ConfigError("task metadata must describe the production task set")
    period_id = monitoring_period_id(
        run_configuration_hash(production), frozen.configuration_hash
    )
    store = EpisodeStore(Path(args.store))
    clock = lambda: datetime.now(timezone.utc)
    monitor = ProductionMonitor(
        monitor_settings,
        frozen,
        clock=clock,
        dedup_seconds=float(args.dedup_seconds),
        period_id=period_id,
        use_slice_attribution=bool(args.slice_attribution),
    )
    distributional_settings: tuple[DistributionalMonitorSettings, ...] = ()
    if args.distributional_monitors is not None:
        raw_distributional = _load_json(Path(args.distributional_monitors))
        if not isinstance(raw_distributional, list):
            raise ConfigError("distributional monitors document must be a list")
        distributional_settings = tuple(
            DistributionalMonitorSettings.from_dict(item)
            for item in raw_distributional
        )
    distributional_monitors = build_distributional_monitors(
        distributional_settings,
        reference_configuration_hash=frozen.configuration_hash,
        clock=clock,
        dedup_seconds=float(args.dedup_seconds),
        period_id=period_id,
    )
    return ServiceDependencies(
        registry=registry,
        runtime_factory=runtime_factory,
        store=store,
        monitor=monitor,
        clock=clock,
        max_in_flight=int(args.max_in_flight),
        shutdown_timeout_seconds=float(args.shutdown_timeout_seconds),
        metadata_for=_metadata_for_factory(manifest),
        canary_settings=canary_settings,
        canary_assignment_seed=int(args.canary_assignment_seed),
        task_selection_allowance=allowance,
        tool_selection_monitor=distributional_monitors.get("tool_selection"),
        task_mix_monitor=distributional_monitors.get("task_mix"),
    )


def main(argv: Sequence[str] | None = None) -> int:
    if argv is None and len(sys.argv) <= 1:
        return 2
    parser = argparse.ArgumentParser(
        description=(
            "Serve the llm-behavior-ci gateway. Requires configuration files "
            "and the canary assignment seed. A vLLM configuration needs its "
            "own base URL; a hosted one takes none and reads its key from the "
            "environment. Does not import vLLM."
        )
    )
    parser.add_argument("--production-config", required=True)
    parser.add_argument("--candidate-config", default=None)
    parser.add_argument("--store", required=True)
    parser.add_argument("--max-in-flight", required=True, type=int)
    parser.add_argument("--shutdown-timeout-seconds", required=True, type=float)
    parser.add_argument("--canary-settings", required=True)
    parser.add_argument("--monitor-settings", required=True)
    parser.add_argument("--frozen-reference", required=True)
    parser.add_argument("--dedup-seconds", required=True, type=float)
    parser.add_argument("--canary-assignment-seed", required=True, type=int)
    parser.add_argument("--production-base-url", default=None)
    parser.add_argument("--candidate-base-url", default=None)
    parser.add_argument("--task-metadata", default=None)
    parser.add_argument("--distributional-monitors", default=None)
    parser.add_argument("--slice-attribution", action="store_true")
    parser.add_argument("--task-selection-allowance", default=None)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8000, type=int)
    try:
        args = parser.parse_args(list(argv) if argv is not None else None)
    except SystemExit as error:
        code = error.code
        if code is None:
            return 2
        return int(code)

    production_path = Path(args.production_config)
    store_path = Path(args.store)
    canary_path = Path(args.canary_settings)
    monitor_path = Path(args.monitor_settings)
    frozen_path = Path(args.frozen_reference)
    required_paths = [production_path, canary_path, monitor_path, frozen_path]
    if args.candidate_config is not None:
        required_paths.append(Path(args.candidate_config))
    if args.distributional_monitors is not None:
        required_paths.append(Path(args.distributional_monitors))
    if args.task_metadata is not None:
        required_paths.append(Path(args.task_metadata))
    if args.task_selection_allowance is not None:
        required_paths.append(Path(args.task_selection_allowance))
    for path in required_paths:
        if not path.is_file():
            print(f"missing file: {path}", file=sys.stderr)
            return 2
    _refuse_results_store(store_path)

    try:
        dependencies = _build_dependencies(args)
        app = create_app(dependencies)
    except (ConfigError, StorageError, SystemExit) as error:
        if isinstance(error, SystemExit):
            code = error.code
            return 2 if code is None else int(code)
        print(str(error) or "service startup failed", file=sys.stderr)
        return 2
    except Exception:
        print("service startup failed", file=sys.stderr)
        return 2

    import uvicorn

    uvicorn.run(app, host=args.host, port=int(args.port))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
