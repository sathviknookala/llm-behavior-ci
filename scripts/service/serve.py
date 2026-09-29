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
from llm_behavior_ci.lifecycle.monitoring import (
    FrozenReference,
    ProductionMonitor,
    TaskMetadata,
    build_distributional_monitors,
)
from llm_behavior_ci.records import EpisodeResult
from llm_behavior_ci.runtime.episode import (
    RuntimeDependencies,
    RuntimeUnavailable,
    build_runtime,
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


def _frozen_reference(payload: object) -> FrozenReference:
    if not isinstance(payload, dict):
        raise ConfigError("frozen reference must be an object")
    configuration_hash = payload.get("configuration_hash")
    baselines = payload.get("baselines")
    if not isinstance(configuration_hash, str) or configuration_hash == "":
        raise ConfigError("frozen reference configuration_hash is required")
    if not isinstance(baselines, list):
        raise ConfigError("frozen reference baselines must be a list")
    pairs: list[tuple[str, float]] = []
    for item in baselines:
        if (
            not isinstance(item, (list, tuple))
            or len(item) != 2
            or not isinstance(item[0], str)
        ):
            raise ConfigError("each baseline must be [signal, estimate]")
        pairs.append((item[0], float(item[1])))
    return FrozenReference(
        configuration_hash=configuration_hash,
        baselines=tuple(pairs),
    )


def _runtime_factory_for(
    registry: ConfigurationRegistry,
    *,
    production_base_url: str,
    candidate_base_url: str,
) -> Callable[[RunConfiguration], RuntimeDependencies]:
    production_hash = run_configuration_hash(registry.production)
    candidate_hash = (
        None
        if registry.candidate is None
        else run_configuration_hash(registry.candidate)
    )

    def runtime_factory(config: RunConfiguration) -> RuntimeDependencies:
        digest = run_configuration_hash(config)
        if digest == production_hash:
            url = production_base_url
        elif candidate_hash is not None and digest == candidate_hash:
            url = candidate_base_url
        else:
            raise RuntimeUnavailable("configuration is not in the registry")
        if url == "":
            raise RuntimeUnavailable("base url is not configured for this role")
        return build_runtime(config, url, mode="execute")

    return runtime_factory


def _metadata_for_factory() -> Callable[[EpisodeResult], TaskMetadata]:
    counter = {"index": 0}

    def metadata_for(episode: EpisodeResult) -> TaskMetadata:
        del episode
        index = counter["index"]
        counter["index"] = index + 1
        return TaskMetadata(signal="task_success", completion_index=index)

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
    store = EpisodeStore(Path(args.store))
    clock = lambda: datetime.now(timezone.utc)
    monitor = ProductionMonitor(
        monitor_settings,
        frozen,
        clock=clock,
        dedup_seconds=float(args.dedup_seconds),
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
    )
    registry = ConfigurationRegistry(
        production=production,
        candidate=candidate,
    )
    return ServiceDependencies(
        registry=registry,
        runtime_factory=_runtime_factory_for(
            registry,
            production_base_url=args.production_base_url,
            candidate_base_url=args.candidate_base_url,
        ),
        store=store,
        monitor=monitor,
        clock=clock,
        max_in_flight=int(args.max_in_flight),
        shutdown_timeout_seconds=float(args.shutdown_timeout_seconds),
        metadata_for=_metadata_for_factory(),
        canary_settings=canary_settings,
        canary_assignment_seed=int(args.canary_assignment_seed),
        tool_selection_monitor=distributional_monitors.get("tool_selection"),
        task_mix_monitor=distributional_monitors.get("task_mix"),
    )


def main(argv: Sequence[str] | None = None) -> int:
    if argv is None and len(sys.argv) <= 1:
        return 2
    parser = argparse.ArgumentParser(
        description=(
            "Serve the llm-behavior-ci gateway. Requires configuration files, "
            "canary assignment seed, and distinct production and candidate base URLs. "
            "Does not import vLLM."
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
    parser.add_argument("--production-base-url", required=True)
    parser.add_argument("--candidate-base-url", required=True)
    parser.add_argument("--distributional-monitors", default=None)
    parser.add_argument("--slice-attribution", action="store_true")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8000, type=int)
    try:
        args = parser.parse_args(list(argv) if argv is not None else None)
    except SystemExit as error:
        code = error.code
        if code is None:
            return 2
        return int(code)

    if args.production_base_url.strip() == "" or args.candidate_base_url.strip() == "":
        print("production and candidate base URLs are required", file=sys.stderr)
        return 2
    if args.production_base_url == args.candidate_base_url:
        print(
            "production and candidate base URLs must be distinct",
            file=sys.stderr,
        )
        return 2

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
