from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path

from llm_behavior_ci.config import ConfigError
from llm_behavior_ci.experiments.benchmark import (
    BenchmarkError,
    run_lifecycle_benchmark,
)
from llm_behavior_ci.experiments.faults import FaultError, load_fault
from llm_behavior_ci.experiments.protocol import ProtocolError, require_protocol_lock
from llm_behavior_ci.lifecycle.monitoring import FrozenReference
from llm_behavior_ci.lifecycle.offline_gate import (
    GateExecutionError,
    PlanEvidenceInputs,
    plan_evidence_from_dict,
)
from llm_behavior_ci.runtime.episode import RuntimeDependencies
from llm_behavior_ci.tasks.selection import (
    SelectionError,
    TaskSet,
    canonical_task_set_bytes,
    task_set_hash_from_bytes,
)


def _under_results(path: Path) -> bool:
    for parent in path.resolve().parents:
        if parent.name == "results":
            return True
    return False


def _refuse_results_path(path: Path, label: str) -> None:
    if _under_results(path):
        print(f"{label} must not be under results/", file=sys.stderr)
        raise SystemExit(2)


def _load_json(path: Path) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise BenchmarkError(f"unable to read {path}") from error


def _load_task_set(payload: object) -> TaskSet:
    if not isinstance(payload, dict):
        raise BenchmarkError("task set must be an object")
    try:
        task_ids = tuple(payload["task_ids"])
        scenario_ids = tuple(payload["scenario_ids"])
        task_set = TaskSet(
            appworld_version=str(payload["appworld_version"]),
            split=str(payload["split"]),
            selection_rule=str(payload["selection_rule"]),
            selection_seed=int(payload["selection_seed"]),
            task_count=int(payload["task_count"]),
            scenario_count=int(payload["scenario_count"]),
            task_ids=task_ids,
            scenario_ids=scenario_ids,
            task_set_hash=str(payload["task_set_hash"]),
        )
    except (KeyError, TypeError, ValueError, SelectionError) as error:
        raise BenchmarkError("task set is incomplete") from error
    digest = canonical_task_set_bytes(
        appworld_version=task_set.appworld_version,
        split=task_set.split,
        selection_rule=task_set.selection_rule,
        selection_seed=task_set.selection_seed,
        tasks=tuple(zip(task_set.task_ids, task_set.scenario_ids, strict=True)),
    )
    if task_set_hash_from_bytes(digest) != task_set.task_set_hash:
        raise BenchmarkError("task_set_hash does not match the canonical digest")
    return task_set


def _load_plan_evidence(payload: object) -> PlanEvidenceInputs:
    if not isinstance(payload, dict):
        raise BenchmarkError("plan evidence must be an object")
    try:
        return plan_evidence_from_dict(payload)
    except GateExecutionError as error:
        raise BenchmarkError("plan evidence is incomplete") from error


def _frozen_reference(payload: object) -> FrozenReference:
    if not isinstance(payload, dict):
        raise BenchmarkError("baselines must be an object")
    configuration_hash = payload.get("configuration_hash")
    baselines = payload.get("baselines")
    if not isinstance(configuration_hash, str) or configuration_hash == "":
        raise BenchmarkError("baselines configuration_hash is required")
    if not isinstance(baselines, list):
        raise BenchmarkError("baselines must be a list")
    pairs: list[tuple[str, float]] = []
    for item in baselines:
        if (
            not isinstance(item, (list, tuple))
            or len(item) != 2
            or not isinstance(item[0], str)
        ):
            raise BenchmarkError("each baseline must be [signal, estimate]")
        pairs.append((item[0], float(item[1])))
    return FrozenReference(
        configuration_hash=configuration_hash,
        baselines=tuple(pairs),
    )


def _load_runtime(spec: str) -> RuntimeDependencies:
    if ":" not in spec:
        raise BenchmarkError("LLM_BEHAVIOR_CI_RUNTIME must be module:function")
    module_name, _, function_name = spec.partition(":")
    if not module_name or not function_name:
        raise BenchmarkError("LLM_BEHAVIOR_CI_RUNTIME must be module:function")
    module_path = Path(module_name)
    try:
        if module_path.suffix == ".py" and module_path.is_file():
            loader = importlib.util.spec_from_file_location(
                "llm_behavior_ci_runtime_factory",
                module_path,
            )
            if loader is None or loader.loader is None:
                raise BenchmarkError("runtime module could not be loaded")
            module = importlib.util.module_from_spec(loader)
            loader.loader.exec_module(module)
        else:
            module = importlib.import_module(module_name)
        factory = getattr(module, function_name)
    except BenchmarkError:
        raise
    except Exception as error:
        raise BenchmarkError("runtime import failed") from error
    if not callable(factory):
        raise BenchmarkError("runtime factory must be callable")
    runtime = factory()
    if not isinstance(runtime, RuntimeDependencies):
        raise BenchmarkError("runtime factory must return RuntimeDependencies")
    return runtime


def main(argv: Sequence[str] | None = None) -> int:
    if argv is None and len(sys.argv) <= 1:
        return 2
    parser = argparse.ArgumentParser(description="Run the lifecycle benchmark.")
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--fault", action="append", dest="faults", required=True)
    parser.add_argument("--train-tasks", required=True)
    parser.add_argument("--test-normal-tasks", required=True)
    parser.add_argument("--baselines", required=True)
    parser.add_argument("--plan-evidence", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--export", default=None)
    try:
        args = parser.parse_args(list(argv) if argv is not None else None)
    except SystemExit as error:
        code = error.code
        if code is None:
            return 2
        return int(code)

    runtime_spec = os.environ.get("LLM_BEHAVIOR_CI_RUNTIME")
    if runtime_spec is None or runtime_spec.strip() == "":
        print(
            "LLM_BEHAVIOR_CI_RUNTIME is required (module:function returning RuntimeDependencies)",
            file=sys.stderr,
        )
        return 2

    checkpoint = Path(args.checkpoint)
    export_path = None if args.export is None else Path(args.export)
    try:
        _refuse_results_path(checkpoint, "checkpoint")
        if export_path is not None:
            _refuse_results_path(export_path, "export")
    except SystemExit as error:
        return int(error.code)

    try:
        protocol = require_protocol_lock(Path(args.protocol))
        faults = tuple(load_fault(Path(path)) for path in args.faults)
        train_tasks = _load_task_set(_load_json(Path(args.train_tasks)))
        test_normal_tasks = _load_task_set(_load_json(Path(args.test_normal_tasks)))
        baselines = _frozen_reference(_load_json(Path(args.baselines)))
        plan_evidence = _load_plan_evidence(_load_json(Path(args.plan_evidence)))
        if plan_evidence.validation_provenance == "synthetic_fixture":
            admission_mode = "test"
        elif plan_evidence.validation_provenance == "validated":
            admission_mode = "release"
        else:
            raise BenchmarkError(
                "plan evidence validation_provenance must be synthetic_fixture or validated"
            )
        runtime = _load_runtime(runtime_spec)
        result = run_lifecycle_benchmark(
            protocol,
            faults,
            runtime=runtime,
            train_tasks=train_tasks,
            test_normal_tasks=test_normal_tasks,
            reference_baselines=baselines,
            checkpoint_path=checkpoint,
            plan_evidence=plan_evidence,
            admission_mode=admission_mode,
            export_path=export_path,
        )
    except SystemExit as error:
        code = error.code
        if code is None:
            return 2
        return int(code)
    except (
        BenchmarkError,
        ProtocolError,
        FaultError,
        ConfigError,
        SelectionError,
        GateExecutionError,
    ) as error:
        print(str(error) or "lifecycle benchmark failed", file=sys.stderr)
        return 1
    except Exception:
        print("lifecycle benchmark failed", file=sys.stderr)
        return 1

    print(
        json.dumps(
            {
                "status": result.status,
                "agent_execution_seconds": result.agent_execution_seconds,
                "detector_compute_seconds": result.detector_compute_seconds,
                "wall_seconds": result.wall_seconds,
                "gpu_memory_mib": result.gpu_memory_mib,
                "gpu_hours": result.gpu_hours,
            },
            sort_keys=True,
        )
    )
    return 0 if result.status == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
