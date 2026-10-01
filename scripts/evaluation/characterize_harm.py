"""Characterize whether a declared fault is harmful on a frozen dev task set.

Requires --base, --candidate, --task-set, --fault, --margin,
--confidence-level, --resamples, and --seed. Runtime comes from
LLM_BEHAVIOR_CI_RUNTIME (module:function). Does not freeze a label under
results/. Bare invocation exits 2.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path

from llm_behavior_ci.config import ConfigError, RunConfiguration
from llm_behavior_ci.experiments.faults import (
    FaultError,
    load_fault,
    measure_harm,
)
from llm_behavior_ci.runtime.episode import RuntimeDependencies
from llm_behavior_ci.runtime.provenance import ProvenanceError, enforce_committed_provenance
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


def _load_json(path: Path) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise FaultError("input document is not readable JSON") from error


def _load_task_set(payload: object) -> TaskSet:
    if not isinstance(payload, dict):
        raise FaultError("task set must be an object")
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
        raise FaultError("task set is incomplete") from error
    digest = canonical_task_set_bytes(
        appworld_version=task_set.appworld_version,
        split=task_set.split,
        selection_rule=task_set.selection_rule,
        selection_seed=task_set.selection_seed,
        tasks=tuple(zip(task_set.task_ids, task_set.scenario_ids, strict=True)),
    )
    if task_set_hash_from_bytes(digest) != task_set.task_set_hash:
        raise FaultError("task_set_hash does not match the canonical digest")
    return task_set


def _load_runtime(spec: str) -> RuntimeDependencies:
    if ":" not in spec:
        raise FaultError("LLM_BEHAVIOR_CI_RUNTIME must be module:function")
    module_name, _, function_name = spec.partition(":")
    if not module_name or not function_name:
        raise FaultError("LLM_BEHAVIOR_CI_RUNTIME must be module:function")
    module_path = Path(module_name)
    try:
        if module_path.suffix == ".py" and module_path.is_file():
            loader = importlib.util.spec_from_file_location(
                "llm_behavior_ci_harm_runtime_factory",
                module_path,
            )
            if loader is None or loader.loader is None:
                raise FaultError("runtime module could not be loaded")
            module = importlib.util.module_from_spec(loader)
            loader.loader.exec_module(module)
        else:
            module = importlib.import_module(module_name)
        factory = getattr(module, function_name)
    except FaultError:
        raise
    except Exception as error:
        raise FaultError("runtime import failed") from error
    if not callable(factory):
        raise FaultError("runtime factory must be callable")
    runtime = factory()
    if not isinstance(runtime, RuntimeDependencies):
        raise FaultError("runtime factory must return RuntimeDependencies")
    return runtime


def main(argv: Sequence[str] | None = None) -> int:
    args_list = list(sys.argv[1:] if argv is None else argv)
    if not args_list:
        return 2
    parser = argparse.ArgumentParser(
        description="Measure harm for one declared fault without freezing under results/."
    )
    parser.add_argument("--base", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--task-set", required=True)
    parser.add_argument("--fault", required=True)
    parser.add_argument("--margin", required=True, type=float)
    parser.add_argument("--confidence-level", required=True, type=float)
    parser.add_argument("--resamples", required=True, type=int)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--output", default=None)
    try:
        args = parser.parse_args(args_list)
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

    output_path = None if args.output is None else Path(args.output)
    if output_path is not None and _under_results(output_path):
        print("output must not be under results/", file=sys.stderr)
        return 2

    try:
        base = RunConfiguration.from_dict(_load_json(Path(args.base)))
        candidate = RunConfiguration.from_dict(_load_json(Path(args.candidate)))
        task_set = _load_task_set(_load_json(Path(args.task_set)))
        fault = load_fault(Path(args.fault))
        enforce_committed_provenance(base)
        runtime = _load_runtime(runtime_spec)
        label = measure_harm(
            base,
            candidate,
            task_set,
            margin=float(args.margin),
            runtime=runtime,
            fault=fault,
            confidence_level=float(args.confidence_level),
            resamples=int(args.resamples),
            seed=int(args.seed),
        )
        document = {
            "record": "harm_label",
            "fault_version": label.fault_version,
            "base_configuration_hash": label.base_configuration_hash,
            "candidate_configuration_hash": label.candidate_configuration_hash,
            "task_set_hash": label.task_set_hash,
            "effect_estimate": label.effect_estimate,
            "interval_low": label.interval_low,
            "interval_high": label.interval_high,
            "margin": label.margin,
            "harmful": label.harmful,
            "split": label.split,
            "confidence_level": label.confidence_level,
            "resamples": label.resamples,
            "seed": label.seed,
        }
        text = json.dumps(document, sort_keys=True, separators=(",", ":"))
        if output_path is not None:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(text + "\n", encoding="utf-8")
        print(text)
        return 0
    except (FaultError, ConfigError, SelectionError, ProvenanceError) as error:
        print(str(error) or "harm characterization failed", file=sys.stderr)
        return 1
    except Exception:
        print("harm characterization failed", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
