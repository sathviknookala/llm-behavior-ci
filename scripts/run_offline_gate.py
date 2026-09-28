from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path

from llm_behavior_ci.config import ConfigError, GateSettings, RunConfiguration
from llm_behavior_ci.lifecycle.offline_gate import (
    GateDecision,
    GateExecutionError,
    run_offline_gate,
)
from llm_behavior_ci.records import assert_public_payload, public_record_dict
from llm_behavior_ci.runtime.episode import RuntimeDependencies
from llm_behavior_ci.tasks.selection import (
    SelectionError,
    TaskSet,
    canonical_task_set_bytes,
    task_set_hash_from_bytes,
)


def _load_json(raw: str) -> object:
    path = Path(raw)
    if path.is_file():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise GateExecutionError("input is not readable JSON") from error
    try:
        return json.loads(raw)
    except json.JSONDecodeError as error:
        raise GateExecutionError("input is not readable JSON") from error


def _load_task_set(payload: object) -> TaskSet:
    if not isinstance(payload, dict):
        raise GateExecutionError("task set must be an object")
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
        raise GateExecutionError("task set is incomplete") from error
    digest = canonical_task_set_bytes(
        appworld_version=task_set.appworld_version,
        split=task_set.split,
        selection_rule=task_set.selection_rule,
        selection_seed=task_set.selection_seed,
        tasks=tuple(zip(task_set.task_ids, task_set.scenario_ids, strict=True)),
    )
    if task_set_hash_from_bytes(digest) != task_set.task_set_hash:
        raise GateExecutionError("task_set_hash does not match the canonical digest")
    return task_set


def _live_runtime() -> RuntimeDependencies:
    base_url = os.environ.get("LLM_BEHAVIOR_CI_VLLM_BASE_URL")
    if not base_url:
        raise GateExecutionError("live runtime requires LLM_BEHAVIOR_CI_VLLM_BASE_URL")
    from llm_behavior_ci.runtime.agent import VLLMAgent
    from llm_behavior_ci.runtime.appworld import LiveAppWorldSession

    agent = VLLMAgent(base_url)
    agent.set_mode("plan")
    return RuntimeDependencies(
        session_factory=LiveAppWorldSession,
        agent=agent,
        clock=lambda: datetime.now(timezone.utc),
    )


def _public_document(decision: GateDecision) -> dict[str, object]:
    document: dict[str, object] = {
        "outcome": decision.outcome,
        "reason_codes": list(decision.reason_codes),
        "reference_configuration_hash": decision.reference_configuration_hash,
        "candidate_configuration_hash": decision.candidate_configuration_hash,
        "task_set_hash": decision.task_set_hash,
        "reference_protocol_hash": decision.reference_protocol_hash,
        "candidate_protocol_hash": decision.candidate_protocol_hash,
        "thresholds": decision.thresholds.to_dict(),
        "statistics": [
            public_record_dict(item) for item in decision.statistics
        ],
        "public_decision": (
            public_record_dict(decision.public_decision)
            if decision.public_decision is not None
            else None
        ),
    }
    assert_public_payload(document)
    return document


def main(
    argv: Sequence[str] | None = None,
    *,
    runtime: RuntimeDependencies | None = None,
) -> int:
    parser = argparse.ArgumentParser(description="Run the plan-only offline gate.")
    parser.add_argument("--reference", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--task-set", required=True)
    parser.add_argument("--settings", required=True)
    try:
        args = parser.parse_args(list(argv) if argv is not None else None)
    except SystemExit as error:
        code = error.code
        if code is None:
            return 2
        return int(code)

    try:
        reference = RunConfiguration.from_dict(_load_json(args.reference))
        candidate = RunConfiguration.from_dict(_load_json(args.candidate))
        task_set = _load_task_set(_load_json(args.task_set))
        settings = GateSettings.from_dict(_load_json(args.settings))
        active_runtime = runtime if runtime is not None else _live_runtime()
        decision = run_offline_gate(
            reference,
            candidate,
            task_set,
            settings=settings,
            runtime=active_runtime,
        )
    except (GateExecutionError, ConfigError, SelectionError) as error:
        print(str(error) or "offline gate execution failed", file=sys.stderr)
        return 2
    except Exception:
        print("offline gate execution failed", file=sys.stderr)
        return 2

    print(json.dumps(_public_document(decision), sort_keys=True))
    return 0 if decision.outcome == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
