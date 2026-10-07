"""Run the plan-only offline gate.

Requires --reference, --candidate, --task-set, --settings, and
--plan-evidence. --live-runtime builds one plan-mode runtime per role: a
vLLM role needs --reference-endpoint or --candidate-endpoint, a hosted
role takes none and reads its key from the environment. A required
``kl`` statistic on a hosted configuration fails in preflight. With
--episode-store the validation artifact is recorded for later admission.
Bare invocation exits 2.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from llm_behavior_ci.config import ConfigError, GateSettings, RunConfiguration
from llm_behavior_ci.lifecycle.offline_gate import (
    GateDecision,
    GateExecutionError,
    PlanEvidenceInputs,
    plan_evidence_from_dict,
    require_gate_capabilities,
    run_offline_gate,
)
from llm_behavior_ci.records import assert_public_payload, public_record_dict
from llm_behavior_ci.runtime.episode import EpisodeRejected, RuntimeDependencies
from llm_behavior_ci.runtime.factory import (
    LiveRuntimeFactory,
    RuntimeFactory,
    RuntimeFactoryError,
)
from llm_behavior_ci.runtime.provenance import ProvenanceError, enforce_committed_provenance
from llm_behavior_ci.storage import EpisodeStore, StorageError
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


def _load_plan_evidence(payload: object) -> PlanEvidenceInputs:
    if not isinstance(payload, dict):
        raise GateExecutionError("plan evidence must be an object")
    return plan_evidence_from_dict(payload)


def _role_runtimes(
    reference: RunConfiguration,
    candidate: RunConfiguration,
    factory: RuntimeFactory,
) -> tuple[RuntimeDependencies, RuntimeDependencies]:
    """Separate plan-mode runtimes for the two roles, after preflight.

    A vLLM role needs its own endpoint; two vLLM roles with different
    configurations need distinct endpoints. A hosted role takes no endpoint
    and needs its key in the environment. Identical configurations still
    get two agents, so the roles never share history.
    """

    factory.preflight({"reference": reference, "candidate": candidate})
    return (
        factory(reference, mode="plan", role="reference"),
        factory(candidate, mode="plan", role="candidate"),
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
        "validation_provenance": decision.validation_provenance,
        "artifact": public_record_dict(decision.artifact),
        "artifact_id": decision.artifact.artifact_id,
    }
    assert_public_payload(document)
    return document


def main(
    argv: Sequence[str] | None = None,
    *,
    runtime: RuntimeDependencies | None = None,
    runtime_factory: RuntimeFactory | None = None,
) -> int:
    if argv is None and len(sys.argv) <= 1:
        return 2
    parser = argparse.ArgumentParser(
        description=(
            "Run the plan-only offline gate. CPU use injects runtime in-process. "
            "Live endpoints require --live-runtime."
        )
    )
    parser.add_argument("--reference", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--task-set", required=True)
    parser.add_argument("--settings", required=True)
    parser.add_argument("--plan-evidence", required=True)
    parser.add_argument("--live-runtime", action="store_true")
    parser.add_argument("--reference-endpoint", default=None)
    parser.add_argument("--candidate-endpoint", default=None)
    parser.add_argument(
        "--episode-store",
        default=None,
        help=(
            "SQLite episode log to record the validation artifact into. "
            "Candidate admission (POST /candidates) looks the artifact up "
            "by id from this same store; without --episode-store the gate "
            "still prints artifact_id, but no admission call can find it."
        ),
    )
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
        plan_evidence = _load_plan_evidence(_load_json(args.plan_evidence))
        require_gate_capabilities(
            reference, candidate, plan_evidence.required_statistics
        )
        candidate_runtime: RuntimeDependencies | None = None
        if runtime is not None or runtime_factory is not None:
            if args.live_runtime:
                raise GateExecutionError(
                    "injected runtime cannot be combined with --live-runtime"
                )
            if runtime_factory is not None:
                active_runtime, candidate_runtime = _role_runtimes(
                    reference, candidate, runtime_factory
                )
            else:
                active_runtime = runtime
        elif args.live_runtime:
            enforce_committed_provenance(reference, candidate)
            factory = LiveRuntimeFactory.from_endpoints(
                reference=args.reference_endpoint,
                candidate=args.candidate_endpoint,
            )
            try:
                active_runtime, candidate_runtime = _role_runtimes(
                    reference, candidate, factory
                )
            except RuntimeFactoryError as error:
                raise GateExecutionError(str(error)) from error
        else:
            raise GateExecutionError(
                "runtime required: inject runtime for CPU or pass --live-runtime "
                "(vLLM roles also need --reference-endpoint/--candidate-endpoint)"
            )
        decision = run_offline_gate(
            reference,
            candidate,
            task_set,
            settings=settings,
            runtime=active_runtime,
            plan_evidence=plan_evidence,
            candidate_runtime=candidate_runtime,
        )
        if args.episode_store is not None:
            store = EpisodeStore(Path(args.episode_store))
            try:
                store.append_validation_artifact(decision.artifact)
            finally:
                store.close()
    except (
        GateExecutionError,
        ConfigError,
        SelectionError,
        StorageError,
        ProvenanceError,
        EpisodeRejected,
    ) as error:
        print(str(error) or "offline gate execution failed", file=sys.stderr)
        return 2
    except Exception:
        print("offline gate execution failed", file=sys.stderr)
        return 2

    print(json.dumps(_public_document(decision), sort_keys=True))
    return 0 if decision.outcome == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
