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
    run_offline_gate,
)
from llm_behavior_ci.records import assert_public_payload, public_record_dict
from llm_behavior_ci.runtime.episode import RuntimeDependencies, build_runtime
from llm_behavior_ci.storage import EpisodeStore, StorageError
from llm_behavior_ci.tasks.plan_specs import PlanSpecError, task_plan_specs_from_mapping
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
    try:
        features = tuple(str(item) for item in payload["plan_quality_features"])
        weights = tuple(float(item) for item in payload["plan_quality_weights"])
        mmd_features = tuple(str(item) for item in payload["mmd_features"])
        required = tuple(str(item) for item in payload["required_statistics"])
        approximation = str(payload["kl_approximation"])
        if approximation not in {"full", "top_k"}:
            raise GateExecutionError("kl_approximation must be full or top_k")
        task_plan_specs = task_plan_specs_from_mapping(
            payload.get("task_plan_specs", [])
        )
        return PlanEvidenceInputs(
            plan_format_version=str(payload["plan_format_version"]),
            plan_quality_features=features,
            plan_quality_weights=weights,
            mmd_features=mmd_features,
            kl_approximation=approximation,  # type: ignore[arg-type]
            required_statistics=required,
            validation_provenance=str(payload["validation_provenance"]),
            task_plan_specs=task_plan_specs,
        )
    except (GateExecutionError, PlanSpecError, KeyError, TypeError, ValueError) as error:
        raise GateExecutionError("plan evidence is incomplete") from error


class _SwitchingAgent:
    def __init__(
        self,
        *,
        reference_agent: object,
        candidate_agent: object,
        reference_hash: str,
        candidate_hash: str,
    ) -> None:
        self._reference_agent = reference_agent
        self._candidate_agent = candidate_agent
        self._reference_hash = reference_hash
        self._candidate_hash = candidate_hash
        self._active = reference_agent

    def begin(self, context: object, config: RunConfiguration) -> None:
        from llm_behavior_ci.config import run_configuration_hash

        digest = run_configuration_hash(config)
        if digest == self._reference_hash:
            self._active = self._reference_agent
        elif digest == self._candidate_hash:
            self._active = self._candidate_agent
        else:
            raise GateExecutionError("live runtime configuration is not registered")
        self._active.begin(context, config)

    def messages(self, tool_output: str | None = None) -> object:
        return self._active.messages(tool_output=tool_output)

    def next_turn(self, *, tool_output: str | None) -> object:
        return self._active.next_turn(tool_output=tool_output)

    def teacher_force_plan(self, **kwargs: object) -> object:
        method = getattr(self._active, "teacher_force_plan", None)
        if not callable(method):
            raise AttributeError("teacher_force_plan")
        return method(**kwargs)

    def set_mode(self, mode: str) -> None:
        for agent in (self._reference_agent, self._candidate_agent):
            setter = getattr(agent, "set_mode", None)
            if callable(setter):
                setter(mode)


def _live_runtime(
    reference: RunConfiguration,
    candidate: RunConfiguration,
    *,
    reference_endpoint: str,
    candidate_endpoint: str,
) -> RuntimeDependencies:
    from llm_behavior_ci.config import run_configuration_hash

    if reference_endpoint.strip() == "" or candidate_endpoint.strip() == "":
        raise GateExecutionError("live runtime requires reference and candidate endpoints")
    if reference_endpoint == candidate_endpoint:
        raise GateExecutionError("live runtime endpoints must be distinct")
    reference_runtime = build_runtime(reference, reference_endpoint, mode="plan")
    candidate_runtime = build_runtime(candidate, candidate_endpoint, mode="plan")
    agent = _SwitchingAgent(
        reference_agent=reference_runtime.agent,
        candidate_agent=candidate_runtime.agent,
        reference_hash=run_configuration_hash(reference),
        candidate_hash=run_configuration_hash(candidate),
    )
    return RuntimeDependencies(
        session_factory=reference_runtime.session_factory,
        agent=agent,
        clock=reference_runtime.clock,
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
        if runtime is not None:
            if args.live_runtime:
                raise GateExecutionError(
                    "injected runtime cannot be combined with --live-runtime"
                )
            active_runtime = runtime
        elif args.live_runtime:
            if args.reference_endpoint is None or args.candidate_endpoint is None:
                raise GateExecutionError(
                    "--live-runtime requires --reference-endpoint and --candidate-endpoint"
                )
            active_runtime = _live_runtime(
                reference,
                candidate,
                reference_endpoint=args.reference_endpoint,
                candidate_endpoint=args.candidate_endpoint,
            )
        else:
            raise GateExecutionError(
                "runtime required: inject runtime for CPU or pass --live-runtime "
                "with --reference-endpoint and --candidate-endpoint"
            )
        decision = run_offline_gate(
            reference,
            candidate,
            task_set,
            settings=settings,
            runtime=active_runtime,
            plan_evidence=plan_evidence,
        )
        if args.episode_store is not None:
            store = EpisodeStore(Path(args.episode_store))
            try:
                store.append_validation_artifact(decision.artifact)
            finally:
                store.close()
    except (GateExecutionError, ConfigError, SelectionError, StorageError) as error:
        print(str(error) or "offline gate execution failed", file=sys.stderr)
        return 2
    except Exception:
        print("offline gate execution failed", file=sys.stderr)
        return 2

    print(json.dumps(_public_document(decision), sort_keys=True))
    return 0 if decision.outcome == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
