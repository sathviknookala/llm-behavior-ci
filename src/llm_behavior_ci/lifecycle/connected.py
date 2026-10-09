"""One dev release lifecycle: train gate, admission, paired canary, production monitoring.

``run_three_tier_dev`` drives the existing pieces in order and adds no
decision logic of its own. Tier 1 is ``run_offline_gate`` on a train task
set, and its ``ValidationArtifact`` is written into the service's own
SQLite store. A BLOCK stops there. Admission is ``POST /candidates`` with
that artifact id, so the service's release or test admission, its
task-selection allowance, and its hash checks decide. Tier 2 sends dev
arrivals to ``POST /episodes`` until the service's ``CanaryController``
reaches PROMOTED or ROLLED_BACK. Tier 3 sends the remaining arrivals as
ordinary traffic, which the service routes to whatever it now serves and
feeds to its production monitor. The summary is read back from the store
and ``GET /deployment``, never from the runner's own bookkeeping.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from llm_behavior_ci.config import GateSettings, RunConfiguration, run_configuration_hash
from llm_behavior_ci.lifecycle.offline_gate import (
    GateDecision,
    PlanEvidenceInputs,
    require_gate_capabilities,
    run_offline_gate,
)
from llm_behavior_ci.records import assert_public_payload
from llm_behavior_ci.runtime.factory import RuntimeFactory
from llm_behavior_ci.storage import EpisodeStore
from llm_behavior_ci.tasks.selection import TaskSet

_TERMINAL = frozenset({"PROMOTED", "ROLLED_BACK"})
_STATUS = {"PROMOTED": "promoted", "ROLLED_BACK": "rolled_back"}


class ConnectedLifecycleError(RuntimeError):
    """The lifecycle cannot continue; no later stage ran."""


class ServiceClient(Protocol):
    def get(self, url: str) -> Any: ...

    def post(self, url: str, *, json: object) -> Any: ...


@dataclass(frozen=True)
class GateInputs:
    reference: RunConfiguration
    candidate: RunConfiguration
    task_set: TaskSet
    settings: GateSettings
    plan_evidence: PlanEvidenceInputs
    runtime_factory: RuntimeFactory

    def __post_init__(self) -> None:
        if not isinstance(self.runtime_factory, RuntimeFactory):
            raise ConnectedLifecycleError("gate runtime_factory must be a RuntimeFactory")
        for label, configuration in (
            ("reference", self.reference),
            ("candidate", self.candidate),
        ):
            if configuration.task.split != "train":
                raise ConnectedLifecycleError(f"gate {label} must run on train")
        if self.task_set.split != "train":
            raise ConnectedLifecycleError("gate task set must be train")


@dataclass(frozen=True)
class DevTraffic:
    production: RunConfiguration
    candidate: RunConfiguration
    canary_task_ids: tuple[str, ...]
    production_task_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        for label, configuration in (
            ("production", self.production),
            ("candidate", self.candidate),
        ):
            if configuration.task.split != "dev":
                raise ConnectedLifecycleError(f"{label} must serve dev")
        for label, task_ids in (
            ("canary_task_ids", self.canary_task_ids),
            ("production_task_ids", self.production_task_ids),
        ):
            if not isinstance(task_ids, tuple) or not all(
                isinstance(task_id, str) and task_id != "" for task_id in task_ids
            ):
                raise ConnectedLifecycleError(f"{label} must be a tuple of task ids")
        if not self.canary_task_ids:
            raise ConnectedLifecycleError("canary_task_ids must not be empty")


def _json(response: Any, label: str) -> dict[str, object]:
    if response.status_code != 200:
        raise ConnectedLifecycleError(
            f"{label} returned {response.status_code}: {_detail(response)}"
        )
    body = response.json()
    if not isinstance(body, dict):
        raise ConnectedLifecycleError(f"{label} did not return an object")
    return body


def _detail(response: Any) -> str:
    try:
        body = response.json()
    except ValueError:
        return ""
    if isinstance(body, dict):
        return str(body.get("detail", ""))
    return ""


def _require_fresh_store(store_path: Path) -> None:
    reader = EpisodeStore(store_path)
    try:
        if reader.load_deployment_decisions() or reader.load_finished_episodes():
            raise ConnectedLifecycleError(
                "store already holds deployment decisions or episodes; "
                "one lifecycle needs its own store"
            )
    finally:
        reader.close()


def _require_registered(service: ServiceClient, traffic: DevTraffic) -> dict[str, object]:
    deployment = _json(service.get("/deployment"), "GET /deployment")
    if deployment.get("production_configuration_hash") != run_configuration_hash(
        traffic.production
    ):
        raise ConnectedLifecycleError("service production configuration does not match")
    if deployment.get("candidate_configuration_hash") != run_configuration_hash(
        traffic.candidate
    ):
        raise ConnectedLifecycleError("service candidate configuration does not match")
    if "state" in deployment:
        raise ConnectedLifecycleError("service already has a canary controller")
    return deployment


def _run_gate(gate: GateInputs) -> GateDecision:
    require_gate_capabilities(
        gate.reference, gate.candidate, gate.plan_evidence.required_statistics
    )
    factory = gate.runtime_factory
    factory.preflight({"reference": gate.reference, "candidate": gate.candidate})
    return run_offline_gate(
        gate.reference,
        gate.candidate,
        gate.task_set,
        settings=gate.settings,
        runtime=factory(gate.reference, mode="plan", role="reference"),
        plan_evidence=gate.plan_evidence,
        candidate_runtime=factory(gate.candidate, mode="plan", role="candidate"),
    )


def _persist_artifact(store_path: Path, decision: GateDecision) -> None:
    writer = EpisodeStore(store_path)
    try:
        writer.append_validation_artifact(decision.artifact)
    finally:
        writer.close()


def _tier1(decision: GateDecision) -> dict[str, object]:
    return {
        "outcome": decision.outcome,
        "reason_codes": list(decision.reason_codes),
        "artifact_id": decision.artifact.artifact_id,
        "evidence_source": decision.artifact.evidence_source,
        "reference_configuration_hash": decision.reference_configuration_hash,
        "candidate_configuration_hash": decision.candidate_configuration_hash,
        "task_set_hash": decision.task_set_hash,
    }


def _episode(service: ServiceClient, task_id: str, key: str) -> dict[str, object]:
    return _json(
        service.post(
            "/episodes",
            json={"task_id": task_id, "mode": "execute", "assignment_key": key},
        ),
        "POST /episodes",
    )


def _receipts(receipts: Sequence[dict[str, object]]) -> dict[str, object]:
    roles = Counter(str(item["role"]) for item in receipts)
    hashes = Counter(str(item["configuration_hash"]) for item in receipts)
    monitoring = Counter(str(item["monitoring_status"]) for item in receipts)
    return {
        "arrivals": len(receipts),
        "by_role": dict(sorted(roles.items())),
        "by_configuration_hash": dict(sorted(hashes.items())),
        "by_monitoring_status": dict(sorted(monitoring.items())),
    }


def _deployment_view(deployment: dict[str, object]) -> dict[str, object]:
    keys = (
        "admission",
        "state",
        "production_configuration_hash",
        "candidate_configuration_hash",
        "serving_configuration_hash",
        "previous_production_configuration_hash",
        "promoted_configuration_hash",
        "candidate_episodes_served",
        "served_before_rollback",
        "monitor_period_id",
    )
    return {key: deployment.get(key) for key in keys}


def _evidence(store_path: Path, artifact_id: str) -> dict[str, object]:
    reader = EpisodeStore(store_path)
    try:
        artifact = reader.load_validation_artifact(artifact_id)
        decisions = reader.load_deployment_decisions()
        alerts = reader.load_alerts()
        episodes = reader.load_finished_episodes()
    finally:
        reader.close()
    if artifact is None:
        raise ConnectedLifecycleError("validation artifact was not persisted")
    episode_hashes = Counter(item.run.configuration_hash for item in episodes)
    return {
        "validation_artifact": {
            "artifact_id": artifact.artifact_id,
            "outcome": artifact.outcome,
            "evidence_source": artifact.evidence_source,
        },
        "deployment_decisions": [
            {
                "decision": item.decision,
                "method": item.method,
                "configuration_hash": item.configuration_hash,
                "reference_configuration_hash": item.reference_configuration_hash,
                "sample_size": item.sample_size,
                "evidence_artifact_id": item.evidence_artifact_id,
                "evidence_source": item.evidence_source,
            }
            for item in decisions
        ],
        "alerts": [
            {
                "signal": item.signal,
                "slice_name": item.slice_name,
                "method": item.method,
                "configuration_hash": item.configuration_hash,
                "reference_configuration_hash": item.reference_configuration_hash,
                "period_id": item.period_id,
            }
            for item in alerts
        ],
        "finished_episodes": len(episodes),
        "episodes_by_configuration_hash": dict(sorted(episode_hashes.items())),
    }


def run_three_tier_dev(
    gate: GateInputs,
    traffic: DevTraffic,
    *,
    store_path: Path,
    service: ServiceClient,
) -> dict[str, object]:
    """Run Tier 1, admission, Tier 2, and Tier 3 against one service and one store.

    ``service`` is an HTTP client bound to a service whose store is
    ``store_path``: a ``TestClient`` in process, or an ``httpx.Client`` on
    a running ``serve.py``. Status is ``blocked``, ``admission_refused``,
    ``canary_incomplete``, ``promoted``, or ``rolled_back``. Gate execution
    errors and capability refusals propagate before anything is persisted.
    """

    if not isinstance(gate, GateInputs):
        raise ConnectedLifecycleError("gate must be GateInputs")
    if not isinstance(traffic, DevTraffic):
        raise ConnectedLifecycleError("traffic must be DevTraffic")
    _require_fresh_store(store_path)
    _require_registered(service, traffic)

    decision = _run_gate(gate)
    _persist_artifact(store_path, decision)
    artifact_id = decision.artifact.artifact_id
    summary: dict[str, object] = {
        "record": "three_tier_dev_lifecycle",
        "tier1": _tier1(decision),
    }
    if decision.outcome != "PASS":
        summary["status"] = "blocked"
        summary["deployment"] = _deployment_view(
            _json(service.get("/deployment"), "GET /deployment")
        )
        summary["evidence"] = _evidence(store_path, artifact_id)
        assert_public_payload(summary)
        return summary

    response = service.post("/candidates", json={"artifact_id": artifact_id})
    admission: dict[str, object] = {"status_code": response.status_code}
    summary["admission"] = admission
    if response.status_code != 200:
        admission["detail"] = _detail(response)
        summary["status"] = "admission_refused"
        summary["deployment"] = _deployment_view(
            _json(service.get("/deployment"), "GET /deployment")
        )
        summary["evidence"] = _evidence(store_path, artifact_id)
        assert_public_payload(summary)
        return summary
    admitted = _json(response, "POST /candidates")
    if admitted.get("candidate_configuration_hash") != run_configuration_hash(
        traffic.candidate
    ):
        raise ConnectedLifecycleError("service admitted a different candidate")
    admission["state"] = admitted.get("state")

    canary: list[dict[str, object]] = []
    state = admitted.get("state")
    for index, task_id in enumerate(traffic.canary_task_ids):
        if state in _TERMINAL:
            break
        canary.append(_episode(service, task_id, f"canary:{index}:{task_id}"))
        state = _json(service.get("/deployment"), "GET /deployment").get("state")
    summary["tier2"] = {**_receipts(canary), "state": state}
    if state not in _TERMINAL:
        summary["status"] = "canary_incomplete"
        summary["deployment"] = _deployment_view(
            _json(service.get("/deployment"), "GET /deployment")
        )
        summary["evidence"] = _evidence(store_path, artifact_id)
        assert_public_payload(summary)
        return summary

    production = [
        _episode(service, task_id, f"production:{index}:{task_id}")
        for index, task_id in enumerate(traffic.production_task_ids)
    ]
    summary["tier3"] = _receipts(production)
    summary["status"] = _STATUS[str(state)]
    summary["deployment"] = _deployment_view(
        _json(service.get("/deployment"), "GET /deployment")
    )
    summary["evidence"] = _evidence(store_path, artifact_id)
    assert_public_payload(summary)
    return summary
