"""One dev release lifecycle: train gate, admission, paired canary, production monitoring.

``run_three_tier_dev`` drives the existing pieces in order and adds no
decision logic of its own. Tier 1 is ``run_offline_gate`` on a train task
set, and its ``ValidationArtifact`` is written into the service's own
SQLite store. A BLOCK stops there. Admission is ``POST /candidates`` with
that artifact id, so the service's release or test admission, its
task-selection allowance, and its hash checks decide. Tier 2 sends dev
arrivals to ``POST /episodes`` until the service's ``CanaryController``
reaches PROMOTED or ROLLED_BACK, and closes it through ``POST
/deployment/rollback`` when the arrivals run out first. After PROMOTED the
remaining arrivals are Tier 3: ordinary traffic the service routes to the
promoted configuration and feeds to its production monitor. After
ROLLED_BACK the same arrivals only verify that the known-good
configuration still serves. The summary is read back from the store and
``GET /deployment``, never from the runner's own bookkeeping.

Consecutive releases each use a fresh store. ``release_summary`` reduces
one run to a public ``release-summary-v1`` document, and
``require_release_lineage`` refuses release N+1 unless its production
configuration is the one release N left serving, at the same commit.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Mapping, Sequence
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
_LIFECYCLE_DECISIONS = frozenset({"admit", "promote", "rollback"})
RELEASE_SUMMARY_RECORD = "release_summary"
RELEASE_SUMMARY_VERSION = "release-summary-v1"
RELEASE_STATUSES = frozenset(
    {"blocked", "admission_refused", "canary_incomplete", "promoted", "rolled_back"}
)


class ConnectedLifecycleError(RuntimeError):
    """The lifecycle cannot continue; no later stage ran."""


class ReleaseLineageError(ConnectedLifecycleError):
    """Release N+1 does not continue from the configuration release N left serving."""


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


def require_fresh_store(store_path: Path) -> None:
    """Refuse a store that already holds a gate artifact, decision, or episode."""

    reader = EpisodeStore(store_path)
    try:
        if (
            reader.load_validation_artifact_ids()
            or reader.load_deployment_decisions()
            or reader.load_finished_episodes()
        ):
            raise ConnectedLifecycleError(
                "store already holds validation artifacts, deployment decisions, "
                "or episodes; one lifecycle needs its own store"
            )
    finally:
        reader.close()


def require_registered(service: ServiceClient, traffic: DevTraffic) -> dict[str, object]:
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


def deployment_view(deployment: dict[str, object]) -> dict[str, object]:
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


def lifecycle_evidence(store_path: Path, artifact_id: str) -> dict[str, object]:
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


def _roll_back_incomplete_canary(
    service: ServiceClient,
    production_hash: str,
) -> dict[str, object]:
    """Close a canary that ran out of arrivals before its rule decided.

    Uses the service's ``POST /deployment/rollback``, which records a
    ``manual_rollback`` decision, and refuses to report success unless the
    service confirms the candidate is closed and the known-good
    configuration serves.
    """

    response = service.post("/deployment/rollback", json={})
    if response.status_code != 200:
        raise ConnectedLifecycleError(
            f"canary cleanup rollback returned {response.status_code}: {_detail(response)}"
        )
    deployment = response.json()
    if not isinstance(deployment, dict):
        raise ConnectedLifecycleError("canary cleanup rollback did not return an object")
    confirmed = (
        deployment.get("state") == "ROLLED_BACK"
        and deployment.get("admission") == "rollback_requested"
        and deployment.get("serving_configuration_hash") == production_hash
        and deployment.get("production_configuration_hash") == production_hash
        and deployment.get("promoted_configuration_hash") is None
    )
    if not confirmed:
        raise ConnectedLifecycleError(
            "canary cleanup rollback was not confirmed; the candidate may still serve"
        )
    return {
        "rollback": "manual_rollback",
        "reason": "canary_arrivals_exhausted",
        "state": deployment["state"],
        "admission": deployment["admission"],
        "serving_configuration_hash": deployment["serving_configuration_hash"],
    }


def admit_and_run_canary(
    traffic: DevTraffic,
    *,
    store_path: Path,
    service: ServiceClient,
    artifact_id: str,
) -> tuple[dict[str, object], str | None]:
    """Admission and Tier 2: ``POST /candidates``, then canary arrivals.

    The artifact behind ``artifact_id`` must already be in the service's
    store; the service's own admission mode decides whether it is accepted.
    Returns the summary entries and the final canary state. An
    ``admission_refused`` or ``canary_incomplete`` result is complete, with
    ``deployment`` and ``evidence`` read back; after PROMOTED or ROLLED_BACK
    the caller continues. A failed arrival rolls the canary back through
    the service before the error propagates.
    """

    stage: dict[str, object] = {}
    response = service.post("/candidates", json={"artifact_id": artifact_id})
    admission: dict[str, object] = {"status_code": response.status_code}
    stage["admission"] = admission
    if response.status_code != 200:
        admission["detail"] = _detail(response)
        stage["status"] = "admission_refused"
        stage["deployment"] = deployment_view(
            _json(service.get("/deployment"), "GET /deployment")
        )
        stage["evidence"] = lifecycle_evidence(store_path, artifact_id)
        assert_public_payload(stage)
        return stage, None
    admitted = _json(response, "POST /candidates")
    if admitted.get("candidate_configuration_hash") != run_configuration_hash(
        traffic.candidate
    ):
        raise ConnectedLifecycleError("service admitted a different candidate")
    admission["state"] = admitted.get("state")

    canary: list[dict[str, object]] = []
    state = admitted.get("state")
    production_hash = run_configuration_hash(traffic.production)
    try:
        for index, task_id in enumerate(traffic.canary_task_ids):
            if state in _TERMINAL:
                break
            canary.append(_episode(service, task_id, f"canary:{index}:{task_id}"))
            state = _json(service.get("/deployment"), "GET /deployment").get("state")
    except Exception as failure:
        try:
            current = _json(service.get("/deployment"), "GET /deployment").get("state")
        except Exception:
            current = None
        if current not in _TERMINAL:
            try:
                _roll_back_incomplete_canary(service, production_hash)
            except Exception as cleanup:
                raise ConnectedLifecycleError(
                    f"canary failed ({failure}) and cleanup rollback failed ({cleanup})"
                ) from failure
        raise
    stage["tier2"] = {**_receipts(canary), "state": state}
    stage["canary_pair_ids"] = [
        str(item["pair_id"]) for item in canary if item.get("pair_id") is not None
    ]
    if state not in _TERMINAL:
        stage["cleanup"] = _roll_back_incomplete_canary(service, production_hash)
        stage["status"] = "canary_incomplete"
        stage["deployment"] = deployment_view(
            _json(service.get("/deployment"), "GET /deployment")
        )
        stage["evidence"] = lifecycle_evidence(store_path, artifact_id)
        assert_public_payload(stage)
        return stage, None if state is None else str(state)
    stage["status"] = _STATUS[str(state)]
    return stage, str(state)


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
    ``canary_incomplete``, ``promoted``, or ``rolled_back``. A canary whose
    arrivals run out before its rule decides is rolled back through the
    service and reported as ``canary_incomplete`` with a ``cleanup`` entry;
    an unconfirmed cleanup raises. Arrivals after PROMOTED are ``tier3``;
    after ROLLED_BACK they are ``fallback_verification``, and either raises
    when the service routed one to the wrong configuration. Gate execution
    errors and capability refusals propagate before anything is persisted.
    """

    if not isinstance(gate, GateInputs):
        raise ConnectedLifecycleError("gate must be GateInputs")
    if not isinstance(traffic, DevTraffic):
        raise ConnectedLifecycleError("traffic must be DevTraffic")
    require_fresh_store(store_path)
    require_registered(service, traffic)

    decision = _run_gate(gate)
    _persist_artifact(store_path, decision)
    artifact_id = decision.artifact.artifact_id
    summary: dict[str, object] = {
        "record": "three_tier_dev_lifecycle",
        "tier1": _tier1(decision),
    }
    if decision.outcome != "PASS":
        summary["status"] = "blocked"
        summary["deployment"] = deployment_view(
            _json(service.get("/deployment"), "GET /deployment")
        )
        summary["evidence"] = lifecycle_evidence(store_path, artifact_id)
        assert_public_payload(summary)
        return summary

    stage, state = admit_and_run_canary(
        traffic, store_path=store_path, service=service, artifact_id=artifact_id
    )
    summary.update(stage)
    if state not in _TERMINAL:
        return summary

    production_hash = run_configuration_hash(traffic.production)
    expected_hash = (
        run_configuration_hash(traffic.candidate) if state == "PROMOTED" else production_hash
    )
    production = [
        _episode(service, task_id, f"production:{index}:{task_id}")
        for index, task_id in enumerate(traffic.production_task_ids)
    ]
    misrouted = [item for item in production if item["configuration_hash"] != expected_hash]
    if misrouted:
        raise ConnectedLifecycleError(
            f"{len(misrouted)} production arrivals after {state} were not served by "
            "the expected configuration"
        )
    label = "tier3" if state == "PROMOTED" else "fallback_verification"
    summary[label] = _receipts(production)
    summary["status"] = _STATUS[str(state)]
    summary["deployment"] = deployment_view(
        _json(service.get("/deployment"), "GET /deployment")
    )
    summary["evidence"] = lifecycle_evidence(store_path, artifact_id)
    assert_public_payload(summary)
    return summary


def release_summary(
    summary: Mapping[str, object],
    traffic: DevTraffic,
    *,
    lineage: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Reduce one ``run_three_tier_dev`` summary to a public release record.

    ``serving`` is the configuration the service serves after the run, the
    one release N+1 must start from. ``known_good`` is the production
    configuration this release started from, and ``frozen_reference`` is
    the reference the store's last lifecycle decision was measured
    against. Holds hashes, outcomes, and counts only.
    """

    if not isinstance(traffic, DevTraffic):
        raise ConnectedLifecycleError("traffic must be DevTraffic")
    status = summary.get("status")
    if status not in RELEASE_STATUSES:
        raise ConnectedLifecycleError(f"unknown lifecycle status {status}")
    deployment = summary["deployment"]
    evidence = summary["evidence"]
    tier1 = summary["tier1"]
    if not isinstance(deployment, Mapping) or not isinstance(evidence, Mapping):
        raise ConnectedLifecycleError("summary has no deployment or evidence")
    if not isinstance(tier1, Mapping):
        raise ConnectedLifecycleError("summary has no tier1 decision")
    serving = deployment.get("serving_configuration_hash") or deployment.get(
        "production_configuration_hash"
    )
    if serving != deployment.get("production_configuration_hash"):
        raise ConnectedLifecycleError("service reports two serving configurations")
    production_hash = run_configuration_hash(traffic.production)
    candidate_hash = run_configuration_hash(traffic.candidate)
    expected_serving = candidate_hash if status == "promoted" else production_hash
    if serving != expected_serving:
        raise ConnectedLifecycleError(
            f"service serves a configuration that a {status} release does not leave serving"
        )
    decisions = [
        item
        for item in _sequence(evidence.get("deployment_decisions"))
        if item.get("decision") in _LIFECYCLE_DECISIONS
    ]
    last = decisions[-1] if decisions else None
    alerts = _sequence(evidence.get("alerts"))
    document: dict[str, object] = {
        "record": RELEASE_SUMMARY_RECORD,
        "version": RELEASE_SUMMARY_VERSION,
        "git_commit": traffic.production.git_commit,
        "status": status,
        "configurations": {
            "production": production_hash,
            "candidate": candidate_hash,
            "serving": serving,
            "known_good": production_hash,
            "frozen_reference": (
                None if last is None else last.get("reference_configuration_hash")
            ),
        },
        "gate": {
            key: tier1.get(key)
            for key in (
                "outcome",
                "reason_codes",
                "artifact_id",
                "evidence_source",
                "reference_configuration_hash",
                "candidate_configuration_hash",
                "task_set_hash",
            )
        },
        "deployment": {
            "decision": None if last is None else last.get("decision"),
            "method": None if last is None else last.get("method"),
            "decisions": [item.get("decision") for item in decisions],
            "state": deployment.get("state"),
            "admission": deployment.get("admission"),
            "candidate_episodes_served": deployment.get("candidate_episodes_served"),
        },
        "monitoring": {
            "monitor_period_id": deployment.get("monitor_period_id"),
            "tier3_arrivals": _arrivals(summary.get("tier3")),
            "fallback_verification_arrivals": _arrivals(
                summary.get("fallback_verification")
            ),
            "alerts": len(alerts),
            "alerts_by_signal": dict(
                sorted(Counter(str(item.get("signal")) for item in alerts).items())
            ),
        },
        "lineage": None if lineage is None else dict(lineage),
    }
    assert_public_payload(document)
    return document


def require_release_lineage(
    previous: object,
    production: RunConfiguration,
) -> dict[str, object]:
    """Accept release N+1 only if its production is what release N left serving.

    The configuration hash includes ``git_commit``, and live preflight
    binds every configuration to HEAD, so a release at another commit
    cannot continue a lineage: that is refused by name rather than by a
    rehashed history.
    """

    if (
        not isinstance(previous, Mapping)
        or previous.get("record") != RELEASE_SUMMARY_RECORD
        or previous.get("version") != RELEASE_SUMMARY_VERSION
    ):
        raise ReleaseLineageError("previous release is not a release-summary-v1 record")
    if previous.get("status") not in RELEASE_STATUSES:
        raise ReleaseLineageError("previous release has no lifecycle status")
    configurations = previous.get("configurations")
    if not isinstance(configurations, Mapping):
        raise ReleaseLineageError("previous release records no configurations")
    serving = configurations.get("serving")
    if not isinstance(serving, str) or serving == "":
        raise ReleaseLineageError("previous release records no serving configuration")
    if previous.get("git_commit") != production.git_commit:
        raise ReleaseLineageError(
            f"previous release ran at commit {previous.get('git_commit')} and this "
            f"release at {production.git_commit}; cross-commit rollout is not supported"
        )
    if run_configuration_hash(production) != serving:
        raise ReleaseLineageError(
            "production configuration is not the configuration the previous release "
            "left serving"
        )
    canonical = json.dumps(previous, sort_keys=True, separators=(",", ":"))
    return {
        "previous_release_sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        "previous_status": previous["status"],
        "previous_serving_configuration_hash": serving,
        "previous_known_good_configuration_hash": configurations.get("known_good"),
        "previous_frozen_reference_configuration_hash": configurations.get(
            "frozen_reference"
        ),
    }


def _sequence(value: object) -> list[Mapping[str, object]]:
    if not isinstance(value, (list, tuple)):
        return []
    return [item for item in value if isinstance(item, Mapping)]


def _arrivals(stage: object) -> int:
    if not isinstance(stage, Mapping):
        return 0
    return int(stage.get("arrivals", 0))
