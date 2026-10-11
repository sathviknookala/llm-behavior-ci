"""The release contract: what one GitHub-driven release may run, and from what.

A release runs at one clean, pinned commit. Its candidate is built at that
commit. Its production is either built there too (the first release) or is
the configuration the previous release left serving, carried unchanged with
the commit it was built at. Three checks decide whether that carry is safe:

``verify_release_record`` re-derives release N's outcome from release N's
own SQLite store (gate artifact, lifecycle decisions, configurations served)
and refuses a summary that disagrees, so a tampered or stale summary cannot
name what serves. A post-promotion rollback recorded after the summary was
written is accepted, and the known-good configuration then continues.

``require_runtime_transition`` decides whether code at the release commit
may execute a configuration built at an earlier commit. The checkout must
have full history (a shallow clone is refused), the production commit
must be an ancestor of the release commit, and either the two
commits run episodes through byte-identical execution code
(``runtime.provenance.execution_fingerprint``) or the release commit's
``configs/release/runtime_compatibility.json`` declares the production
commit's execution fingerprint compatible. Anything else is refused: an
incompatible runtime change ships as a candidate built at the new commit,
through the gate and the canary, never as a silent re-execution.

``require_deployed`` checks the running service before a release takes it
over: it must serve exactly the configuration the lineage expects, with no
canary open.

``provision_release`` turns a local ``release-spec-v1`` document into an
immutable release directory: configurations, task manifests, gate settings,
plan evidence, task-selection allowance, canary and monitor settings, frozen
reference, service settings, budgets, a fresh store, a ``git archive`` of
the release commit for the persistent service, and a ``release-manifest-v2``
that pins every file by SHA-256 and the archived code tree by
``code_tree_digest``. It never overwrites a release.

``verify_release_manifest`` re-checks all of it, code tree included, without
git, so the persistent service can verify the tree it runs from.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import shutil
import stat
import subprocess
import tarfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from llm_behavior_ci.config import (
    CanarySettings,
    ConfigError,
    GateSettings,
    MonitorSettings,
    RunConfiguration,
    run_configuration_hash,
)
from llm_behavior_ci.experiments.faults import FaultError, apply_fault, load_fault
from llm_behavior_ci.experiments.protocol import (
    ProtocolError,
    bind_train_task_selection,
    task_selection_allowance_document,
    task_selection_allowance_for,
)
from llm_behavior_ci.experiments.run_config import (
    LocalTaskManifest,
    RunConfigError,
    build_run_configuration,
    configuration_document,
    load_local_task_manifest,
    preflight_tool_access,
)
from llm_behavior_ci.lifecycle.connected import (
    PRODUCTION_ROLLED_BACK,
    ReleaseLineageError,
    require_release_lineage,
    require_release_record,
)
from llm_behavior_ci.lifecycle.monitoring import FrozenReference, MonitorRejected
from llm_behavior_ci.lifecycle.offline_gate import (
    plan_evidence_from_dict,
    require_gate_capabilities,
)
from llm_behavior_ci.runtime.provenance import (
    ProvenanceError,
    RepositoryState,
    execution_fingerprint,
    is_ancestor,
    relevant_dirty_paths,
    repository_root,
    require_commit_id,
    require_full_history,
)
from llm_behavior_ci.storage import EpisodeStore

RELEASE_SPEC_VERSION = "release-spec-v1"
RELEASE_MANIFEST_RECORD = "release_manifest"
RELEASE_MANIFEST_VERSION = "release-manifest-v2"
CODE_DIGEST_VERSION = "release-code-v1"
RUNTIME_COMPATIBILITY_VERSION = "runtime-compatibility-v1"
RUNTIME_COMPATIBILITY_PATH = "configs/release/runtime_compatibility.json"
RELEASE_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,99}$")
LATEST_POINTER = "LATEST"
MANIFEST_FILE = "release.json"
SUMMARY_FILE = "release_summary.json"
STORE_FILE = "episodes.sqlite"
CODE_DIR = "code"
CODE_COMMIT_FILE = "RELEASE_COMMIT"
POST_PROMOTION_ROLLBACK_POLICIES = ("record_only", "rollback_on_alert")
_LIFECYCLE_DECISIONS = ("admit", "promote", "rollback", "production_rollback")
_SETTLED_STATES = frozenset({None, "PROMOTED", "ROLLED_BACK", PRODUCTION_ROLLED_BACK})
_STATUS_LAST_DECISION = {
    "blocked": None,
    "admission_refused": None,
    "promoted": "promote",
    "rolled_back": "rollback",
    "canary_incomplete": "rollback",
    "production_rolled_back": "production_rollback",
}
CONFIGURATION_FILES = {
    "gate_reference": "gate_reference.json",
    "gate_candidate": "gate_candidate.json",
    "production": "production.json",
    "candidate": "candidate.json",
}
SETTINGS_FILES = (
    "train_task_set.json",
    "dev_task_set.json",
    "gate_settings.json",
    "plan_evidence.json",
    "canary_settings.json",
    "monitor_settings.json",
    "frozen_reference.json",
    "task_selection_allowance.json",
    "service.json",
)
_SPEC_KEYS = frozenset(
    {
        "version",
        "candidate",
        "initial_production",
        "run_seed",
        "protocol_hash",
        "train_manifest",
        "dev_manifest",
        "gate_settings",
        "plan_evidence",
        "canary_settings",
        "monitor_settings",
        "frozen_reference",
        "service",
        "gate_endpoints",
        "budget",
    }
)
_TEMPLATE_KEYS = frozenset({"template", "fault"})
_SERVICE_KEYS = frozenset(
    {
        "host",
        "port",
        "max_in_flight",
        "shutdown_timeout_seconds",
        "dedup_seconds",
        "canary_assignment_seed",
        "production_base_url",
        "candidate_base_url",
        "post_promotion_rollback",
    }
)
_ENDPOINT_KEYS = frozenset({"reference", "candidate"})
_BUDGET_KEYS = frozenset(
    {
        "canary_arrivals",
        "production_arrivals",
        "max_episodes",
        "max_gate_model_calls",
        "max_service_model_calls",
    }
)
_FROZEN_KEYS = frozenset({"baselines", "slice_baselines", "source"})


class ReleaseError(ValueError):
    """A release input is invalid or unsafe; nothing was provisioned or run."""


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _canonical(document: object) -> bytes:
    return (json.dumps(document, sort_keys=True, indent=2) + "\n").encode("utf-8")


def require_release_id(value: object, label: str = "release id") -> str:
    if not isinstance(value, str) or not RELEASE_ID_PATTERN.match(value):
        raise ReleaseError(f"{label} must match {RELEASE_ID_PATTERN.pattern}")
    return value


@dataclass(frozen=True)
class RuntimeTransition:
    """Why code at ``release_commit`` may execute a production built at ``production_commit``.

    ``basis`` is ``same_commit``, ``identical_execution`` (equal execution
    fingerprints), or ``declared_compatible`` (the release commit's
    compatibility file lists the production fingerprint).
    """

    production_commit: str
    release_commit: str
    basis: str
    production_execution_fingerprint: str
    release_execution_fingerprint: str

    def to_dict(self) -> dict[str, str]:
        return {
            "production_commit": self.production_commit,
            "release_commit": self.release_commit,
            "basis": self.basis,
            "production_execution_fingerprint": self.production_execution_fingerprint,
            "release_execution_fingerprint": self.release_execution_fingerprint,
        }


def runtime_compatibility_from_dict(payload: object) -> frozenset[str]:
    """Strictly parse a ``runtime-compatibility-v1`` document into fingerprints."""

    if not isinstance(payload, Mapping) or set(payload) != {"version", "declarations"}:
        raise ReleaseError("runtime compatibility must hold exactly version and declarations")
    if payload["version"] != RUNTIME_COMPATIBILITY_VERSION:
        raise ReleaseError(f"runtime compatibility version must be {RUNTIME_COMPATIBILITY_VERSION}")
    declarations = payload["declarations"]
    if not isinstance(declarations, list):
        raise ReleaseError("runtime compatibility declarations must be a list")
    fingerprints: set[str] = set()
    for item in declarations:
        if not isinstance(item, Mapping) or set(item) != {"execution_fingerprint", "reason"}:
            raise ReleaseError(
                "each declaration holds exactly execution_fingerprint and reason"
            )
        fingerprint = item["execution_fingerprint"]
        if (
            not isinstance(fingerprint, str)
            or len(fingerprint) != 64
            or any(character not in "0123456789abcdef" for character in fingerprint)
        ):
            raise ReleaseError("execution_fingerprint must be a lowercase SHA-256")
        if not isinstance(item["reason"], str) or item["reason"].strip() == "":
            raise ReleaseError("a compatibility declaration needs a reason")
        fingerprints.add(fingerprint)
    return frozenset(fingerprints)


def committed_runtime_compatibility(
    commit: str,
    root: Path | None = None,
    *,
    runner: Callable[..., object] = subprocess.run,
) -> frozenset[str]:
    """The compatibility declarations committed at ``commit``; none when the file is absent."""

    repo = root if root is not None else repository_root()
    try:
        completed = runner(
            ["git", "show", f"{commit}:{RUNTIME_COMPATIBILITY_PATH}"],
            cwd=repo,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as error:
        raise ReleaseError("git show failed") from error
    if getattr(completed, "returncode", 1) != 0:
        return frozenset()
    try:
        payload = json.loads(getattr(completed, "stdout", ""))
    except json.JSONDecodeError as error:
        raise ReleaseError("runtime compatibility file is not JSON") from error
    return runtime_compatibility_from_dict(payload)


def require_runtime_transition(
    production_commit: str,
    release_commit: str,
    root: Path | None = None,
    *,
    runner: Callable[..., object] = subprocess.run,
) -> RuntimeTransition:
    """Refuse unless code at ``release_commit`` executes the production as its own commit did."""

    try:
        require_commit_id(production_commit, "production commit")
        require_commit_id(release_commit, "release commit")
        release_fingerprint = execution_fingerprint(release_commit, root, runner=runner)
        if production_commit == release_commit:
            return RuntimeTransition(
                production_commit=production_commit,
                release_commit=release_commit,
                basis="same_commit",
                production_execution_fingerprint=release_fingerprint,
                release_execution_fingerprint=release_fingerprint,
            )
        require_full_history(root, runner=runner)
        if not is_ancestor(production_commit, release_commit, root, runner=runner):
            raise ReleaseError(
                f"production commit {production_commit} is not an ancestor of release "
                f"commit {release_commit}; a release only moves forward"
            )
        production_fingerprint = execution_fingerprint(production_commit, root, runner=runner)
    except ProvenanceError as error:
        raise ReleaseError(str(error)) from error
    if production_fingerprint == release_fingerprint:
        basis = "identical_execution"
    elif production_fingerprint in committed_runtime_compatibility(
        release_commit, root, runner=runner
    ):
        basis = "declared_compatible"
    else:
        raise ReleaseError(
            f"execution code changed between production commit {production_commit} and "
            f"release commit {release_commit} (fingerprint {production_fingerprint}), and "
            f"{RUNTIME_COMPATIBILITY_PATH} does not declare it compatible; ship the "
            "change as a candidate built at the new commit"
        )
    return RuntimeTransition(
        production_commit=production_commit,
        release_commit=release_commit,
        basis=basis,
        production_execution_fingerprint=production_fingerprint,
        release_execution_fingerprint=release_fingerprint,
    )


def require_release_provenance(
    *,
    release_configurations: Sequence[RunConfiguration],
    gate_reference: RunConfiguration,
    production: RunConfiguration,
    state: RepositoryState,
    carried: bool,
) -> None:
    """The candidate side is built at a clean HEAD; production is HEAD's or carried.

    ``carried`` is true only after ``require_release_lineage`` accepted the
    production as the previous release's serving configuration. A
    production at another commit without that lineage is unrecognized.
    """

    if relevant_dirty_paths(state.dirty_paths):
        raise ReleaseError("tracked source or config files are dirty")
    for configuration in release_configurations:
        if configuration.git_commit != state.head:
            raise ReleaseError("candidate configurations must be built at HEAD")
    if gate_reference.git_commit != production.git_commit:
        raise ReleaseError("the gate reference must be built at the production's commit")
    if production.git_commit != state.head and not carried:
        raise ReleaseError(
            "a production built at another commit needs the previous release that left "
            "it serving"
        )


def verify_release_record(previous: object, store_path: Path) -> str:
    """Re-derive release N from its store and return the configuration it leaves serving.

    Refuses a summary whose gate artifact, lifecycle decisions, status, or
    serving configuration the store does not show, a store holding another
    artifact or a configuration outside the release, and a canary left
    open. Decisions in the store must equal the summary's, or extend a
    ``promoted`` summary by exactly one later ``production_rollback``; the
    known-good configuration then serves.
    """

    record = require_release_record(previous)
    if not Path(store_path).is_file():
        raise ReleaseLineageError("previous release store is missing")
    configurations = record["configurations"]
    gate = record.get("gate")
    deployment = record.get("deployment")
    assert isinstance(configurations, Mapping)
    if not isinstance(gate, Mapping) or not isinstance(deployment, Mapping):
        raise ReleaseLineageError("previous release records no gate or deployment")
    production = configurations.get("production")
    candidate = configurations.get("candidate")
    known = {production, candidate}
    reader = EpisodeStore(Path(store_path))
    try:
        artifact_ids = reader.load_validation_artifact_ids()
        decisions = [
            item
            for item in reader.load_deployment_decisions()
            if item.decision in _LIFECYCLE_DECISIONS
        ]
        artifact = (
            None
            if gate.get("artifact_id") is None
            else reader.load_validation_artifact(str(gate["artifact_id"]))
        )
        episodes = reader.load_finished_episodes()
    finally:
        reader.close()
    if artifact is None or tuple(artifact_ids) != (artifact.artifact_id,):
        raise ReleaseLineageError("previous release store does not hold its gate artifact")
    for key, value in (
        ("outcome", artifact.outcome),
        ("evidence_source", artifact.evidence_source),
        ("reference_configuration_hash", artifact.reference_configuration_hash),
        ("candidate_configuration_hash", artifact.candidate_configuration_hash),
        ("task_set_hash", artifact.task_set_hash),
    ):
        if gate.get(key) != value:
            raise ReleaseLineageError(f"previous release summary gate {key} disagrees with its store")
    names = [item.decision for item in decisions]
    recorded = deployment.get("decisions")
    if not isinstance(recorded, list):
        raise ReleaseLineageError("previous release records no decision sequence")
    status = record["status"]
    rolled_back_later = False
    if names != recorded:
        if status == "promoted" and names == [*recorded, "production_rollback"]:
            rolled_back_later = True
        else:
            raise ReleaseLineageError(
                "previous release summary decisions disagree with its store"
            )
    for item in decisions:
        if item.configuration_hash not in known or item.reference_configuration_hash not in known:
            raise ReleaseLineageError(
                "previous release store holds a decision outside its configurations"
            )
    if any(item.run.configuration_hash not in known for item in episodes):
        raise ReleaseLineageError(
            "previous release store holds an episode outside its configurations"
        )
    expected_last = _STATUS_LAST_DECISION[str(status)]
    last = recorded[-1] if recorded else None
    if last != expected_last:
        raise ReleaseLineageError(
            f"a {status} release cannot end with decision {last}"
        )
    if status in {"blocked", "admission_refused"}:
        if (gate.get("outcome") == "PASS") != (status == "admission_refused"):
            raise ReleaseLineageError("previous release status disagrees with its gate")
    derived = candidate if last == "promote" else production
    if configurations.get("serving") != derived:
        raise ReleaseLineageError(
            "previous release summary serving configuration disagrees with its store"
        )
    if rolled_back_later:
        return str(production)
    return str(derived)


def require_deployed(deployment: Mapping[str, object], serving: str) -> None:
    """The running service serves ``serving`` and has no canary in flight."""

    if not isinstance(deployment, Mapping):
        raise ReleaseError("deployment document must be an object")
    state = deployment.get("state")
    if state not in _SETTLED_STATES:
        raise ReleaseError(f"the running service is {state}; a canary is still in flight")
    if deployment.get("admission") == "shutting_down":
        raise ReleaseError("the running service is shutting down")
    actual = deployment.get("serving_configuration_hash") or deployment.get(
        "production_configuration_hash"
    )
    if actual != serving:
        raise ReleaseError(
            "the running service does not serve the configuration the previous release "
            "left serving"
        )


def read_latest(release_root: Path) -> str | None:
    pointer = Path(release_root) / LATEST_POINTER
    if not pointer.exists():
        return None
    return require_release_id(pointer.read_text(encoding="utf-8").strip(), "LATEST")


def write_latest(release_root: Path, release_id: str) -> None:
    """Point ``LATEST`` at a release whose summary was written, atomically."""

    require_release_id(release_id)
    root = Path(release_root)
    if not (root / release_id / SUMMARY_FILE).is_file():
        raise ReleaseError("a release becomes LATEST only after its summary is written")
    temporary = root / f".{LATEST_POINTER}.{os.getpid()}"
    temporary.write_text(release_id + "\n", encoding="utf-8")
    os.replace(temporary, root / LATEST_POINTER)


@dataclass(frozen=True)
class ReleaseSpec:
    """A parsed ``release-spec-v1``: the operator's local recipe for a release."""

    candidate_template: str
    candidate_fault: str | None
    initial_production_template: str | None
    run_seed: int
    protocol_hash: str | None
    train_manifest: Path
    dev_manifest: Path
    gate_settings: GateSettings
    plan_evidence: Mapping[str, object]
    canary_settings: CanarySettings
    monitor_settings: Mapping[str, object]
    frozen_reference: Mapping[str, object]
    service: Mapping[str, object]
    gate_endpoints: Mapping[str, object]
    budget: Mapping[str, int]


def _exact(payload: object, keys: frozenset[str], label: str) -> Mapping[str, object]:
    if not isinstance(payload, Mapping):
        raise ReleaseError(f"{label} must be an object")
    missing = sorted(keys - set(payload))
    unknown = sorted(set(payload) - keys)
    if missing or unknown:
        detail = []
        if missing:
            detail.append("missing " + ", ".join(missing))
        if unknown:
            detail.append("unknown " + ", ".join(unknown))
        raise ReleaseError(f"{label}: " + "; ".join(detail))
    return payload


def _repo_relative(value: object, label: str, prefix: str) -> str:
    if not isinstance(value, str) or value == "":
        raise ReleaseError(f"{label} must be a repository path")
    normalized = Path(value)
    if normalized.is_absolute() or ".." in normalized.parts or not value.startswith(prefix):
        raise ReleaseError(f"{label} must be a path under {prefix}")
    return value


def _positive_int(value: object, label: str, *, allow_zero: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < (0 if allow_zero else 1):
        raise ReleaseError(f"{label} must be a {'non-negative' if allow_zero else 'positive'} int")
    return value


def _template(payload: object, label: str) -> tuple[str, str | None]:
    document = _exact(payload, _TEMPLATE_KEYS, label)
    template = _repo_relative(document["template"], f"{label}.template", "configs/models/")
    fault = document["fault"]
    if fault is not None:
        fault = _repo_relative(fault, f"{label}.fault", "configs/faults/")
    return template, fault


def release_spec_from_dict(payload: object, *, base: Path) -> ReleaseSpec:
    """Strictly parse a ``release-spec-v1`` document; relative local paths resolve against ``base``."""

    document = _exact(payload, _SPEC_KEYS, "release spec")
    if document["version"] != RELEASE_SPEC_VERSION:
        raise ReleaseError(f"release spec version must be {RELEASE_SPEC_VERSION}")
    candidate_template, candidate_fault = _template(document["candidate"], "candidate")
    initial = document["initial_production"]
    initial_template = None
    if initial is not None:
        initial_template, initial_fault = _template(initial, "initial_production")
        if initial_fault is not None:
            raise ReleaseError("initial_production takes no fault")
    if isinstance(document["run_seed"], bool) or not isinstance(document["run_seed"], int):
        raise ReleaseError("run_seed must be an int")
    protocol_hash = document["protocol_hash"]
    if protocol_hash is not None and not isinstance(protocol_hash, str):
        raise ReleaseError("protocol_hash must be a string or null")
    manifests = {}
    for key in ("train_manifest", "dev_manifest", "plan_evidence"):
        value = document[key]
        if not isinstance(value, str) or value == "":
            raise ReleaseError(f"{key} must be a local file path")
        path = Path(value)
        manifests[key] = path if path.is_absolute() else Path(base) / path
    try:
        gate_settings = GateSettings.from_dict(document["gate_settings"])
        canary_settings = CanarySettings.from_dict(document["canary_settings"])
    except (ConfigError, TypeError, KeyError) as error:
        raise ReleaseError(f"release settings are invalid: {error}") from error
    monitor = document["monitor_settings"]
    if not isinstance(monitor, Mapping) or "reference_configuration_hash" in monitor:
        raise ReleaseError(
            "monitor_settings must be an object without reference_configuration_hash; "
            "the release binds it to production"
        )
    frozen = _exact(
        document["frozen_reference"], _FROZEN_KEYS, "frozen_reference"
    )
    service = _exact(document["service"], _SERVICE_KEYS, "service")
    if not isinstance(service["host"], str) or service["host"] not in {"127.0.0.1", "localhost"}:
        raise ReleaseError("the release service listens on localhost only")
    port = _positive_int(service["port"], "service.port")
    if port > 65535:
        raise ReleaseError("service.port must be a TCP port")
    _positive_int(service["max_in_flight"], "service.max_in_flight")
    for key in ("shutdown_timeout_seconds", "dedup_seconds"):
        if isinstance(service[key], bool) or not isinstance(service[key], (int, float)) or service[key] < 0:
            raise ReleaseError(f"service.{key} must be a non-negative number")
    if isinstance(service["canary_assignment_seed"], bool) or not isinstance(
        service["canary_assignment_seed"], int
    ):
        raise ReleaseError("service.canary_assignment_seed must be an int")
    for key in ("production_base_url", "candidate_base_url"):
        if service[key] is not None and not isinstance(service[key], str):
            raise ReleaseError(f"service.{key} must be a string or null")
    if service["post_promotion_rollback"] not in POST_PROMOTION_ROLLBACK_POLICIES:
        raise ReleaseError(
            "service.post_promotion_rollback must be one of: "
            + ", ".join(POST_PROMOTION_ROLLBACK_POLICIES)
        )
    endpoints = _exact(document["gate_endpoints"], _ENDPOINT_KEYS, "gate_endpoints")
    for key, value in endpoints.items():
        if value is not None and not isinstance(value, str):
            raise ReleaseError(f"gate_endpoints.{key} must be a string or null")
    budget_document = _exact(document["budget"], _BUDGET_KEYS, "budget")
    budget = {
        key: _positive_int(
            budget_document[key], f"budget.{key}", allow_zero=key == "production_arrivals"
        )
        for key in sorted(_BUDGET_KEYS)
    }
    return ReleaseSpec(
        candidate_template=candidate_template,
        candidate_fault=candidate_fault,
        initial_production_template=initial_template,
        run_seed=document["run_seed"],
        protocol_hash=protocol_hash,
        train_manifest=manifests["train_manifest"],
        dev_manifest=manifests["dev_manifest"],
        gate_settings=gate_settings,
        plan_evidence=_load_json_file(manifests["plan_evidence"], "plan_evidence"),
        canary_settings=canary_settings,
        monitor_settings=dict(monitor),
        frozen_reference=dict(frozen),
        service=dict(service),
        gate_endpoints=dict(endpoints),
        budget=budget,
    )


def _load_json_file(path: Path, label: str) -> Mapping[str, object]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ReleaseError(f"{label} is not a readable JSON file") from error
    if not isinstance(payload, Mapping):
        raise ReleaseError(f"{label} must be a JSON object")
    return payload


def load_release_spec(path: Path) -> ReleaseSpec:
    return release_spec_from_dict(_load_json_file(path, "release spec"), base=Path(path).parent)


def _load_configuration(path: Path) -> RunConfiguration:
    try:
        return RunConfiguration.from_dict(_load_json_file(path, path.name))
    except ConfigError as error:
        raise ReleaseError(f"{path.name} is not a run configuration: {error}") from error


def _build(
    root: Path,
    template: str,
    fault: str | None,
    manifest: LocalTaskManifest,
    *,
    spec: ReleaseSpec,
    commit: str,
) -> RunConfiguration:
    try:
        configuration = build_run_configuration(
            _load_json_file(root / template, template),
            manifest.task,
            run_seed=spec.run_seed,
            git_commit=commit,
            protocol_hash=spec.protocol_hash,
        )
        if fault is not None:
            configuration = apply_fault(configuration, load_fault(root / fault))
        preflight_tool_access(configuration, manifest)
    except (RunConfigError, FaultError) as error:
        raise ReleaseError(str(error)) from error
    return configuration


@dataclass(frozen=True)
class _Carried:
    gate_reference: RunConfiguration
    production: RunConfiguration
    previous_id: str
    summary_sha256: str
    lineage: dict[str, object]


def _carry_previous(release_root: Path, previous_id: str) -> _Carried:
    """Release N's serving configuration files, verified against its summary and store."""

    previous_dir = Path(release_root) / previous_id
    manifest = verify_release_manifest(previous_dir)
    summary_path = previous_dir / SUMMARY_FILE
    if not summary_path.is_file():
        raise ReleaseLineageError(f"release {previous_id} has no release summary")
    summary_bytes = summary_path.read_bytes()
    try:
        summary = json.loads(summary_bytes)
    except json.JSONDecodeError as error:
        raise ReleaseLineageError("previous release summary is not JSON") from error
    serving = verify_release_record(summary, previous_dir / STORE_FILE)
    roles = manifest["configurations"]
    assert isinstance(roles, Mapping)
    for dev_role, train_role in (("candidate", "gate_candidate"), ("production", "gate_reference")):
        entry = roles[dev_role]
        assert isinstance(entry, Mapping)
        if entry["hash"] == serving:
            production = _load_configuration(previous_dir / CONFIGURATION_FILES[dev_role])
            gate_reference = _load_configuration(previous_dir / CONFIGURATION_FILES[train_role])
            break
    else:
        raise ReleaseLineageError(
            "neither configuration of the previous release is the one it left serving"
        )
    lineage = require_release_lineage(summary, production, serving=serving)
    return _Carried(
        gate_reference=gate_reference,
        production=production,
        previous_id=previous_id,
        summary_sha256=_sha256_bytes(summary_bytes),
        lineage=lineage,
    )


def _require_bindings(
    configuration: RunConfiguration, manifest: LocalTaskManifest, label: str
) -> None:
    if configuration.task != manifest.task:
        raise ReleaseError(
            f"{label} is bound to another {manifest.task_set.split} task set; a "
            "configuration's task binding is part of its hash"
        )


def _archive_commit(root: Path, commit: str, destination: Path) -> None:
    completed = subprocess.run(
        ["git", "archive", "--format=tar", commit],
        cwd=root,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise ReleaseError("git archive of the release commit failed")
    destination.mkdir()
    with tarfile.open(fileobj=io.BytesIO(completed.stdout)) as archive:
        archive.extractall(destination, filter="data")
    (destination / CODE_COMMIT_FILE).write_text(commit + "\n", encoding="utf-8")


def code_tree_digest(code: Path) -> dict[str, object]:
    """A deterministic digest of an archived code tree, computed without git.

    Every directory, regular file, and symlink under ``code`` is one entry,
    sorted by relative path: a file by its SHA-256 and executable bit, a
    symlink by its target text (never followed), a directory by its name.
    An added, removed, renamed, or modified entry, a retargeted symlink, or
    a changed executable bit changes the digest. ``code`` must be a real
    directory, and any other kind of file in it is refused.
    """

    code = Path(code)
    if code.is_symlink() or not code.is_dir():
        raise ReleaseError("release code must be a real directory")
    entries: list[list[str]] = []

    def unreadable(error: OSError) -> None:
        raise ReleaseError(f"release code is unreadable: {error.filename}") from error

    for current, directories, files in os.walk(code, onerror=unreadable, followlinks=False):
        base = Path(current)
        for name in (*directories, *files):
            path = base / name
            relative = path.relative_to(code).as_posix()
            mode = os.lstat(path).st_mode
            if stat.S_ISLNK(mode):
                entries.append([relative, "symlink", os.readlink(path)])
            elif stat.S_ISDIR(mode):
                entries.append([relative, "directory", ""])
            elif stat.S_ISREG(mode):
                executable = "x" if mode & 0o111 else "-"
                entries.append([relative, "file", f"{executable}{file_sha256(path)}"])
            else:
                raise ReleaseError(f"release code holds an unsupported file: {relative}")
    entries.sort()
    return {
        "version": CODE_DIGEST_VERSION,
        "sha256": _sha256_bytes(json.dumps(entries, separators=(",", ":")).encode("utf-8")),
        "entries": len(entries),
    }


def require_release_code(release_dir: Path, manifest: Mapping[str, object]) -> None:
    """Refuse an archived code tree that differs from the one provisioned."""

    pinned = manifest.get("code")
    if not isinstance(pinned, Mapping) or pinned.get("version") != CODE_DIGEST_VERSION:
        raise ReleaseError("release manifest does not pin its code tree")
    code = Path(release_dir) / CODE_DIR
    if code_tree_digest(code) != dict(pinned):
        raise ReleaseError("release code changed after provisioning")
    commit_file = code / CODE_COMMIT_FILE
    if commit_file.read_text(encoding="utf-8").strip() != manifest.get("release_commit"):
        raise ReleaseError("release code is not the release commit")


def _read_only(path: Path) -> None:
    for item in sorted(path.rglob("*"), reverse=True):
        if item.is_symlink():
            continue
        mode = item.stat().st_mode
        item.chmod(mode & ~0o222)


def provision_release(
    spec: ReleaseSpec,
    *,
    release_root: Path,
    release_id: str,
    state: RepositoryState,
    deployment: Mapping[str, object] | None,
    now: datetime,
    root: Path | None = None,
    archive: Callable[[Path, str, Path], None] = _archive_commit,
    runner: Callable[..., object] = subprocess.run,
) -> dict[str, object]:
    """Provision ``release_root/release_id`` from ``spec`` at the clean HEAD in ``state``.

    The previous release is ``LATEST`` under ``release_root``. With none,
    production is built from ``initial_production`` and no service may be
    deployed (``deployment`` is ``None``). Otherwise production is release
    N's serving configuration, carried unchanged after
    ``verify_release_record``, ``require_release_lineage``, and
    ``require_runtime_transition``, and ``deployment`` (the running
    service's ``GET /deployment``) must serve it. Everything is written
    to a hidden directory and renamed into place last, so a failure leaves
    no release behind and an existing release is never touched.
    """

    repo = root if root is not None else repository_root()
    release_root = Path(release_root)
    require_release_id(release_id)
    if not release_root.is_dir():
        raise ReleaseError("release root does not exist")
    final = release_root / release_id
    if final.exists():
        raise ReleaseError(f"release {release_id} already exists; releases are never overwritten")
    if relevant_dirty_paths(state.dirty_paths):
        raise ReleaseError("tracked source or config files are dirty")
    commit = require_commit_id(state.head, "HEAD")
    try:
        train = load_local_task_manifest(
            spec.train_manifest, committed_tasks_dir=repo / "configs" / "tasks"
        )
        dev = load_local_task_manifest(
            spec.dev_manifest, committed_tasks_dir=repo / "configs" / "tasks"
        )
    except (RunConfigError, OSError) as error:
        raise ReleaseError(f"task manifest is invalid: {error}") from error
    if train.task_set.split != "train" or dev.task_set.split != "dev":
        raise ReleaseError("the gate runs on a train manifest and the service on a dev manifest")

    gate_candidate = _build(repo, spec.candidate_template, spec.candidate_fault, train, spec=spec, commit=commit)
    candidate = _build(repo, spec.candidate_template, spec.candidate_fault, dev, spec=spec, commit=commit)

    previous_id = read_latest(release_root)
    carried: _Carried | None = None
    if previous_id is None:
        if spec.initial_production_template is None:
            raise ReleaseError("the first release needs initial_production")
        if deployment is not None:
            raise ReleaseError(
                "a service is deployed but no previous release is recorded; refuse to "
                "replace an unrecorded deployment"
            )
        gate_reference = _build(repo, spec.initial_production_template, None, train, spec=spec, commit=commit)
        production = _build(repo, spec.initial_production_template, None, dev, spec=spec, commit=commit)
    else:
        if previous_id == release_id:
            raise ReleaseError("a release cannot continue itself")
        carried = _carry_previous(release_root, previous_id)
        gate_reference = carried.gate_reference
        production = carried.production
        if deployment is None:
            raise ReleaseError("the previous release's service must be running to be continued")
        require_deployed(deployment, run_configuration_hash(production))
    _require_bindings(gate_reference, train, "gate reference")
    _require_bindings(production, dev, "production")
    require_release_provenance(
        release_configurations=(gate_candidate, candidate),
        gate_reference=gate_reference,
        production=production,
        state=state,
        carried=carried is not None,
    )
    transition = require_runtime_transition(production.git_commit, commit, repo, runner=runner)

    production_hash = run_configuration_hash(production)
    candidate_hash = run_configuration_hash(candidate)
    if production_hash == candidate_hash:
        raise ReleaseError("the candidate is the production configuration")
    allowance = task_selection_allowance_for(gate_reference)
    try:
        for label, served, gated in (
            ("production", production, gate_reference),
            ("candidate", candidate, gate_candidate),
        ):
            if run_configuration_hash(bind_train_task_selection(served, allowance)) != run_configuration_hash(gated):
                raise ReleaseError(
                    f"{label} differs from its gate configuration beyond task selection"
                )
    except ProtocolError as error:
        raise ReleaseError(str(error)) from error
    try:
        plan_evidence = plan_evidence_from_dict(spec.plan_evidence)
    except (ValueError, KeyError, TypeError) as error:
        raise ReleaseError(f"plan evidence is invalid: {error}") from error
    try:
        require_gate_capabilities(gate_reference, gate_candidate, plan_evidence.required_statistics)
    except Exception as error:
        raise ReleaseError(str(error)) from error
    monitor_document = {**spec.monitor_settings, "reference_configuration_hash": production_hash}
    try:
        MonitorSettings.from_dict(monitor_document)
    except (ConfigError, TypeError, KeyError) as error:
        raise ReleaseError(f"monitor_settings are invalid: {error}") from error
    frozen_document = {"configuration_hash": production_hash, **spec.frozen_reference}
    _require_frozen(frozen_document)

    budget = dict(spec.budget)
    train_count = len(train.task_set.task_ids)
    dev_count = len(dev.task_set.task_ids)
    if budget["canary_arrivals"] + budget["production_arrivals"] > dev_count:
        raise ReleaseError("arrival budget exceeds the dev task set")
    episode_bound = 2 * train_count + 2 * budget["canary_arrivals"] + budget["production_arrivals"]
    if episode_bound > budget["max_episodes"]:
        raise ReleaseError(
            f"this release can run {episode_bound} episodes, above max_episodes "
            f"{budget['max_episodes']}"
        )

    staging = release_root / f".{release_id}.provisioning"
    if staging.exists():
        raise ReleaseError(f"release {release_id} is already being provisioned")
    staging.mkdir()
    try:
        documents: dict[str, bytes] = {
            CONFIGURATION_FILES["gate_reference"]: _canonical(configuration_document(gate_reference)),
            CONFIGURATION_FILES["gate_candidate"]: _canonical(configuration_document(gate_candidate)),
            CONFIGURATION_FILES["production"]: _canonical(configuration_document(production)),
            CONFIGURATION_FILES["candidate"]: _canonical(configuration_document(candidate)),
            "train_task_set.json": Path(spec.train_manifest).read_bytes(),
            "dev_task_set.json": Path(spec.dev_manifest).read_bytes(),
            "gate_settings.json": _canonical(spec.gate_settings.to_dict()),
            "plan_evidence.json": _canonical(dict(spec.plan_evidence)),
            "canary_settings.json": _canonical(spec.canary_settings.to_dict()),
            "monitor_settings.json": _canonical(monitor_document),
            "frozen_reference.json": _canonical(frozen_document),
            "task_selection_allowance.json": _canonical(task_selection_allowance_document(allowance)),
            "service.json": _canonical(dict(spec.service)),
        }
        for name, payload in documents.items():
            (staging / name).write_bytes(payload)
        EpisodeStore(staging / STORE_FILE).close()
        archive(repo, commit, staging / CODE_DIR)
        code_digest = code_tree_digest(staging / CODE_DIR)
        manifest: dict[str, object] = {
            "record": RELEASE_MANIFEST_RECORD,
            "version": RELEASE_MANIFEST_VERSION,
            "release_id": release_id,
            "release_commit": commit,
            "created_at": now.isoformat(),
            "previous": (
                None
                if carried is None
                else {
                    "release_id": carried.previous_id,
                    "summary_sha256": carried.summary_sha256,
                    "lineage": carried.lineage,
                }
            ),
            "configurations": {
                role: {
                    "file": CONFIGURATION_FILES[role],
                    "hash": run_configuration_hash(configuration),
                    "git_commit": configuration.git_commit,
                }
                for role, configuration in (
                    ("gate_reference", gate_reference),
                    ("gate_candidate", gate_candidate),
                    ("production", production),
                    ("candidate", candidate),
                )
            },
            "runtime": transition.to_dict(),
            "task_sets": {
                "train": train.task_set.task_set_hash,
                "dev": dev.task_set.task_set_hash,
                "train_tasks": train_count,
                "dev_tasks": dev_count,
            },
            "budget": {**budget, "episode_bound": episode_bound},
            "service": dict(spec.service),
            "gate_endpoints": dict(spec.gate_endpoints),
            "files": {name: _sha256_bytes(payload) for name, payload in sorted(documents.items())},
            "code": code_digest,
        }
        (staging / MANIFEST_FILE).write_bytes(_canonical(manifest))
        _read_only(staging / CODE_DIR)
        for name in (*documents, MANIFEST_FILE):
            _read_only_file(staging / name)
        if final.exists():
            raise ReleaseError(f"release {release_id} already exists; releases are never overwritten")
        os.rename(staging, final)
    except BaseException:
        _remove_tree(staging)
        raise
    return manifest


def _read_only_file(path: Path) -> None:
    path.chmod(path.stat().st_mode & ~0o222)


def _remove_tree(path: Path) -> None:
    if not path.exists():
        return
    for item in path.rglob("*"):
        if not item.is_symlink():
            item.chmod(item.stat().st_mode | 0o200)
    path.chmod(path.stat().st_mode | 0o200)
    shutil.rmtree(path)


def _require_frozen(document: Mapping[str, object]) -> None:
    baselines = document.get("baselines")
    slices = document.get("slice_baselines")
    if not isinstance(baselines, list) or not isinstance(slices, list):
        raise ReleaseError("frozen_reference baselines and slice_baselines must be lists")
    source = document.get("source")
    if source is not None and not isinstance(source, str):
        raise ReleaseError("frozen_reference source must be a string or null")
    try:
        FrozenReference(
            configuration_hash=str(document["configuration_hash"]),
            baselines=tuple((str(name), float(value)) for name, value in baselines),
            slice_baselines=tuple(
                (str(name), tuple((str(signal), float(value)) for signal, value in items))
                for name, items in slices
            ),
            source=source,
        )
    except (MonitorRejected, ValueError, TypeError) as error:
        raise ReleaseError(f"frozen_reference is invalid: {error}") from error


def verify_release_manifest(release_dir: Path, *, head: str | None = None) -> dict[str, object]:
    """Load ``release.json`` and refuse a changed file, configuration, code tree, or commit.

    With ``head``, the manifest must have been provisioned at that commit.
    The code tree is checked against its ``code_tree_digest`` and its
    ``RELEASE_COMMIT``; the store and later lifecycle outputs sit outside
    it and stay writable.
    """

    release_dir = Path(release_dir)
    manifest = _load_json_file(release_dir / MANIFEST_FILE, MANIFEST_FILE)
    if (
        manifest.get("record") != RELEASE_MANIFEST_RECORD
        or manifest.get("version") != RELEASE_MANIFEST_VERSION
    ):
        raise ReleaseError(f"not a {RELEASE_MANIFEST_VERSION} document")
    if manifest.get("release_id") != release_dir.name:
        raise ReleaseError("release manifest names another release")
    if head is not None and manifest.get("release_commit") != head:
        raise ReleaseError(
            f"release was provisioned at {manifest.get('release_commit')}, not at HEAD {head}"
        )
    files = manifest.get("files")
    if not isinstance(files, Mapping) or set(files) != {*CONFIGURATION_FILES.values(), *SETTINGS_FILES}:
        raise ReleaseError("release manifest does not pin the release files")
    for name, digest in files.items():
        path = release_dir / str(name)
        if not path.is_file() or file_sha256(path) != digest:
            raise ReleaseError(f"release file {name} changed after provisioning")
    configurations = manifest.get("configurations")
    if not isinstance(configurations, Mapping) or set(configurations) != set(CONFIGURATION_FILES):
        raise ReleaseError("release manifest does not name its configurations")
    for role, entry in configurations.items():
        if not isinstance(entry, Mapping):
            raise ReleaseError("release manifest configuration entry must be an object")
        configuration = _load_configuration(release_dir / CONFIGURATION_FILES[str(role)])
        if (
            run_configuration_hash(configuration) != entry.get("hash")
            or configuration.git_commit != entry.get("git_commit")
        ):
            raise ReleaseError(f"release {role} configuration does not match its manifest")
    require_release_code(release_dir, manifest)
    return dict(manifest)


def require_previous_summary(release_root: Path, manifest: Mapping[str, object]) -> Path | None:
    """Release N's summary path, unchanged since release N+1 was provisioned from it."""

    previous = manifest.get("previous")
    if previous is None:
        return None
    if not isinstance(previous, Mapping):
        raise ReleaseError("release manifest previous entry must be an object")
    previous_id = require_release_id(previous.get("release_id"), "previous release id")
    path = Path(release_root) / previous_id / SUMMARY_FILE
    if not path.is_file() or file_sha256(path) != previous.get("summary_sha256"):
        raise ReleaseLineageError("previous release summary changed after provisioning")
    return path


def require_current_lineage(release_root: Path, manifest: Mapping[str, object]) -> None:
    """Refuse a release provisioned from a release that is no longer ``LATEST``."""

    previous = manifest.get("previous")
    expected = previous.get("release_id") if isinstance(previous, Mapping) else None
    latest = read_latest(release_root)
    if latest != expected:
        raise ReleaseError(
            f"release continues {expected}, but LATEST is {latest}; provision it again"
        )


def service_argv(release_dir: Path, manifest: Mapping[str, object]) -> list[str]:
    """``serve.py`` arguments for a provisioned release, read only from its manifest."""

    release_dir = Path(release_dir)
    service = manifest["service"]
    budget = manifest["budget"]
    assert isinstance(service, Mapping) and isinstance(budget, Mapping)
    argv = [
        "--production-config", str(release_dir / CONFIGURATION_FILES["production"]),
        "--candidate-config", str(release_dir / CONFIGURATION_FILES["candidate"]),
        "--store", str(release_dir / STORE_FILE),
        "--max-in-flight", str(service["max_in_flight"]),
        "--shutdown-timeout-seconds", str(service["shutdown_timeout_seconds"]),
        "--canary-settings", str(release_dir / "canary_settings.json"),
        "--monitor-settings", str(release_dir / "monitor_settings.json"),
        "--frozen-reference", str(release_dir / "frozen_reference.json"),
        "--dedup-seconds", str(service["dedup_seconds"]),
        "--canary-assignment-seed", str(service["canary_assignment_seed"]),
        "--task-selection-allowance", str(release_dir / "task_selection_allowance.json"),
        "--task-metadata", str(release_dir / "dev_task_set.json"),
        "--post-promotion-rollback", str(service["post_promotion_rollback"]),
        "--max-model-calls", str(budget["max_service_model_calls"]),
        "--host", str(service["host"]),
        "--port", str(service["port"]),
    ]
    for flag, key in (
        ("--production-base-url", "production_base_url"),
        ("--candidate-base-url", "candidate_base_url"),
    ):
        if service.get(key) is not None:
            argv.extend([flag, str(service[key])])
    return argv


def service_url(manifest: Mapping[str, object]) -> str:
    service = manifest["service"]
    assert isinstance(service, Mapping)
    return f"http://{service['host']}:{service['port']}"

