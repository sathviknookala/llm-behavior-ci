"""Structured, content-addressed evidence that a real offline gate ran.

``run_offline_gate`` no longer hands candidate admission a free-form
``validation_provenance`` string. It builds a ``ValidationArtifact``: one
frozen, content-addressed record that binds candidate and reference
configuration identity, protocol identity, task-set and split identity,
the gate outcome and reason codes, the statistics the gate actually
computed, the plan-feature schema version, the run identities of the two
paired executions, and a creation timestamp.

``evidence_source`` is not read back from the caller at admission time.
The gate itself sets it, once, from ``PlanEvidenceInputs.
validation_provenance``: the reserved token ``synthetic_fixture`` produces
``evidence_source="synthetic_fixture"``; every other provenance string
produces ``evidence_source="gate_run"``. Admission never sees the raw
provenance string again, only this closed, gate-assigned field, and
``experiments.protocol.authorize_gated_candidate`` refuses
``synthetic_fixture`` outright. ``authorize_test_gated_candidate`` is the
explicit, separately named entry point that accepts either, so CPU tests
can exercise the admission contract against a synthetic fixture without
authorize_gated_candidate itself ever growing an escape hatch.

``artifact_id`` is the SHA-256 hex digest of the artifact's canonical
JSON (same encoding ``config.run_configuration_hash`` uses: sorted keys,
``","``/``":"`` separators, UTF-8, no non-finite numbers). Two artifacts
with the same id have identical content; an admission call that is only
handed an id has no way to substitute different evidence under it, which
is what lets ``storage.EpisodeStore`` act as the record of what a real
gate run actually produced, keyed by that id, without any signing or PKI.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime

from llm_behavior_ci.config import RunConfiguration, RunIdentity, SPLITS, run_configuration_hash
from llm_behavior_ci.lifecycle.plan_features import PLAN_FEATURE_SCHEMA_VERSION
from llm_behavior_ci.records import PUBLIC, Record, RecordError, StatisticalEvidence

GATE_VALIDATION_ARTIFACT_SCHEMA_VERSION = "gate-validation-artifact-v1"
SYNTHETIC_FIXTURE_PROVENANCE = "synthetic_fixture"
EVIDENCE_SOURCES = frozenset({"gate_run", "synthetic_fixture"})
_OUTCOMES = frozenset({"PASS", "BLOCK"})
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GIT_REVISION = re.compile(r"^[0-9a-f]{40}$")
_RUN_ID = re.compile(r"^[0-9a-f]{64}\.[0-9a-f]{32}$")


def _sha256(value: object, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise RecordError(f"{name} must be a lowercase SHA-256 hex digest")
    return value


def _optional_sha256(value: object, name: str) -> str | None:
    if value is None:
        return None
    return _sha256(value, name)


def _choice(value: object, allowed: frozenset[str], name: str) -> str:
    if not isinstance(value, str) or value not in allowed:
        choices = ", ".join(sorted(allowed))
        raise RecordError(f"{name} must be one of: {choices}")
    return value


def _git_revision(value: object, name: str) -> str:
    if not isinstance(value, str) or _GIT_REVISION.fullmatch(value) is None:
        raise RecordError(f"{name} must be a 40-character lowercase git id")
    return value


def _run_id(value: object, name: str) -> str:
    if not isinstance(value, str) or _RUN_ID.fullmatch(value) is None:
        raise RecordError(f"{name} must be a configuration-hash-scoped run id")
    return value


def _line(value: object, name: str) -> str:
    if not isinstance(value, str) or value == "":
        raise RecordError(f"{name} must be a non-empty string")
    return value


@dataclass(frozen=True)
class ValidationArtifact(Record):
    """A public, content-addressed record of one offline-gate execution."""

    visibility = PUBLIC
    record_name = "validation artifact"

    schema_version: str
    evidence_source: str
    outcome: str
    reason_codes: tuple[str, ...]
    reference_configuration_hash: str
    candidate_configuration_hash: str
    reference_protocol_hash: str | None
    candidate_protocol_hash: str | None
    task_set_hash: str
    task_split: str
    reference_run_id: str
    candidate_run_id: str
    reference_git_commit: str
    candidate_git_commit: str
    plan_feature_schema_version: str
    statistics: tuple[StatisticalEvidence, ...]
    created_at: datetime

    def __post_init__(self) -> None:
        if self.schema_version != GATE_VALIDATION_ARTIFACT_SCHEMA_VERSION:
            raise RecordError(
                "schema_version must be " + GATE_VALIDATION_ARTIFACT_SCHEMA_VERSION
            )
        _choice(self.evidence_source, EVIDENCE_SOURCES, "evidence_source")
        _choice(self.outcome, _OUTCOMES, "outcome")
        if not isinstance(self.reason_codes, tuple) or not all(
            isinstance(code, str) and code for code in self.reason_codes
        ):
            raise RecordError("reason_codes must be a tuple of non-empty strings")
        if not isinstance(self.statistics, tuple) or not all(
            isinstance(item, StatisticalEvidence) for item in self.statistics
        ):
            raise RecordError("statistics must be a tuple of StatisticalEvidence")
        if self.outcome == "PASS":
            if self.reason_codes:
                raise RecordError("a PASS artifact must carry no reason codes")
            if not self.statistics:
                raise RecordError("a PASS artifact must carry measured statistics")
        elif not self.reason_codes:
            raise RecordError("a BLOCK artifact must carry at least one reason code")
        _sha256(self.reference_configuration_hash, "reference_configuration_hash")
        _sha256(self.candidate_configuration_hash, "candidate_configuration_hash")
        _optional_sha256(self.reference_protocol_hash, "reference_protocol_hash")
        _optional_sha256(self.candidate_protocol_hash, "candidate_protocol_hash")
        _sha256(self.task_set_hash, "task_set_hash")
        _choice(self.task_split, SPLITS, "task_split")
        _run_id(self.reference_run_id, "reference_run_id")
        _run_id(self.candidate_run_id, "candidate_run_id")
        _git_revision(self.reference_git_commit, "reference_git_commit")
        _git_revision(self.candidate_git_commit, "candidate_git_commit")
        _line(self.plan_feature_schema_version, "plan_feature_schema_version")
        if not isinstance(self.created_at, datetime) or self.created_at.tzinfo is None:
            raise RecordError("created_at must be a timezone-aware timestamp")

    def canonical_bytes(self) -> bytes:
        return json.dumps(
            self.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")

    @property
    def artifact_id(self) -> str:
        return hashlib.sha256(self.canonical_bytes()).hexdigest()


def build_validation_artifact(
    *,
    outcome: str,
    reason_codes: Sequence[str],
    reference: RunConfiguration,
    candidate: RunConfiguration,
    reference_run: RunIdentity,
    candidate_run: RunIdentity,
    task_set_hash: str,
    task_split: str,
    statistics: Sequence[StatisticalEvidence],
    validation_provenance: str,
    created_at: datetime,
) -> ValidationArtifact:
    """Build the artifact a real gate execution emits.

    ``evidence_source`` is derived here, once, from ``validation_provenance``
    rather than accepted as a caller-chosen field on the artifact itself:
    the only way to obtain a ``gate_run`` artifact is to call this from
    ``run_offline_gate`` with a provenance string other than the reserved
    ``synthetic_fixture`` token.
    """

    if not isinstance(validation_provenance, str) or not validation_provenance:
        raise RecordError("validation_provenance must be a non-empty string")
    if not isinstance(reference, RunConfiguration) or not isinstance(
        candidate, RunConfiguration
    ):
        raise RecordError("build_validation_artifact requires run configurations")
    if not isinstance(reference_run, RunIdentity) or not isinstance(
        candidate_run, RunIdentity
    ):
        raise RecordError("build_validation_artifact requires run identities")
    evidence_source = (
        "synthetic_fixture"
        if validation_provenance == SYNTHETIC_FIXTURE_PROVENANCE
        else "gate_run"
    )
    return ValidationArtifact(
        schema_version=GATE_VALIDATION_ARTIFACT_SCHEMA_VERSION,
        evidence_source=evidence_source,
        outcome=outcome,
        reason_codes=tuple(reason_codes),
        reference_configuration_hash=run_configuration_hash(reference),
        candidate_configuration_hash=run_configuration_hash(candidate),
        reference_protocol_hash=reference.protocol_hash,
        candidate_protocol_hash=candidate.protocol_hash,
        task_set_hash=task_set_hash,
        task_split=task_split,
        reference_run_id=reference_run.run_id,
        candidate_run_id=candidate_run.run_id,
        reference_git_commit=reference.git_commit,
        candidate_git_commit=candidate.git_commit,
        plan_feature_schema_version=PLAN_FEATURE_SCHEMA_VERSION,
        statistics=tuple(statistics),
        created_at=created_at,
    )
