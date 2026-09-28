"""Caller-built protocol lock and test_normal admission gate.

Thresholds and stopping rules are explicit ``ProtocolSettings`` inputs.
A lock written by tests is not a pre-registered protocol.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from llm_behavior_ci.config import (
    CanarySettings,
    GateSettings,
    MonitorSettings,
    RunConfiguration,
    StreamSettings,
    TaskConfiguration,
    run_configuration_hash,
)
from llm_behavior_ci.experiments.faults import HarmLabel
from llm_behavior_ci.experiments.validation import (
    ValidationReport,
    public_validation_summary,
)

_REQUIRED_FIELDS_PATH = (
    Path(__file__).resolve().parents[3]
    / "configs"
    / "protocol"
    / "required_fields.v1.json"
)
_TRAILING_NEWLINE = b"\n"


class ProtocolError(ValueError):
    pass


def _canonical_json(document: object) -> bytes:
    try:
        text = json.dumps(
            document,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise ProtocolError("protocol document is not finite JSON") from error
    return text.encode("utf-8")


def _document_bytes(digest: str, payload: Mapping[str, object]) -> bytes:
    return _canonical_json({"digest": digest, "payload": payload}) + _TRAILING_NEWLINE


def _under_results(path: Path) -> bool:
    for parent in path.resolve().parents:
        if parent.name == "results":
            return True
    return False


def _load_required_fields() -> tuple[str, ...]:
    try:
        raw = _REQUIRED_FIELDS_PATH.read_text(encoding="utf-8")
        document = json.loads(raw)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ProtocolError("required protocol fields are not readable") from error
    if isinstance(document, list):
        names = document
    elif isinstance(document, Mapping):
        names = document.get("required_fields", document.get("fields"))
        if names is None:
            names = list(document.keys()) if all(
                isinstance(key, str) for key in document
            ) else None
    else:
        names = None
    if not isinstance(names, list) or not names:
        raise ProtocolError("required protocol fields must be a non-empty list")
    fields_list: list[str] = []
    for item in names:
        if not isinstance(item, str) or item == "":
            raise ProtocolError("required protocol field names must be strings")
        fields_list.append(item)
    return tuple(fields_list)


def _harm_label_dict(label: HarmLabel) -> dict[str, object]:
    return {
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


def _strip_protocol_hash(configuration: RunConfiguration) -> dict[str, object]:
    document = configuration.to_dict()
    document["protocol_hash"] = None
    return document


def _analysis_version(value: object) -> str:
    if not isinstance(value, str) or value == "" or value != value.strip():
        raise ProtocolError("analysis_version must be a non-empty single line")
    if "\n" in value or "\r" in value:
        raise ProtocolError("analysis_version must be a non-empty single line")
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        raise ProtocolError("analysis_version must be a non-empty single line")
    return value


def _seeds(value: object) -> tuple[int, ...]:
    if isinstance(value, bool) or not isinstance(value, (list, tuple)):
        raise ProtocolError("seeds must be a non-empty sequence of integers")
    if not value:
        raise ProtocolError("seeds must be a non-empty sequence of integers")
    seeds: list[int] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int):
            raise ProtocolError("seeds must be a non-empty sequence of integers")
        seeds.append(item)
    return tuple(seeds)


def _payload_reports_validated(reports: Sequence[object]) -> None:
    if not reports:
        raise ProtocolError("validation reports are required")
    seen: set[str] = set()
    for item in reports:
        if not isinstance(item, Mapping):
            raise ProtocolError("validation report must be an object")
        method = item.get("method")
        validated = item.get("validated")
        if not isinstance(method, str) or method == "":
            raise ProtocolError("validation report method is required")
        if method in seen:
            raise ProtocolError(f"duplicate validation method: {method}")
        seen.add(method)
        if validated is not True:
            raise ProtocolError("validation reports must all be validated")


@dataclass(frozen=True)
class ProtocolSettings:
    """Caller-supplied lock inputs. No numeric defaults."""

    configurations: tuple[RunConfiguration, ...]
    task_selections: tuple[TaskConfiguration, ...]
    harm_labels: tuple[HarmLabel, ...]
    validation_reports: tuple[ValidationReport, ...]
    gate: GateSettings
    canary: CanarySettings
    monitor: MonitorSettings
    stream: StreamSettings
    analysis_version: str
    seeds: tuple[int, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.configurations, tuple) or not self.configurations:
            raise ProtocolError("configurations must be a non-empty tuple")
        if not isinstance(self.task_selections, tuple) or not self.task_selections:
            raise ProtocolError("task_selections must be a non-empty tuple")
        if not isinstance(self.harm_labels, tuple) or not self.harm_labels:
            raise ProtocolError("harm_labels must be a non-empty tuple")
        if (
            not isinstance(self.validation_reports, tuple)
            or not self.validation_reports
        ):
            raise ProtocolError("validation_reports must be a non-empty tuple")
        if not isinstance(self.gate, GateSettings):
            raise ProtocolError("gate must be GateSettings")
        if not isinstance(self.canary, CanarySettings):
            raise ProtocolError("canary must be CanarySettings")
        if not isinstance(self.monitor, MonitorSettings):
            raise ProtocolError("monitor must be MonitorSettings")
        if not isinstance(self.stream, StreamSettings):
            raise ProtocolError("stream must be StreamSettings")
        object.__setattr__(
            self,
            "analysis_version",
            _analysis_version(self.analysis_version),
        )
        object.__setattr__(self, "seeds", _seeds(self.seeds))

        template_hashes: list[str] = []
        template_tasks: list[dict[str, object]] = []
        dev_task_set_hashes: set[str] = set()
        for configuration in self.configurations:
            if not isinstance(configuration, RunConfiguration):
                raise ProtocolError("configurations must contain RunConfiguration")
            if configuration.protocol_hash is not None:
                raise ProtocolError(
                    "configuration templates must leave protocol_hash unset"
                )
            digest = run_configuration_hash(configuration)
            if digest in template_hashes:
                raise ProtocolError("duplicate configuration template hash")
            template_hashes.append(digest)
            template_tasks.append(configuration.task.to_dict())
            if configuration.task.split == "dev":
                dev_task_set_hashes.add(configuration.task.task_set_hash)

        for selection in self.task_selections:
            if not isinstance(selection, TaskConfiguration):
                raise ProtocolError("task_selections must contain TaskConfiguration")
            if selection.to_dict() not in template_tasks:
                raise ProtocolError("task selection matches no configuration template")

        for label in self.harm_labels:
            if not isinstance(label, HarmLabel):
                raise ProtocolError("harm_labels must contain HarmLabel")
            if label.task_set_hash not in dev_task_set_hashes:
                raise ProtocolError(
                    "harm label task_set_hash matches no dev configuration template"
                )

        methods: list[str] = []
        for report in self.validation_reports:
            if not isinstance(report, ValidationReport):
                raise ProtocolError("validation_reports must contain ValidationReport")
            if report.method in methods:
                raise ProtocolError(f"duplicate validation method: {report.method}")
            methods.append(report.method)
            if report.validated is not True:
                raise ProtocolError("validation reports must all be validated")


@dataclass(frozen=True)
class ProtocolLock:
    """Digest and canonical payload for one locked settings document."""

    digest: str
    payload: Mapping[str, object]

    def __post_init__(self) -> None:
        if not isinstance(self.digest, str) or len(self.digest) != 64:
            raise ProtocolError("digest must be a lowercase SHA-256 hex digest")
        if self.digest != self.digest.lower() or any(
            character not in "0123456789abcdef" for character in self.digest
        ):
            raise ProtocolError("digest must be a lowercase SHA-256 hex digest")
        if not isinstance(self.payload, Mapping):
            raise ProtocolError("payload must be an object")
        object.__setattr__(self, "payload", dict(self.payload))

    @property
    def configurations(self) -> tuple[RunConfiguration, ...]:
        raw = self.payload.get("configurations")
        if not isinstance(raw, list):
            raise ProtocolError("configurations must be a list")
        return tuple(RunConfiguration.from_dict(item) for item in raw)

    @property
    def task_set_hashes(self) -> tuple[str, ...]:
        raw = self.payload.get("task_selections")
        if not isinstance(raw, list):
            raise ProtocolError("task_selections must be a list")
        hashes: list[str] = []
        for item in raw:
            if not isinstance(item, Mapping):
                raise ProtocolError("task selection must be an object")
            value = item.get("task_set_hash")
            if not isinstance(value, str):
                raise ProtocolError("task_set_hash must be a string")
            hashes.append(value)
        return tuple(hashes)

    @property
    def method_names(self) -> tuple[str, ...]:
        raw = self.payload.get("validation_reports")
        if not isinstance(raw, list):
            raise ProtocolError("validation_reports must be a list")
        names: list[str] = []
        for item in raw:
            if not isinstance(item, Mapping):
                raise ProtocolError("validation report must be an object")
            method = item.get("method")
            if not isinstance(method, str):
                raise ProtocolError("validation report method is required")
            names.append(method)
        return tuple(names)

    @property
    def validated_flags(self) -> tuple[bool, ...]:
        raw = self.payload.get("validation_reports")
        if not isinstance(raw, list):
            raise ProtocolError("validation_reports must be a list")
        flags: list[bool] = []
        for item in raw:
            if not isinstance(item, Mapping):
                raise ProtocolError("validation report must be an object")
            validated = item.get("validated")
            if not isinstance(validated, bool):
                raise ProtocolError("validated must be a bool")
            flags.append(validated)
        return tuple(flags)


def _build_payload(settings: ProtocolSettings) -> dict[str, object]:
    return {
        "analysis_version": settings.analysis_version,
        "canary": settings.canary.to_dict(),
        "configurations": [
            configuration.to_dict() for configuration in settings.configurations
        ],
        "gate": settings.gate.to_dict(),
        "harm_labels": [_harm_label_dict(label) for label in settings.harm_labels],
        "monitor": settings.monitor.to_dict(),
        "seeds": list(settings.seeds),
        "stream": settings.stream.to_dict(),
        "task_selections": [
            selection.to_dict() for selection in settings.task_selections
        ],
        "validation_reports": [
            public_validation_summary(report)
            for report in settings.validation_reports
        ],
    }


def _digest_for_payload(payload: Mapping[str, object]) -> str:
    return hashlib.sha256(_canonical_json(payload)).hexdigest()


def lock_protocol(settings: ProtocolSettings, path: Path) -> ProtocolLock:
    """Write an atomic protocol lock document and return its digest wrapper."""

    if not isinstance(settings, ProtocolSettings):
        raise ProtocolError("lock_protocol requires ProtocolSettings")
    output = Path(path)
    if _under_results(output):
        raise ProtocolError("protocol lock cannot be written under results")
    if not output.parent.is_dir():
        raise ProtocolError("protocol lock parent directory does not exist")

    payload = _build_payload(settings)
    digest = _digest_for_payload(payload)
    document = _document_bytes(digest, payload)
    temporary = output.parent / (output.name + ".tmp")
    try:
        with temporary.open("wb") as handle:
            handle.write(document)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output)
    except ProtocolError:
        raise
    except Exception as error:
        if temporary.exists():
            temporary.unlink(missing_ok=True)
        raise ProtocolError("protocol lock write failed") from error

    try:
        on_disk = output.read_bytes()
    except OSError as error:
        raise ProtocolError("protocol lock read-back failed") from error
    if on_disk != document:
        raise ProtocolError("protocol lock bytes disagree with the written document")
    return ProtocolLock(digest=digest, payload=payload)


def require_protocol_lock(path: Path) -> ProtocolLock:
    """Load a lock file and reject missing, mutated, or incomplete documents."""

    target = Path(path)
    try:
        raw = target.read_bytes()
    except FileNotFoundError as error:
        raise ProtocolError("protocol lock file is missing") from error
    except OSError as error:
        raise ProtocolError("protocol lock file is not readable") from error

    try:
        parsed = json.loads(raw.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ProtocolError("protocol lock is not valid JSON") from error
    if not isinstance(parsed, Mapping):
        raise ProtocolError("protocol lock must be an object")
    digest = parsed.get("digest")
    payload = parsed.get("payload")
    if not isinstance(digest, str) or not isinstance(payload, Mapping):
        raise ProtocolError("protocol lock requires digest and payload")

    required = _load_required_fields()
    for name in required:
        if name not in payload:
            raise ProtocolError(f"protocol lock payload missing {name}")

    configurations = payload.get("configurations")
    if not isinstance(configurations, list) or not configurations:
        raise ProtocolError("configurations must be a non-empty list")
    for item in configurations:
        if not isinstance(item, Mapping):
            raise ProtocolError("configuration must be an object")
        if item.get("protocol_hash") is not None:
            raise ProtocolError(
                "stored configuration templates must leave protocol_hash unset"
            )

    reports = payload.get("validation_reports")
    if not isinstance(reports, list):
        raise ProtocolError("validation_reports must be a list")
    _payload_reports_validated(reports)

    recomputed = _digest_for_payload(payload)
    if recomputed != digest:
        raise ProtocolError("protocol lock digest mismatch")
    expected = _document_bytes(digest, payload)
    if raw != expected:
        raise ProtocolError("protocol lock bytes disagree with the canonical document")
    return ProtocolLock(digest=digest, payload=dict(payload))


def bind_protocol(config: RunConfiguration, lock: ProtocolLock) -> RunConfiguration:
    """Return config with ``protocol_hash`` set to ``lock.digest``."""

    if not isinstance(config, RunConfiguration):
        raise ProtocolError("bind_protocol requires a run configuration")
    if not isinstance(lock, ProtocolLock):
        raise ProtocolError("bind_protocol requires a protocol lock")
    if config.protocol_hash is not None:
        if config.protocol_hash == lock.digest:
            return config
        raise ProtocolError("protocol_hash already set to a different digest")

    stripped = _strip_protocol_hash(config)
    templates = [_strip_protocol_hash(item) for item in lock.configurations]
    if stripped not in templates:
        raise ProtocolError("configuration does not match a stored template")

    bound = dict(config.to_dict())
    bound["protocol_hash"] = lock.digest
    return RunConfiguration.from_dict(bound)


def admit_test_normal(
    lock: ProtocolLock,
    config: RunConfiguration,
    *,
    task_set_hash: str,
) -> None:
    """Reject a test_normal run that is not admitted by the lock."""

    if not isinstance(lock, ProtocolLock):
        raise ProtocolError("admit_test_normal requires a protocol lock")
    if not isinstance(config, RunConfiguration):
        raise ProtocolError("admit_test_normal requires a run configuration")
    if not isinstance(task_set_hash, str):
        raise ProtocolError("task_set_hash must be a string")

    recomputed = _digest_for_payload(lock.payload)
    if recomputed != lock.digest:
        raise ProtocolError("protocol lock digest mismatch")

    if config.task.split != "test_normal":
        raise ProtocolError("admit_test_normal requires a test_normal configuration")
    if config.protocol_hash is None or config.protocol_hash != lock.digest:
        raise ProtocolError("configuration protocol_hash does not match the lock")

    stripped = _strip_protocol_hash(config)
    templates = [_strip_protocol_hash(item) for item in lock.configurations]
    if stripped not in templates:
        raise ProtocolError("configuration does not match a stored template")

    if task_set_hash != config.task.task_set_hash:
        raise ProtocolError("task_set_hash does not match the configuration")
    if task_set_hash not in lock.task_set_hashes:
        raise ProtocolError("task_set_hash is not in the locked task selections")

    reports = lock.payload.get("validation_reports")
    if not isinstance(reports, list):
        raise ProtocolError("validation_reports must be a list")
    _payload_reports_validated(reports)
