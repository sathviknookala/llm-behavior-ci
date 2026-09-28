"""Public output is built from aggregate records."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

from llm_behavior_ci.records import (
    AggregateRecord,
    LifecycleDecision,
    RecordError,
    StatisticalEvidence,
    assert_public_payload,
    public_record_dict,
)


class ExportError(ValueError):
    pass


def _require_tuple(value: object, cls: type, name: str) -> None:
    if not isinstance(value, tuple):
        raise ExportError(f"{name} must be a tuple")
    for item in value:
        if not isinstance(item, cls):
            raise ExportError(f"{name} must contain {cls.__name__}")


@dataclass(frozen=True)
class AggregateResults:
    aggregates: tuple[AggregateRecord, ...]
    evidence: tuple[StatisticalEvidence, ...]
    decisions: tuple[LifecycleDecision, ...]

    def __post_init__(self) -> None:
        _require_tuple(self.aggregates, AggregateRecord, "aggregates")
        _require_tuple(self.evidence, StatisticalEvidence, "evidence")
        _require_tuple(self.decisions, LifecycleDecision, "decisions")


def export_public_results(aggregate: AggregateResults, *, output_path: Path) -> None:
    if not isinstance(aggregate, AggregateResults):
        raise ExportError("export requires aggregate results")
    try:
        payload = {
            "aggregates": [public_record_dict(item) for item in aggregate.aggregates],
            "evidence": [public_record_dict(item) for item in aggregate.evidence],
            "decisions": [public_record_dict(item) for item in aggregate.decisions],
        }
        assert_public_payload(payload)
    except RecordError as error:
        raise ExportError("public export failed") from error
    if not output_path.parent.is_dir():
        raise ExportError("parent directory does not exist")
    temporary = output_path.parent / (output_path.name + ".tmp")
    try:
        text = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        with temporary.open("w", encoding="utf-8") as handle:
            handle.write(text)
            handle.write("\n")
            handle.flush()
        os.replace(temporary, output_path)
    except Exception as error:
        if temporary.exists():
            temporary.unlink(missing_ok=True)
        raise ExportError("public export failed") from error
