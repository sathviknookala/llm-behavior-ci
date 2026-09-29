"""Export a local aggregate JSON through the approved public schema.

Requires --input and --output. Rejects payloads that contain plan text,
instructions, or episode trajectories. Paths under results/ are refused.
Bare invocation exits 2.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

from llm_behavior_ci.export import AggregateResults, ExportError, export_public_results
from llm_behavior_ci.records import (
    PROTECTED_FIELDS,
    AggregateRecord,
    LifecycleDecision,
    RecordError,
    StatisticalEvidence,
    assert_public_payload,
)


def _under_results(path: Path) -> bool:
    for parent in path.resolve().parents:
        if parent.name == "results":
            return True
    return False


def _refuse_results(*paths: Path) -> None:
    for path in paths:
        if _under_results(path):
            raise ExportError("paths under a results directory are refused")


def _load_json(path: Path) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ExportError("input document is not readable JSON") from error


def _reject_protected(payload: object) -> None:
    try:
        assert_public_payload(payload)
    except RecordError as error:
        raise ExportError(str(error) or "public payload rejected") from error
    if isinstance(payload, Mapping):
        for key in payload:
            if key in PROTECTED_FIELDS:
                raise ExportError(f"public payload contains {key}")


def _aggregate_results(payload: object) -> AggregateResults:
    if not isinstance(payload, Mapping):
        raise ExportError("aggregate document must be an object")
    try:
        aggregates_raw = payload["aggregates"]
        evidence_raw = payload["evidence"]
        decisions_raw = payload["decisions"]
    except KeyError as error:
        raise ExportError("aggregate document is missing a required field") from error
    if (
        not isinstance(aggregates_raw, list)
        or not isinstance(evidence_raw, list)
        or not isinstance(decisions_raw, list)
    ):
        raise ExportError("aggregates, evidence, and decisions must be lists")
    _reject_protected(payload)
    try:
        return AggregateResults(
            aggregates=tuple(
                AggregateRecord.from_dict(item) for item in aggregates_raw
            ),
            evidence=tuple(
                StatisticalEvidence.from_dict(item) for item in evidence_raw
            ),
            decisions=tuple(
                LifecycleDecision.from_dict(item) for item in decisions_raw
            ),
        )
    except (RecordError, TypeError, ValueError) as error:
        raise ExportError("aggregate document is invalid") from error


def main(argv: Sequence[str] | None = None) -> int:
    args_list = list(sys.argv[1:] if argv is None else argv)
    if not args_list:
        return 2
    parser = argparse.ArgumentParser(
        description="Export approved public aggregates from a local JSON document."
    )
    parser.add_argument("--input")
    parser.add_argument("--output")
    try:
        args = parser.parse_args(args_list)
    except SystemExit as error:
        code = error.code
        if code is None:
            return 2
        return int(code)
    if args.input is None or args.output is None:
        return 2
    try:
        input_path = Path(args.input)
        output_path = Path(args.output)
        _refuse_results(input_path, output_path)
        aggregate = _aggregate_results(_load_json(input_path))
        export_public_results(aggregate, output_path=output_path)
        return 0
    except ExportError as error:
        print(str(error) or "public export failed", file=sys.stderr)
        return 1
    except RecordError as error:
        print(str(error) or "public payload rejected", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
