"""Build or verify a protocol lock from a caller-supplied settings JSON.

Settings JSON fields:
- configurations: RunConfiguration.to_dict() list (protocol_hash null)
- task_selections: TaskConfiguration.to_dict() list
- harm_labels: HarmLabel field objects
- validation_reports: public_validation_summary objects (include
  visibility) or full ValidationReport field trees
- gate, canary, monitor, stream: their to_dict() shapes
- analysis_version: non-empty string
- seeds: non-empty list of integers

Full ValidationReport JSON mirrors ValidationReport fields. Nested
CheckResult, CaseAgreement, and AADependenceReport objects use the same
field names; tuple fields are JSON arrays of values or pairs. The public
summary shape is that tree plus \"visibility\": \"public\". Thresholds are
not defaulted.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

from llm_behavior_ci.config import (
    CanarySettings,
    GateSettings,
    MonitorSettings,
    RunConfiguration,
    StreamSettings,
    TaskConfiguration,
)
from llm_behavior_ci.experiments.faults import HarmLabel
from llm_behavior_ci.experiments.protocol import (
    ProtocolError,
    ProtocolSettings,
    lock_protocol,
    require_protocol_lock,
)
from llm_behavior_ci.experiments.validation import (
    AADependenceReport,
    CaseAgreement,
    CheckResult,
    ValidationReport,
)


def _load_json(path: Path) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ProtocolError("settings document is not readable JSON") from error


def _pairs(value: object, name: str) -> tuple[tuple[str, object], ...]:
    if not isinstance(value, list):
        raise ProtocolError(f"{name} must be a list")
    pairs: list[tuple[str, object]] = []
    for item in value:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise ProtocolError(f"{name} entries must be pairs")
        key, raw = item
        if not isinstance(key, str):
            raise ProtocolError(f"{name} keys must be strings")
        pairs.append((key, raw))
    return tuple(pairs)


def _check_result(payload: Mapping[str, object]) -> CheckResult:
    return CheckResult(
        name=str(payload["name"]),
        status=str(payload["status"]),
        reason=str(payload["reason"]),
        sample_count=int(payload["sample_count"]),
        seeds=tuple(int(item) for item in payload["seeds"]),
        estimate=payload["estimate"],
        interval_low=payload["interval_low"],
        interval_high=payload["interval_high"],
        uncertainty_level=payload["uncertainty_level"],
        details=tuple(
            (str(key), float(raw)) for key, raw in _pairs(payload["details"], "details")
        ),
    )


def _case_agreement(payload: Mapping[str, object]) -> CaseAgreement:
    return CaseAgreement(
        case_id=str(payload["case_id"]),
        statistic=str(payload["statistic"]),
        expected=float(payload["expected"]),
        observed=float(payload["observed"]),
        absolute_error=float(payload["absolute_error"]),
        tolerance=float(payload["tolerance"]),
        source=str(payload["source"]),
        agreed=bool(payload["agreed"]),
    )


def _aa_report(payload: Mapping[str, object]) -> AADependenceReport:
    levels = payload["concurrency_levels"]
    if not isinstance(levels, list):
        raise ProtocolError("concurrency_levels must be a list")
    return AADependenceReport(
        status=str(payload["status"]),
        evidence_accepted=bool(payload["evidence_accepted"]),
        provenance=str(payload["provenance"]),
        hardware_observed=bool(payload["hardware_observed"]),
        observation_count=int(payload["observation_count"]),
        pair_count=int(payload["pair_count"]),
        repeated_task_effect=payload["repeated_task_effect"],
        scenario_clustering_effect=payload["scenario_clustering_effect"],
        inference_variation=payload["inference_variation"],
        inference_source=payload["inference_source"],
        inference_low=payload["inference_low"],
        inference_high=payload["inference_high"],
        trajectory_divergence_rate=payload["trajectory_divergence_rate"],
        interval_width_ratio=payload["interval_width_ratio"],
        concurrency_effect=payload["concurrency_effect"],
        concurrency_levels=tuple(int(item) for item in levels),
        series_alarm=payload["series_alarm"],
        memory_used_mib=payload["memory_used_mib"],
        wall_seconds=payload["wall_seconds"],
        reason=str(payload["reason"]),
    )


def _validation_report(payload: object) -> ValidationReport:
    if not isinstance(payload, Mapping):
        raise ProtocolError("validation report must be an object")
    mapping = {key: value for key, value in payload.items() if key != "visibility"}
    try:
        aa_raw = mapping["aa"]
        if not isinstance(aa_raw, Mapping):
            raise ProtocolError("aa must be an object")
        checks_raw = mapping["checks"]
        agreements_raw = mapping["reference_agreements"]
        if not isinstance(checks_raw, list) or not isinstance(agreements_raw, list):
            raise ProtocolError("checks and reference_agreements must be lists")
        parameters = _pairs(mapping["parameters"], "parameters")
        libraries = tuple(
            (str(key), bool(raw))
            for key, raw in _pairs(mapping["libraries"], "libraries")
        )
        return ValidationReport(
            method=str(mapping["method"]),
            implemented=bool(mapping["implemented"]),
            validated=bool(mapping["validated"]),
            benchmark_eligible=bool(mapping["benchmark_eligible"]),
            calibration=str(mapping["calibration"]),
            study=str(mapping["study"]),
            null_claim=str(mapping["null_claim"]),
            null_draw=str(mapping["null_draw"]),
            input_hash=str(mapping["input_hash"]),
            seeds=tuple(int(item) for item in mapping["seeds"]),
            sample_count=int(mapping["sample_count"]),
            null_sample_size=int(mapping["null_sample_size"]),
            uncertainty_level=float(mapping["uncertainty_level"]),
            alpha=mapping["alpha"],
            horizon=mapping["horizon"],
            false_alarm_tolerance=mapping["false_alarm_tolerance"],
            coverage_tolerance=mapping["coverage_tolerance"],
            parameters=tuple((key, str(raw)) for key, raw in parameters),
            required_checks=tuple(str(item) for item in mapping["required_checks"]),
            omitted_checks=tuple(str(item) for item in mapping["omitted_checks"]),
            checks=tuple(
                _check_result(item) for item in checks_raw if isinstance(item, Mapping)
            ),
            reference_agreements=tuple(
                _case_agreement(item)
                for item in agreements_raw
                if isinstance(item, Mapping)
            ),
            aa=_aa_report(aa_raw),
            libraries=libraries,
            configuration_hashes=tuple(
                str(item) for item in mapping["configuration_hashes"]
            ),
            gpu_evidence=bool(mapping["gpu_evidence"]),
            gpu_floor_measured=bool(mapping["gpu_floor_measured"]),
            split=mapping["split"],
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ProtocolError("validation report fields are incomplete") from error


def _harm_label(payload: object) -> HarmLabel:
    if not isinstance(payload, Mapping):
        raise ProtocolError("harm label must be an object")
    try:
        return HarmLabel(
            fault_version=str(payload["fault_version"]),
            base_configuration_hash=str(payload["base_configuration_hash"]),
            candidate_configuration_hash=str(payload["candidate_configuration_hash"]),
            task_set_hash=str(payload["task_set_hash"]),
            effect_estimate=float(payload["effect_estimate"]),
            interval_low=float(payload["interval_low"]),
            interval_high=float(payload["interval_high"]),
            margin=float(payload["margin"]),
            harmful=bool(payload["harmful"]),
            split=str(payload["split"]),
            confidence_level=float(payload["confidence_level"]),
            resamples=int(payload["resamples"]),
            seed=int(payload["seed"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ProtocolError("harm label fields are incomplete") from error


def _settings_from_mapping(document: object) -> ProtocolSettings:
    if not isinstance(document, Mapping):
        raise ProtocolError("settings document must be an object")
    try:
        configurations = document["configurations"]
        task_selections = document["task_selections"]
        harm_labels = document["harm_labels"]
        validation_reports = document["validation_reports"]
        seeds = document["seeds"]
    except KeyError as error:
        raise ProtocolError("settings document is missing a required field") from error
    if not isinstance(configurations, list) or not isinstance(task_selections, list):
        raise ProtocolError("configurations and task_selections must be lists")
    if not isinstance(harm_labels, list) or not isinstance(validation_reports, list):
        raise ProtocolError("harm_labels and validation_reports must be lists")
    if not isinstance(seeds, list):
        raise ProtocolError("seeds must be a list")
    return ProtocolSettings(
        configurations=tuple(
            RunConfiguration.from_dict(item) for item in configurations
        ),
        task_selections=tuple(
            TaskConfiguration.from_dict(item) for item in task_selections
        ),
        harm_labels=tuple(_harm_label(item) for item in harm_labels),
        validation_reports=tuple(
            _validation_report(item) for item in validation_reports
        ),
        gate=GateSettings.from_dict(document["gate"]),
        canary=CanarySettings.from_dict(document["canary"]),
        monitor=MonitorSettings.from_dict(document["monitor"]),
        stream=StreamSettings.from_dict(document["stream"]),
        analysis_version=str(document["analysis_version"]),
        seeds=tuple(int(item) for item in seeds),
    )


def main(argv: Sequence[str] | None = None) -> int:
    args_list = list(sys.argv[1:] if argv is None else argv)
    if not args_list:
        return 2
    parser = argparse.ArgumentParser(
        description=(
            "Lock or require a protocol document from caller-supplied settings. "
            "A lock file is not preregistration."
        )
    )
    parser.add_argument("--settings")
    parser.add_argument("--output")
    parser.add_argument("--require")
    try:
        args = parser.parse_args(args_list)
    except SystemExit as error:
        code = error.code
        if code is None:
            return 2
        return int(code)

    try:
        if args.require is not None:
            if args.settings is not None or args.output is not None:
                return 2
            lock = require_protocol_lock(Path(args.require))
            print(lock.digest)
            print(
                "protocol lock is caller-supplied settings, not preregistration",
                file=sys.stderr,
            )
            return 0
        if args.settings is None or args.output is None:
            return 2
        settings = _settings_from_mapping(_load_json(Path(args.settings)))
        lock = lock_protocol(settings, Path(args.output))
        print(lock.digest)
        print(
            "protocol lock is caller-supplied settings, not preregistration",
            file=sys.stderr,
        )
        return 0
    except ProtocolError as error:
        print(str(error) or "protocol lock failed", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
