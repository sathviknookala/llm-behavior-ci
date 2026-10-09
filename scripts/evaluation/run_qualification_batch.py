"""Prepare, run, or score a ledgered qualification batch.

``--design dev_four_arm`` needs --task-set (a local dev manifest),
--healthy, --noop, --noop-fault, --regression, --regression-fault, and
--arm-order-seed; --arm-orders optionally names a file whose orders must
equal the seeded ones. ``--design plan_aa`` needs --task-set (a local
train manifest) and --healthy. Every action needs --checkpoint.

--prepare binds the design to a new checkpoint with --max-executions and
--max-plan-generations and calls no provider. --run resumes that
checkpoint with --live-runtime and --episode-store, reconciles open
attempts first, and never dispatches a (task, arm) scope already in the
ledger. Caps cannot exceed the design and are fixed at first use.
--evidence reads the checkpoint and writes --output: the dev batch needs
--margin, --confidence-level, --resamples, and --seed for the harm
labels; the plan batch needs --gate-settings and --plan-evidence. Repeated
--method-spec adds each method's A/A dependence result. Repeated
--lock-inputs (one file per method of the design: canary and CUSUM for the
dev batch, bootstrap and MMD for the plan batch) adds each method's
assembled lock report and its status; a report stays ``pending_live_aa``
until live A/A outcomes exist. test_normal is refused. Bare invocation
exits 2.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

from llm_behavior_ci.config import ConfigError, GateSettings
from llm_behavior_ci.experiments.attempts import AttemptBudget, AttemptError
from llm_behavior_ci.experiments.faults import FaultError, load_fault
from llm_behavior_ci.experiments.qualification_batch import (
    DEV_FOUR_ARM,
    PLAN_AA,
    BatchDesign,
    LockReportInputs,
    QualificationBatchError,
    cusum_scale,
    dev_four_arm_design,
    dev_harm_labels,
    dev_outcome_table,
    execution_aa_reports,
    execution_lock_reports,
    load_batch_checkpoint,
    load_configuration,
    lock_report_inputs_from_dict,
    lock_report_status,
    open_batch,
    plan_aa_design,
    plan_aa_gate_replay,
    plan_aa_reports,
    plan_aa_series,
    plan_lock_reports,
    run_qualification_batch,
    seeded_arm_orders,
)
from llm_behavior_ci.experiments.run_config import RunConfigError, load_local_task_manifest
from llm_behavior_ci.experiments.validation import (
    ValidationError,
    ValidationReport,
    method_spec_from_dict,
    public_validation_summary,
)
from llm_behavior_ci.lifecycle.offline_gate import GateExecutionError, plan_evidence_from_dict
from llm_behavior_ci.records import RecordError
from llm_behavior_ci.runtime.provenance import ProvenanceError, enforce_committed_provenance


def main(argv: Sequence[str] | None = None) -> int:
    args_list = list(sys.argv[1:] if argv is None else argv)
    if not args_list:
        return 2
    parser = argparse.ArgumentParser(description="Run a ledgered qualification batch.")
    parser.add_argument("--design", required=True, choices=(DEV_FOUR_ARM, PLAN_AA))
    parser.add_argument("--task-set", required=True)
    parser.add_argument("--healthy", required=True)
    parser.add_argument("--noop")
    parser.add_argument("--noop-fault")
    parser.add_argument("--regression")
    parser.add_argument("--regression-fault")
    parser.add_argument("--arm-order-seed", type=int)
    parser.add_argument("--arm-orders")
    parser.add_argument("--checkpoint", required=True)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--prepare", action="store_true")
    action.add_argument("--run", action="store_true")
    action.add_argument("--evidence", action="store_true")
    parser.add_argument("--max-executions", type=int)
    parser.add_argument("--max-plan-generations", type=int)
    parser.add_argument("--live-runtime", action="store_true")
    parser.add_argument("--episode-store")
    parser.add_argument("--endpoint")
    parser.add_argument("--output")
    parser.add_argument("--margin", type=float)
    parser.add_argument("--confidence-level", type=float)
    parser.add_argument("--resamples", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--gate-settings")
    parser.add_argument("--plan-evidence")
    parser.add_argument("--method-spec", action="append", default=[])
    parser.add_argument("--lock-inputs", action="append", default=[])
    try:
        args = parser.parse_args(args_list)
    except SystemExit as error:
        return 2 if error.code is None else int(error.code)
    try:
        design = _design(args)
        checkpoint = Path(args.checkpoint)
        if args.prepare:
            document, ledger = open_batch(design, checkpoint, budget=_budget(args, required=True))
            printed: dict[str, object] = {
                "action": "prepare",
                "design": design.kind,
                "identity_hash": document["identity_hash"],
                "attempts": ledger.summary(),
            }
        elif args.run:
            printed = _run(args, design, checkpoint)
        else:
            printed = _evidence(args, design, checkpoint)
    except (
        QualificationBatchError,
        AttemptError,
        ConfigError,
        RunConfigError,
        FaultError,
        GateExecutionError,
        ProvenanceError,
        ValidationError,
        RecordError,
        OSError,
        json.JSONDecodeError,
    ) as error:
        print(str(error) or "qualification batch failed", file=sys.stderr)
        return 1
    print(json.dumps(printed, sort_keys=True))
    return 0


def _design(args: argparse.Namespace) -> BatchDesign:
    manifest = load_local_task_manifest(Path(args.task_set))
    task_set = manifest.task_set
    if task_set.split.startswith("test"):
        raise QualificationBatchError("qualification batches never run a test split")
    healthy = load_configuration(Path(args.healthy))
    if args.design == PLAN_AA:
        return plan_aa_design(task_set, healthy=healthy)
    missing = [
        flag
        for flag, value in (
            ("--noop", args.noop),
            ("--noop-fault", args.noop_fault),
            ("--regression", args.regression),
            ("--regression-fault", args.regression_fault),
            ("--arm-order-seed", args.arm_order_seed),
        )
        if value is None
    ]
    if missing:
        raise QualificationBatchError(f"dev_four_arm needs {', '.join(missing)}")
    orders = seeded_arm_orders(task_set.task_count, args.arm_order_seed)
    if args.arm_orders is not None and _declared_orders(Path(args.arm_orders)) != orders:
        raise QualificationBatchError("declared arm orders differ from the seeded orders")
    return dev_four_arm_design(
        task_set,
        healthy=healthy,
        noop=load_configuration(Path(args.noop)),
        noop_fault=load_fault(Path(args.noop_fault)),
        regression=load_configuration(Path(args.regression)),
        regression_fault=load_fault(Path(args.regression_fault)),
        arm_orders=orders,
    )


def _declared_orders(path: Path) -> tuple[tuple[str, ...], ...]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        payload = payload.get("execution_batch", {}).get("arm_orders")
    if not isinstance(payload, list):
        raise QualificationBatchError("arm orders file has no order list")
    orders = []
    for item in payload:
        order = item.get("execution_order") if isinstance(item, dict) else item
        if not isinstance(order, list):
            raise QualificationBatchError("arm orders file has an invalid order")
        orders.append(tuple(str(arm) for arm in order))
    return tuple(orders)


def _budget(args: argparse.Namespace, *, required: bool) -> AttemptBudget | None:
    supplied = (args.max_executions is not None, args.max_plan_generations is not None)
    if not any(supplied) and not required:
        return None
    if not all(supplied):
        raise QualificationBatchError(
            "pass both --max-executions and --max-plan-generations"
        )
    return AttemptBudget(
        plan_generations=args.max_plan_generations, executions=args.max_executions
    )


def _run(args: argparse.Namespace, design: BatchDesign, checkpoint: Path) -> dict[str, object]:
    if not args.live_runtime:
        raise QualificationBatchError("--run needs --live-runtime")
    if args.episode_store is None:
        raise QualificationBatchError("--run needs --episode-store")
    if not checkpoint.exists():
        raise QualificationBatchError("--run resumes a prepared checkpoint; run --prepare first")
    root = os.environ.get("APPWORLD_ROOT", "").strip()
    if root == "" or not Path(root).is_dir():
        raise QualificationBatchError("APPWORLD_ROOT must be an existing directory")
    configurations = [arm.configuration for arm in design.arms]
    enforce_committed_provenance(*configurations)
    from llm_behavior_ci.runtime.factory import LiveRuntimeFactory
    from llm_behavior_ci.storage import EpisodeStore

    factory = LiveRuntimeFactory.from_endpoints(
        reference=args.endpoint, candidate=args.endpoint
    )
    factory.preflight(
        {"reference": design.arms[0].configuration, "candidate": design.arms[-1].configuration},
        require_distinct_endpoints=False,
    )
    store_path = Path(args.episode_store)
    if any(parent.name == "results" for parent in (store_path, *store_path.parents)):
        raise QualificationBatchError("the episode store cannot be written under results")
    store_path.parent.mkdir(parents=True, exist_ok=True)
    store = EpisodeStore(store_path)
    try:
        summary = run_qualification_batch(
            design,
            runtime_factory=factory,
            checkpoint_path=checkpoint,
            budget=_budget(args, required=False),
            store=store,
        )
    finally:
        store.close()
    return {
        "action": "run",
        "design": design.kind,
        "dispatched": summary.dispatched,
        "skipped": summary.skipped,
        "attempts": summary.attempts,
    }


def _evidence(
    args: argparse.Namespace, design: BatchDesign, checkpoint: Path
) -> dict[str, object]:
    if args.output is None:
        raise QualificationBatchError("--evidence needs --output")
    output = Path(args.output)
    if any(parent.name == "results" for parent in (output, *output.parents)):
        raise QualificationBatchError("evidence cannot be written under results")
    document = load_batch_checkpoint(checkpoint)
    if not document:
        raise QualificationBatchError("checkpoint does not exist")
    specs = []
    for path in args.method_spec:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            for key in ("seeds", "reference_cases", "split"):
                payload.pop(key, None)
        specs.append(method_spec_from_dict(payload))
    lock_inputs: dict[str, LockReportInputs] = {}
    for path in args.lock_inputs:
        inputs = lock_report_inputs_from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
        if inputs.spec.name in lock_inputs:
            raise QualificationBatchError(f"lock inputs repeat {inputs.spec.name}")
        lock_inputs[inputs.spec.name] = inputs
    evidence: dict[str, object] = {
        "record": "qualification_batch_evidence",
        "design": design.kind,
        "identity_hash": design.identity_hash(),
        "attempts": document.get("attempts", {}).get("caps"),
    }
    if design.kind == DEV_FOUR_ARM:
        missing = [
            flag
            for flag, value in (
                ("--margin", args.margin),
                ("--confidence-level", args.confidence_level),
                ("--resamples", args.resamples),
                ("--seed", args.seed),
            )
            if value is None
        ]
        if missing:
            raise QualificationBatchError(f"dev evidence needs {', '.join(missing)}")
        table = dev_outcome_table(design, document)
        evidence["outcomes"] = [
            {"task_index": index, **row} for index, row in enumerate(table)
        ]
        evidence["cusum_scale"] = cusum_scale(design, document)
        evidence["harm_labels"] = dev_harm_labels(
            design,
            document,
            margin=args.margin,
            confidence_level=args.confidence_level,
            resamples=args.resamples,
            seed=args.seed,
        )
        evidence["execution_aa"] = execution_aa_reports(design, document, specs)
        if lock_inputs:
            evidence["lock_reports"] = _lock_documents(
                execution_lock_reports(design, document, lock_inputs)
            )
    else:
        if args.gate_settings is None or args.plan_evidence is None:
            raise QualificationBatchError("plan evidence needs --gate-settings and --plan-evidence")
        settings = GateSettings.from_dict(
            json.loads(Path(args.gate_settings).read_text(encoding="utf-8"))
        )
        plan_evidence = plan_evidence_from_dict(
            json.loads(Path(args.plan_evidence).read_text(encoding="utf-8"))
        )
        evidence["gate_replay"] = plan_aa_gate_replay(
            design, document, settings=settings, plan_evidence=plan_evidence
        )
        series = plan_aa_series(design, document, plan_evidence)
        evidence["plan_series"] = {item.name: len(item.observations) for item in series}
        evidence["plan_aa"] = plan_aa_reports(series, specs)
        if lock_inputs:
            evidence["lock_reports"] = _lock_documents(
                plan_lock_reports(design, document, plan_evidence, lock_inputs)
            )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(evidence, sort_keys=True, indent=1, default=str) + "\n", encoding="utf-8")
    return {"action": "evidence", "design": design.kind, "output": str(output)}


def _lock_documents(reports: Mapping[str, ValidationReport]) -> dict[str, object]:
    return {
        name: {"status": lock_report_status(report), "report": public_validation_summary(report)}
        for name, report in reports.items()
    }


if __name__ == "__main__":
    raise SystemExit(main())
