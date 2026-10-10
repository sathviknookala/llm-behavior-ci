"""Test-only Tier 2 integration: the real canary on real agents, without Tier 1.

This is not a release. The offline gate does not run. Admission uses a
``synthetic_fixture`` PASS artifact built here and accepted only because
the isolated service runs with ``admission_mode="test"``
(``authorize_test_gated_candidate``). Every summary says
``test_only_admission: true`` and ``gate_executed: false``. ``serve.py``,
release admission, and the release workflow are unchanged; the same
artifact is refused by ``authorize_gated_candidate``.

``preflight`` loads and checks every input, builds the service
dependencies exactly as ``serve`` does, and checks that their identity
matches the intended files and that the fixture passes test admission
and fails release admission. It makes no provider call and starts no
service.

``serve`` runs the isolated service in the foreground: ``serve.py``'s own
dependency builder, then ``admission_mode="test"``, uvicorn on 127.0.0.1
only, on a free port, with a store that must not exist yet. Two
test-only routes exist on this app only: ``GET /test-only/identity``
reports what the running service loaded, and ``GET /test-only/canary``
reports how many paired outcomes the stopping rule received.

``drive`` refuses to start unless the running service's identity equals
the intended one (configurations, canary settings, assignment seed,
allowance, monitor reference, test admission). Only then does it persist
the fixture, admit it through ``POST /candidates``, and send canary
arrivals through ``POST /episodes`` (``admit_and_run_canary``). It then
checks every stored pair and writes a public ``tier2-canary-test-v1``
summary. No Tier 3 arrivals are sent.

Exit codes: 0 every integrity check passed, pairs ran, and the stopping
rule decided; 3 every integrity check passed and pairs ran, but the
stopping rule did not decide (statistical integration incomplete); 1 an
integrity check failed, admission was refused, or no pair ran; 2 invalid
input or a service identity mismatch, before any model call; 4 an
execution failure. Bare invocation exits 2.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import socket
import sys
from collections.abc import Sequence
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path

from llm_behavior_ci.config import (
    CanarySettings,
    ConfigError,
    MonitorSettings,
    RunConfiguration,
    new_run_identity,
    run_configuration_hash,
)
from llm_behavior_ci.experiments.protocol import (
    ProtocolError,
    TaskSelectionAllowance,
    authorize_gated_candidate,
    authorize_test_gated_candidate,
    task_selection_allowance_document,
    task_selection_allowance_from_dict,
)
from llm_behavior_ci.lifecycle.connected import (
    ConnectedLifecycleError,
    DevTraffic,
    ServiceClient,
    admit_and_run_canary,
    deployment_view,
    require_fresh_store,
    require_registered,
)
from llm_behavior_ci.lifecycle.monitoring import FrozenReference
from llm_behavior_ci.lifecycle.validation_artifact import (
    ValidationArtifact,
    build_validation_artifact,
)
from llm_behavior_ci.records import StatisticalEvidence, assert_public_payload
from llm_behavior_ci.service import ServiceDependencies, create_app
from llm_behavior_ci.storage import EpisodeStore

_REPO = Path(__file__).resolve().parents[2]
HOST = "127.0.0.1"
RECORD = "tier2_canary_integration_test"
VERSION = "tier2-canary-test-v1"
FIXTURE_METHOD = "test_only_admission_fixture"
IDENTITY_ROUTE = "/test-only/identity"
CANARY_ROUTE = "/test-only/canary"
STOPPING_RULE_METHODS = frozenset({"stopping_rule", "stopping_rule_alarm"})
EXIT_INVALID = 2
EXIT_INCOMPLETE = 3
EXIT_FAILURE = 4


def _load_script(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, _REPO / relative)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


serve = _load_script("serve_for_tier2_canary_test", "scripts/service/serve.py")


class ServiceIdentityError(ConnectedLifecycleError):
    """The running service is not the intended isolated test service."""


@dataclass(frozen=True)
class Tier2Inputs:
    gate_reference: RunConfiguration
    gate_candidate: RunConfiguration
    traffic: DevTraffic


def service_identity(
    *,
    admission_mode: str,
    production: RunConfiguration,
    candidate: RunConfiguration | None,
    canary_settings: CanarySettings,
    assignment_seed: int,
    allowance: TaskSelectionAllowance | None,
    frozen_reference: FrozenReference,
) -> dict[str, object]:
    """The settings a test service runs with, and their SHA-256."""

    document: dict[str, object] = {
        "admission_mode": admission_mode,
        "production_configuration_hash": run_configuration_hash(production),
        "candidate_configuration_hash": (
            None if candidate is None else run_configuration_hash(candidate)
        ),
        "canary_settings": canary_settings.to_dict(),
        "canary_assignment_seed": int(assignment_seed),
        "task_selection_allowance": (
            None if allowance is None else task_selection_allowance_document(allowance)
        ),
        "monitor_reference": {
            "configuration_hash": frozen_reference.configuration_hash,
            "baselines": [[name, value] for name, value in frozen_reference.baselines],
        },
    }
    canonical = json.dumps(document, sort_keys=True, separators=(",", ":"))
    return {**document, "identity_sha256": hashlib.sha256(canonical.encode()).hexdigest()}


def identity_from_dependencies(dependencies: ServiceDependencies) -> dict[str, object]:
    return service_identity(
        admission_mode=dependencies.admission_mode,
        production=dependencies.registry.production,
        candidate=dependencies.registry.candidate,
        canary_settings=dependencies.canary_settings,
        assignment_seed=dependencies.canary_assignment_seed,
        allowance=dependencies.task_selection_allowance,
        frozen_reference=dependencies.monitor.reference,
    )


def isolated_app(dependencies: ServiceDependencies):
    """``create_app`` plus the two test-only routes; never used by ``serve.py``."""

    app = create_app(dependencies)
    identity = identity_from_dependencies(dependencies)
    state = app.state.service_state
    horizon = dependencies.canary_settings.stopping_rule.horizon_episodes

    @app.get(IDENTITY_ROUTE)
    async def get_identity() -> dict[str, object]:
        return identity

    @app.get(CANARY_ROUTE)
    async def get_canary() -> dict[str, object]:
        with state.deployment_lock:
            controller = state.controller
            if controller is None:
                return {
                    "admitted": False,
                    "stopping_rule_observations": 0,
                    "horizon_episodes": horizon,
                }
            snapshot = controller.snapshot()
            return {
                "admitted": True,
                "state": snapshot.state,
                "stopping_rule_observations": controller.detector_updates,
                "candidate_episodes_served": snapshot.candidate_episodes_served,
                "horizon_episodes": horizon,
            }

    return app


def require_service_identity(service: ServiceClient, expected: dict[str, object]) -> None:
    """Refuse a service whose loaded settings differ from the intended ones."""

    response = service.get(IDENTITY_ROUTE)
    if response.status_code != 200:
        raise ServiceIdentityError(
            "service has no test-only identity; it is not the isolated test service"
        )
    actual = response.json()
    if not isinstance(actual, dict):
        raise ServiceIdentityError("service identity is not an object")
    mismatched = sorted(
        key for key in set(expected) | set(actual) if expected.get(key) != actual.get(key)
    )
    if mismatched:
        raise ServiceIdentityError(
            "running service does not match the intended settings: " + ", ".join(mismatched)
        )


def fixture_artifact(inputs: Tier2Inputs) -> ValidationArtifact:
    """A PASS artifact labelled ``synthetic_fixture``; no gate produced it."""

    reference = inputs.gate_reference
    candidate = inputs.gate_candidate
    return build_validation_artifact(
        outcome="PASS",
        reason_codes=(),
        reference=reference,
        candidate=candidate,
        reference_run=new_run_identity(reference),
        candidate_run=new_run_identity(candidate),
        task_set_hash=reference.task.task_set_hash,
        task_split=reference.task.split,
        statistics=(
            StatisticalEvidence(
                method=FIXTURE_METHOD,
                split=reference.task.split,
                configuration_hash=run_configuration_hash(candidate),
                estimate=0.0,
                sample_size=1,
                unit="fixture",
                reference_configuration_hash=run_configuration_hash(reference),
            ),
        ),
        evidence_source="synthetic_fixture",
        created_at=datetime.now(timezone.utc),
    )


def admission_checks(
    artifact: ValidationArtifact,
    inputs: Tier2Inputs,
    allowance: TaskSelectionAllowance | None,
) -> dict[str, bool]:
    """The fixture passes test admission and is refused by release admission."""

    production = inputs.traffic.production
    candidate = inputs.traffic.candidate
    try:
        authorize_test_gated_candidate(artifact, production, candidate, allowance)
        test_accepts = True
    except ProtocolError:
        test_accepts = False
    try:
        authorize_gated_candidate(artifact, production, candidate, allowance)
        release_refuses = False
    except ProtocolError:
        release_refuses = True
    return {
        "fixture_labelled_synthetic": artifact.evidence_source == "synthetic_fixture",
        "test_admission_accepts_fixture": test_accepts,
        "release_admission_refuses_fixture": release_refuses,
    }


def _pair_checks(
    store_path: Path,
    pair_ids: Sequence[str],
    production_hash: str,
    candidate_hash: str,
) -> tuple[dict[str, object], dict[str, bool]]:
    reader = EpisodeStore(store_path)
    try:
        pairs = [reader.load_pair(pair_id) for pair_id in pair_ids]
        finished = {item.episode.episode_id for item in reader.load_finished_episodes()}
    finally:
        reader.close()
    persisted = [
        pair
        for pair in pairs
        if pair.reference.episode.episode_id in finished
        and pair.candidate.episode.episode_id in finished
    ]
    both = [
        pair
        for pair in pairs
        if pair.reference.evaluator_outcome is not None
        and pair.candidate.evaluator_outcome is not None
    ]
    counts: dict[str, object] = {
        "pairs_executed": len(pair_ids),
        "pairs_persisted": len(persisted),
        "pairs_with_both_evaluator_outcomes": len(both),
        "pairs_missing_an_evaluator_outcome": len(pairs) - len(both),
        "reference_success": sum(
            1 for pair in both if pair.reference.evaluator_outcome.success  # type: ignore[union-attr]
        ),
        "candidate_success": sum(
            1 for pair in both if pair.candidate.evaluator_outcome.success  # type: ignore[union-attr]
        ),
        "candidate_status": dict(
            sorted(
                {
                    status: sum(1 for pair in pairs if pair.candidate.status == status)
                    for status in {pair.candidate.status for pair in pairs}
                }.items()
            )
        ),
    }
    checks = {
        "pairs_use_registered_hashes": all(
            pair.reference.run.configuration_hash == production_hash
            and pair.candidate.run.configuration_hash == candidate_hash
            for pair in pairs
        ),
        "pairs_have_distinct_episodes": all(
            pair.reference.episode.episode_id != pair.candidate.episode.episode_id
            for pair in pairs
        ),
        "pair_episodes_persisted": len(persisted) == len(pair_ids),
    }
    return counts, checks


def run_tier2_test(
    inputs: Tier2Inputs,
    *,
    store_path: Path,
    service: ServiceClient,
    allowance: TaskSelectionAllowance | None,
    expected_identity: dict[str, object],
    git_commit: str,
) -> dict[str, object]:
    """Verify the service identity, admit the fixture, and run the real canary.

    ``service`` is an HTTP client on an ``isolated_app`` whose store is
    ``store_path``. Nothing is written and no model is called until the
    service's identity equals ``expected_identity``.
    """

    traffic = inputs.traffic
    if traffic.production_task_ids:
        raise ConnectedLifecycleError("the Tier 2 test sends no Tier 3 arrivals")
    require_fresh_store(store_path)
    require_registered(service, traffic)
    require_service_identity(service, expected_identity)
    artifact = fixture_artifact(inputs)
    admission = admission_checks(artifact, inputs, allowance)
    writer = EpisodeStore(store_path)
    try:
        writer.append_validation_artifact(artifact)
    finally:
        writer.close()
    stage = admit_and_run_canary(
        traffic,
        store_path=store_path,
        service=service,
        artifact_id=artifact.artifact_id,
    )
    public = stage.public
    production_hash = run_configuration_hash(traffic.production)
    candidate_hash = run_configuration_hash(traffic.candidate)
    deployment = public.get("deployment")
    if not isinstance(deployment, dict):
        deployment = deployment_view(service.get("/deployment").json())
    canary = service.get(CANARY_ROUTE).json()
    pairs, integrity = _pair_checks(store_path, stage.pair_ids, production_hash, candidate_hash)
    reader = EpisodeStore(store_path)
    try:
        decisions = [
            item
            for item in reader.load_deployment_decisions()
            if item.decision in {"admit", "promote", "rollback"}
        ]
    finally:
        reader.close()
    status = public["status"]
    admitted = status != "admission_refused"
    admission_status = public.get("admission")
    admission_verification: dict[str, object] = {
        **admission,
        "admitted": admitted,
        "status_code": (
            admission_status.get("status_code") if isinstance(admission_status, dict) else None
        ),
    }
    final = decisions[-1] if decisions and decisions[-1].decision != "admit" else None
    observations = int(canary.get("stopping_rule_observations", 0))
    horizon = int(canary.get("horizon_episodes", 0))
    executed = len(stage.pair_ids)
    statistical = (
        "complete"
        if final is not None and final.method in STOPPING_RULE_METHODS and observations > 0
        else "incomplete"
    )
    canary_execution: dict[str, object] = {
        **pairs,
        "pairs_eligible_for_stopping_rule": pairs["pairs_with_both_evaluator_outcomes"],
        "stopping_rule_observations": observations,
        "horizon_episodes": horizon,
        "horizon_reached": observations >= horizon > 0,
        "controller_decision": None if final is None else final.decision,
        "controller_decision_method": None if final is None else final.method,
        "final_state": deployment.get("state"),
        "statistical_integration": statistical,
    }
    if admitted:
        integrity["controller_served_every_pair"] = (
            deployment.get("candidate_episodes_served") == executed
            or deployment.get("served_before_rollback") == executed
        )
        integrity["stopping_rule_received_every_eligible_pair"] = (
            observations == pairs["pairs_with_both_evaluator_outcomes"]
        )
        integrity["admission_used_fixture"] = bool(decisions) and (
            decisions[0].decision == "admit"
            and decisions[0].evidence_artifact_id == artifact.artifact_id
            and decisions[0].evidence_source == "synthetic_fixture"
        )
    if status == "promoted":
        integrity["deployment_reflects_promotion"] = (
            deployment.get("state") == "PROMOTED"
            and deployment.get("serving_configuration_hash") == candidate_hash
            and final is not None
            and final.decision == "promote"
        )
    elif status in {"rolled_back", "canary_incomplete"}:
        integrity["candidate_traffic_closed"] = (
            deployment.get("state") == "ROLLED_BACK"
            and deployment.get("serving_configuration_hash") == production_hash
            and deployment.get("admission") == "rollback_requested"
            and final is not None
            and final.decision == "rollback"
        )
    else:
        integrity["refused_admission_left_no_decision"] = not decisions
    summary: dict[str, object] = {
        "record": RECORD,
        "version": VERSION,
        "test_only_admission": True,
        "gate_executed": False,
        "admission_evidence_source": "synthetic_fixture",
        "git_commit": git_commit,
        "status": status,
        "configurations": {
            "production": production_hash,
            "candidate": candidate_hash,
            "gate_reference": run_configuration_hash(inputs.gate_reference),
            "gate_candidate": run_configuration_hash(inputs.gate_candidate),
        },
        "service_identity_sha256": expected_identity.get("identity_sha256"),
        "fixture_artifact_id": artifact.artifact_id,
        "admission_verification": admission_verification,
        "tier2_executed": admitted and executed > 0,
        "canary_execution": canary_execution,
        "tier2": public.get("tier2"),
        "cleanup": public.get("cleanup"),
        "deployment": deployment,
        "lifecycle_decisions": [
            {
                "decision": item.decision,
                "method": item.method,
                "configuration_hash": item.configuration_hash,
                "reference_configuration_hash": item.reference_configuration_hash,
                "sample_size": item.sample_size,
                "evidence_source": item.evidence_source,
            }
            for item in decisions
        ],
        "integrity_checks": integrity,
        "integrity_verified": all(integrity.values()) and all(admission.values()),
    }
    assert_public_payload(summary)
    return summary


def exit_code_for(summary: dict[str, object]) -> int:
    """0 only when integrity holds, pairs ran, and the stopping rule decided."""

    if not summary.get("integrity_verified") or not summary.get("tier2_executed"):
        return 1
    execution = summary.get("canary_execution")
    if not isinstance(execution, dict) or execution.get("statistical_integration") != "complete":
        return EXIT_INCOMPLETE
    return 0


def _under_results(path: Path) -> bool:
    resolved = path.resolve()
    return any(parent.name == "results" for parent in (resolved, *resolved.parents))


def _json(raw: str) -> object:
    return json.loads(Path(raw).read_text(encoding="utf-8"))


def _configuration(raw: str) -> RunConfiguration:
    return RunConfiguration.from_dict(_json(raw))


def load_inputs(args: argparse.Namespace, *, check_provenance: bool = True) -> Tier2Inputs:
    from llm_behavior_ci.experiments.run_config import load_local_task_manifest
    from llm_behavior_ci.runtime.provenance import enforce_committed_provenance

    gate_reference = _configuration(args.gate_reference)
    gate_candidate = _configuration(args.gate_candidate)
    production = _configuration(args.production_config)
    candidate = _configuration(args.candidate_config)
    if check_provenance:
        enforce_committed_provenance(gate_reference, gate_candidate, production, candidate)
    if gate_reference.task.split != "train" or gate_candidate.task.split != "train":
        raise ConfigError("the fixture's gate identities must be train configurations")
    dev = load_local_task_manifest(Path(args.dev_task_set)).task_set
    if dev.split != "dev" or dev.task_set_hash != production.task.task_set_hash:
        raise ConfigError("--dev-task-set must be the dev task set production serves")
    arrivals = int(args.canary_arrivals)
    if arrivals < 1 or arrivals > len(dev.task_ids):
        raise ConfigError("--canary-arrivals must be between 1 and the dev task count")
    return Tier2Inputs(
        gate_reference=gate_reference,
        gate_candidate=gate_candidate,
        traffic=DevTraffic(
            production=production,
            candidate=candidate,
            canary_task_ids=tuple(dev.task_ids[:arrivals]),
            production_task_ids=(),
        ),
    )


def load_allowance(args: argparse.Namespace) -> TaskSelectionAllowance:
    return task_selection_allowance_from_dict(_json(args.task_selection_allowance))


def intended_identity(args: argparse.Namespace, inputs: Tier2Inputs) -> dict[str, object]:
    """The identity the test service must report, from the intended files."""

    monitor = MonitorSettings.from_dict(_json(args.monitor_settings))
    frozen = serve._frozen_reference(_json(args.frozen_reference))
    if monitor.reference_configuration_hash != frozen.configuration_hash:
        raise ConfigError("monitor settings reference hash must match the frozen reference")
    return service_identity(
        admission_mode="test",
        production=inputs.traffic.production,
        candidate=inputs.traffic.candidate,
        canary_settings=CanarySettings.from_dict(_json(args.canary_settings)),
        assignment_seed=int(args.canary_assignment_seed),
        allowance=load_allowance(args),
        frozen_reference=frozen,
    )


def port_is_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        try:
            probe.bind((HOST, port))
        except OSError:
            return False
    return True


def require_isolated_target(store: Path, port: int) -> None:
    """The test service gets a new store and a free localhost port."""

    if _under_results(store):
        raise ConfigError("--store must not be under results/")
    if store.exists():
        raise ConfigError("--store must not exist yet; the test service needs a new store")
    if not 1024 <= port <= 65535:
        raise ConfigError("--port must be an unprivileged port")
    if not port_is_free(port):
        raise ConfigError(f"port {port} is already in use")


def service_arguments(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        production_config=args.production_config,
        candidate_config=args.candidate_config,
        store=args.store,
        max_in_flight=1,
        shutdown_timeout_seconds=30.0,
        canary_settings=args.canary_settings,
        monitor_settings=args.monitor_settings,
        frozen_reference=args.frozen_reference,
        dedup_seconds=0.0,
        canary_assignment_seed=int(args.canary_assignment_seed),
        production_base_url=None,
        candidate_base_url=None,
        task_metadata=None,
        distributional_monitors=None,
        slice_attribution=False,
        task_selection_allowance=args.task_selection_allowance,
    )


def isolated_service_dependencies(args: argparse.Namespace) -> ServiceDependencies:
    """``serve.py``'s dependencies with test admission; nothing else changes."""

    built = serve._build_dependencies(service_arguments(args))
    return replace(built, admission_mode="test")


def preflight(args: argparse.Namespace, *, check_provenance: bool = True) -> dict[str, object]:
    from tempfile import TemporaryDirectory

    inputs = load_inputs(args, check_provenance=check_provenance)
    require_isolated_target(Path(args.store), int(args.port))
    allowance = load_allowance(args)
    expected = intended_identity(args, inputs)
    artifact = fixture_artifact(inputs)
    checks: dict[str, bool] = dict(admission_checks(artifact, inputs, allowance))
    with TemporaryDirectory() as scratch:
        scratch_args = argparse.Namespace(
            **{**vars(args), "store": str(Path(scratch) / "preflight.sqlite")}
        )
        dependencies = isolated_service_dependencies(scratch_args)
        dependencies.store.close()
    built = identity_from_dependencies(dependencies)
    canary = dependencies.canary_settings
    arrivals = len(inputs.traffic.canary_task_ids)
    checks["service_identity_matches_intended"] = built == expected
    checks["canary_fraction_is_one"] = canary.fraction == 1.0
    checks["arrivals_cover_horizon"] = arrivals >= canary.stopping_rule.horizon_episodes
    summary: dict[str, object] = {
        "record": RECORD + "_preflight",
        "version": VERSION,
        "test_only_admission": True,
        "gate_executed": False,
        "provider_calls": 0,
        "git_commit": inputs.traffic.production.git_commit,
        "service_identity": expected,
        "canary_arrivals": arrivals,
        "max_execute_episodes": 2 * arrivals,
        "checks": checks,
        "verified": all(checks.values()),
    }
    assert_public_payload(summary)
    return summary


def _fail(error: BaseException, code: int) -> int:
    print(str(error) or type(error).__name__, file=sys.stderr)
    return code


_INVALID = (
    ConfigError,
    ConnectedLifecycleError,
    ProtocolError,
    OSError,
    ValueError,
    KeyError,
    TypeError,
)


def _main_preflight(args: argparse.Namespace) -> int:
    try:
        summary = preflight(args)
    except _INVALID as error:
        return _fail(error, EXIT_INVALID)
    print(json.dumps(summary, sort_keys=True))
    return 0 if summary["verified"] else 1


def _main_serve(args: argparse.Namespace) -> int:
    try:
        load_inputs(args)
        require_isolated_target(Path(args.store), int(args.port))
        dependencies = isolated_service_dependencies(args)
    except _INVALID as error:
        return _fail(error, EXIT_INVALID)
    import uvicorn

    uvicorn.run(isolated_app(dependencies), host=HOST, port=int(args.port))
    return 0


def _main_drive(args: argparse.Namespace) -> int:
    import httpx

    output = Path(args.summary)
    try:
        inputs = load_inputs(args)
        allowance = load_allowance(args)
        expected = intended_identity(args, inputs)
        store = Path(args.store)
        if not store.is_file():
            raise ConfigError("--store must be the running test service's store")
        if output.exists() or _under_results(output):
            raise ConfigError("--summary must be a new file outside results/")
    except _INVALID as error:
        return _fail(error, EXIT_INVALID)
    try:
        with httpx.Client(base_url=f"http://{HOST}:{int(args.port)}", timeout=None) as client:
            summary = run_tier2_test(
                inputs,
                store_path=store,
                service=client,
                allowance=allowance,
                expected_identity=expected,
                git_commit=inputs.traffic.production.git_commit,
            )
        with output.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps(summary, sort_keys=True) + "\n")
    except ServiceIdentityError as error:
        return _fail(error, EXIT_INVALID)
    except Exception as error:
        return _fail(error, EXIT_FAILURE)
    print(json.dumps(summary, sort_keys=True))
    return exit_code_for(summary)


def main(argv: Sequence[str] | None = None) -> int:
    args_list = list(sys.argv[1:] if argv is None else argv)
    if not args_list:
        return EXIT_INVALID
    parser = argparse.ArgumentParser(description="Test-only Tier 2 canary integration run.")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("preflight", "serve", "drive"):
        command = commands.add_parser(name)
        for flag in (
            "--gate-reference",
            "--gate-candidate",
            "--production-config",
            "--candidate-config",
            "--dev-task-set",
            "--task-selection-allowance",
            "--canary-arrivals",
            "--canary-settings",
            "--monitor-settings",
            "--frozen-reference",
            "--store",
            "--port",
        ):
            command.add_argument(flag, required=True)
        command.add_argument("--canary-assignment-seed", default="3")
        if name == "drive":
            command.add_argument("--summary", required=True)
    try:
        args = parser.parse_args(args_list)
    except SystemExit as error:
        return EXIT_INVALID if error.code is None else int(error.code)
    if args.command == "preflight":
        return _main_preflight(args)
    if args.command == "serve":
        return _main_serve(args)
    return _main_drive(args)


if __name__ == "__main__":
    raise SystemExit(main())
