"""Test-only Tier 2 integration: the real canary on real agents, without Tier 1.

This is not a release. The offline gate does not run. Admission uses a
``synthetic_fixture`` PASS artifact built here and accepted only because
the isolated service runs with ``admission_mode="test"``
(``authorize_test_gated_candidate``). Every summary says
``test_only_admission: true`` and ``gate_executed: false``. ``serve.py``,
release admission, and the release workflow are unchanged; the same
artifact is refused by ``authorize_gated_candidate``.

``preflight`` loads and checks every input, builds the service
dependencies, and checks the fixture against both admission paths. It
makes no provider call and starts no service.

``serve`` runs the isolated service in the foreground: ``serve.py``'s own
dependency builder, then ``admission_mode="test"``, uvicorn on 127.0.0.1
only, on a port that must be free, with a store that must not exist yet.
The store is opened and used on the main thread, as under ``serve.py``.

``drive`` persists the fixture into that store, admits it through ``POST
/candidates``, sends canary arrivals through ``POST /episodes`` until the
controller promotes or rolls back (``lifecycle.connected``'s
``admit_and_run_canary``), checks every persisted pair, and writes a
public ``tier2-canary-test-v1`` summary. No Tier 3 arrivals are sent; a
promoted service and its store are left for Tier 3.

Exit codes: 0 the run completed and every integrity check passed,
whatever the canary decided; 1 an integrity check failed; 2 invalid
input; 4 an execution failure. Bare invocation exits 2.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import socket
import sys
from collections.abc import Sequence
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path

from llm_behavior_ci.config import (
    ConfigError,
    RunConfiguration,
    new_run_identity,
    run_configuration_hash,
)
from llm_behavior_ci.experiments.protocol import (
    ProtocolError,
    authorize_gated_candidate,
    authorize_test_gated_candidate,
    task_selection_allowance_from_dict,
)
from llm_behavior_ci.lifecycle.connected import (
    ConnectedLifecycleError,
    DevTraffic,
    ServiceClient,
    admit_and_run_canary,
    require_fresh_store,
    require_registered,
)
from llm_behavior_ci.lifecycle.validation_artifact import (
    ValidationArtifact,
    build_validation_artifact,
)
from llm_behavior_ci.records import StatisticalEvidence, assert_public_payload
from llm_behavior_ci.storage import EpisodeStore

_REPO = Path(__file__).resolve().parents[2]
HOST = "127.0.0.1"
RECORD = "tier2_canary_integration_test"
VERSION = "tier2-canary-test-v1"
FIXTURE_METHOD = "test_only_admission_fixture"
EXIT_INVALID = 2
EXIT_FAILURE = 4


def _load_script(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, _REPO / relative)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


serve = _load_script("serve_for_tier2_canary_test", "scripts/service/serve.py")


@dataclass(frozen=True)
class Tier2Inputs:
    gate_reference: RunConfiguration
    gate_candidate: RunConfiguration
    traffic: DevTraffic


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
    allowance: object,
) -> dict[str, bool]:
    """The fixture passes test admission and is refused by release admission."""

    production = inputs.traffic.production
    candidate = inputs.traffic.candidate
    try:
        authorize_test_gated_candidate(artifact, production, candidate, allowance)  # type: ignore[arg-type]
        test_accepts = True
    except ProtocolError:
        test_accepts = False
    try:
        authorize_gated_candidate(artifact, production, candidate, allowance)  # type: ignore[arg-type]
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
    eligible = [
        pair
        for pair in pairs
        if pair.reference.evaluator_outcome is not None
        and pair.candidate.evaluator_outcome is not None
    ]
    counts: dict[str, object] = {
        "pairs": len(pairs),
        "eligible_pairs": len(eligible),
        "missing_evaluator_outcome": len(pairs) - len(eligible),
        "reference_success": sum(
            1 for pair in eligible if pair.reference.evaluator_outcome.success  # type: ignore[union-attr]
        ),
        "candidate_success": sum(
            1 for pair in eligible if pair.candidate.evaluator_outcome.success  # type: ignore[union-attr]
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
        "pair_episodes_persisted": all(
            pair.reference.episode.episode_id in finished
            and pair.candidate.episode.episode_id in finished
            for pair in pairs
        ),
    }
    return counts, checks


def run_tier2_test(
    inputs: Tier2Inputs,
    *,
    store_path: Path,
    service: ServiceClient,
    allowance: object,
    git_commit: str,
) -> dict[str, object]:
    """Admit the fixture through the service and run the real canary.

    ``service`` is an HTTP client on a service whose store is
    ``store_path``. The fixture is written to that store, then ``POST
    /candidates`` decides: an isolated test service admits it, a release
    service refuses it (``admission_refused``).
    """

    traffic = inputs.traffic
    if traffic.production_task_ids:
        raise ConnectedLifecycleError("the Tier 2 test sends no Tier 3 arrivals")
    require_fresh_store(store_path)
    require_registered(service, traffic)
    artifact = fixture_artifact(inputs)
    checks = admission_checks(artifact, inputs, allowance)
    writer = EpisodeStore(store_path)
    try:
        writer.append_validation_artifact(artifact)
    finally:
        writer.close()
    stage, state = admit_and_run_canary(
        traffic,
        store_path=store_path,
        service=service,
        artifact_id=artifact.artifact_id,
    )
    production_hash = run_configuration_hash(traffic.production)
    candidate_hash = run_configuration_hash(traffic.candidate)
    if "deployment" not in stage:
        deployment = service.get("/deployment").json()
        stage["deployment"] = {
            key: deployment.get(key)
            for key in (
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
        }
    raw_pair_ids = stage.pop("canary_pair_ids", [])
    pair_ids = [str(item) for item in raw_pair_ids] if isinstance(raw_pair_ids, list) else []
    pairs, pair_checks = _pair_checks(store_path, pair_ids, production_hash, candidate_hash)
    checks.update(pair_checks)
    reader = EpisodeStore(store_path)
    try:
        decisions = [
            item
            for item in reader.load_deployment_decisions()
            if item.decision in {"admit", "promote", "rollback"}
        ]
    finally:
        reader.close()
    deployment = stage["deployment"]
    assert isinstance(deployment, dict)
    status = stage["status"]
    if status != "admission_refused":
        checks["controller_observed_every_pair"] = (
            deployment.get("candidate_episodes_served") == len(pair_ids)
            or deployment.get("served_before_rollback") == len(pair_ids)
        )
        checks["admission_used_fixture"] = bool(decisions) and (
            decisions[0].decision == "admit"
            and decisions[0].evidence_artifact_id == artifact.artifact_id
            and decisions[0].evidence_source == "synthetic_fixture"
        )
    if status == "promoted":
        checks["deployment_reflects_promotion"] = (
            deployment.get("state") == "PROMOTED"
            and deployment.get("serving_configuration_hash") == candidate_hash
            and decisions[-1].decision == "promote"
        )
    elif status in {"rolled_back", "canary_incomplete"}:
        checks["candidate_traffic_closed"] = (
            deployment.get("state") == "ROLLED_BACK"
            and deployment.get("serving_configuration_hash") == production_hash
            and decisions[-1].decision == "rollback"
            and (status == "rolled_back" or deployment.get("admission") == "rollback_requested")
        )
    else:
        checks["refused_admission_left_no_decision"] = not decisions
    summary: dict[str, object] = {
        "record": RECORD,
        "version": VERSION,
        "test_only_admission": True,
        "gate_executed": False,
        "admission_evidence_source": "synthetic_fixture",
        "git_commit": git_commit,
        "status": status,
        "final_state": state,
        "configurations": {
            "production": production_hash,
            "candidate": candidate_hash,
            "gate_reference": run_configuration_hash(inputs.gate_reference),
            "gate_candidate": run_configuration_hash(inputs.gate_candidate),
        },
        "fixture_artifact_id": artifact.artifact_id,
        "admission": stage.get("admission"),
        "tier2": stage.get("tier2"),
        "cleanup": stage.get("cleanup"),
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
        "pairs": pairs,
        "checks": checks,
        "verified": all(checks.values()),
    }
    assert_public_payload(summary)
    return summary


def _under_results(path: Path) -> bool:
    resolved = path.resolve()
    return any(parent.name == "results" for parent in (resolved, *resolved.parents))


def _configuration(raw: str) -> RunConfiguration:
    return RunConfiguration.from_dict(json.loads(Path(raw).read_text(encoding="utf-8")))


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


def _allowance(args: argparse.Namespace) -> object:
    return task_selection_allowance_from_dict(
        json.loads(Path(args.task_selection_allowance).read_text(encoding="utf-8"))
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


def isolated_service_dependencies(args: argparse.Namespace):
    """``serve.py``'s dependencies with test admission; nothing else changes."""

    built = serve._build_dependencies(service_arguments(args))
    return replace(built, admission_mode="test")


def preflight(args: argparse.Namespace, *, check_provenance: bool = True) -> dict[str, object]:
    from tempfile import TemporaryDirectory

    inputs = load_inputs(args, check_provenance=check_provenance)
    require_isolated_target(Path(args.store), int(args.port))
    allowance = _allowance(args)
    artifact = fixture_artifact(inputs)
    checks = admission_checks(artifact, inputs, allowance)
    with TemporaryDirectory() as scratch:
        scratch_args = argparse.Namespace(**{**vars(args), "store": str(Path(scratch) / "x.sqlite")})
        dependencies = isolated_service_dependencies(scratch_args)
        dependencies.store.close()
    canary = dependencies.canary_settings
    arrivals = len(inputs.traffic.canary_task_ids)
    checks["service_dependencies_built"] = dependencies.admission_mode == "test"
    checks["canary_fraction_is_one"] = canary.fraction == 1.0
    checks["arrivals_cover_horizon"] = arrivals >= canary.stopping_rule.horizon_episodes
    summary: dict[str, object] = {
        "record": RECORD + "_preflight",
        "version": VERSION,
        "test_only_admission": True,
        "gate_executed": False,
        "provider_calls": 0,
        "configurations": {
            "production": run_configuration_hash(inputs.traffic.production),
            "candidate": run_configuration_hash(inputs.traffic.candidate),
            "gate_reference": run_configuration_hash(inputs.gate_reference),
            "gate_candidate": run_configuration_hash(inputs.gate_candidate),
        },
        "git_commit": inputs.traffic.production.git_commit,
        "canary": canary.to_dict(),
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


_INVALID = (ConfigError, ConnectedLifecycleError, ProtocolError, OSError, ValueError, KeyError, TypeError)


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

    from llm_behavior_ci.service import create_app

    uvicorn.run(create_app(dependencies), host=HOST, port=int(args.port))
    return 0


def _main_drive(args: argparse.Namespace) -> int:
    import httpx

    output = Path(args.summary)
    try:
        inputs = load_inputs(args)
        allowance = _allowance(args)
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
                git_commit=inputs.traffic.production.git_commit,
            )
        with output.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps(summary, sort_keys=True) + "\n")
    except Exception as error:
        return _fail(error, EXIT_FAILURE)
    print(json.dumps(summary, sort_keys=True))
    return 0 if summary["verified"] else 1


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
            "--store",
            "--port",
        ):
            command.add_argument(flag, required=True)
        if name != "drive":
            for flag in ("--canary-settings", "--monitor-settings", "--frozen-reference"):
                command.add_argument(flag, required=True)
            command.add_argument("--canary-assignment-seed", default="3")
        else:
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
