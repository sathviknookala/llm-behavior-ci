"""Run the connected three-tier dev lifecycle and print its JSON summary.

``synthetic --scenario blocked|healthy|regression|production_regression`` runs
``lifecycle.connected.run_three_tier_dev`` in process. The gate uses the
synthetic plan-evidence fixture and the fake plan agents of
``scripts/synthetic/run_connected_lifecycle.py``. The service is built by
``serve.py``'s own ``_build_dependencies`` from written settings files and
a train-to-dev task-selection allowance, then given injected execute
worlds, a lazily opened store, and test admission, because the artifact is
``synthetic_fixture``. No network, GPU, AppWorld, or provider is used.
``production_regression`` is ``healthy`` on a service started with
``--post-promotion-rollback rollback_on_alert``: the canary promotes, the
production monitor alerts on the regressed tasks, and the service restores
the previous production.

Each synthetic run also verifies the deployment state, the configuration
left serving, the persisted gate artifact, the stored lifecycle decisions,
and the episodes' configurations; any mismatch exits 1.

``live`` runs the same function against a ``serve.py`` already listening
on --service-url with --store, and builds plan runtimes with the
``LiveRuntimeFactory``. It refuses a dirty tree, a candidate not built at
HEAD, a non-train gate, a non-dev service, and a store under results/. The
store must already be the running service's store and hold nothing: one
release, one store. --release-summary names a new file for the public
``release-summary-v3`` record. --previous-release and --previous-store name
release N's record and store; release N+1 is refused before any model call
unless the store confirms the record and the production configuration is
the one release N left serving, unchanged and at its own commit. A
production at an older commit is accepted only with that lineage and a
safe runtime transition (``lifecycle.release``). --max-model-calls caps the
gate's model calls and --max-episodes the episodes the run can start.

``release --release-dir DIR`` runs ``live`` from a provisioned release
(``scripts/ci/release_ctl.py provision``): every path, arrival count,
endpoint, budget, and the previous release come from DIR's verified
``release.json``, the release must have been provisioned at HEAD from the
current ``LATEST``, and after its summary is written the release becomes
``LATEST``. Exit codes are in ``main``. Bare invocation exits 2.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Callable

from llm_behavior_ci.config import (
    CanarySettings,
    ConfigError,
    GateSettings,
    MonitorSettings,
    RunConfiguration,
    StoppingRule,
    run_configuration_hash,
)
from llm_behavior_ci.experiments.protocol import (
    task_selection_allowance_document,
    task_selection_allowance_for,
)
from llm_behavior_ci.lifecycle.connected import (
    ConnectedLifecycleError,
    DevTraffic,
    GateInputs,
    release_summary,
    require_fresh_store,
    require_release_lineage,
    run_three_tier_dev,
)
from llm_behavior_ci.lifecycle.offline_gate import GateExecutionError
from llm_behavior_ci.lifecycle.release import (
    ReleaseError,
    RuntimeTransition,
    require_current_lineage,
    require_previous_summary,
    require_release_provenance,
    require_runtime_transition,
    service_url,
    verify_release_manifest,
    verify_release_record,
    write_latest,
)
from llm_behavior_ci.runtime.agent import AgentTurn
from llm_behavior_ci.runtime.appworld import EvaluationResult, TaskContext, ToolResult
from llm_behavior_ci.runtime.budget import BudgetedRuntimeFactory, ModelCallBudget
from llm_behavior_ci.runtime.episode import RuntimeDependencies
from llm_behavior_ci.runtime.factory import StaticRuntimeFactory
from llm_behavior_ci.runtime.provenance import RepositoryState, read_repository_state
from llm_behavior_ci.service import ServiceDependencies
from llm_behavior_ci.tasks.selection import (
    TaskSet,
    canonical_task_set_bytes,
    task_set_hash_from_bytes,
)

_REPO = Path(__file__).resolve().parents[2]
SCENARIOS = ("blocked", "healthy", "regression", "production_regression")
CANARY_TASK_IDS = tuple(f"dev-canary-{index}" for index in range(4))
STEADY_TASK_IDS = tuple(f"dev-steady-{index}" for index in range(3))
REGRESS_TASK_IDS = tuple(f"dev-regress-{index}" for index in range(4))
PRODUCTION_TASK_IDS = STEADY_TASK_IDS + REGRESS_TASK_IDS
WRONG_ACTION = "calendar.delete_event()"
GATE_SETTINGS = GateSettings(
    confidence_level=0.9,
    bootstrap_resamples=40,
    score_margin=-0.02,
    kl_limit_nats=0.05,
    mmd_bandwidth=1.0,
    mmd_permutations=99,
    mmd_alpha=0.05,
    plan_format_version="plan-v1",
)


def _load_script(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, _REPO / relative)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


synthetic = _load_script(
    "connected_lifecycle_fixtures", "scripts/synthetic/run_connected_lifecycle.py"
)
serve = _load_script("serve_for_three_tier_dev", "scripts/service/serve.py")


class EmptyPlanAgent(synthetic.PlanAgent):
    """A plan agent that emits no plan, which the gate blocks as ``plan_run_failed``."""

    def next_turn(self, *, tool_output: str | None) -> AgentTurn:
        turn = super().next_turn(tool_output=tool_output)
        return replace(turn, output_text="")


def _dev_task_set() -> TaskSet:
    tasks = tuple((task_id, None) for task_id in CANARY_TASK_IDS + PRODUCTION_TASK_IDS)
    payload = canonical_task_set_bytes(
        appworld_version="0.1.3.post1",
        split="dev",
        selection_rule="deterministic_sample",
        selection_seed=20261009,
        tasks=tasks,
    )
    return TaskSet(
        appworld_version="0.1.3.post1",
        split="dev",
        selection_rule="deterministic_sample",
        selection_seed=20261009,
        task_count=len(tasks),
        scenario_count=len(tasks),
        task_ids=tuple(task_id for task_id, _ in tasks),
        scenario_ids=tuple(None for _ in tasks),
        task_set_hash=task_set_hash_from_bytes(payload),
    )


def _on_dev(configuration: RunConfiguration, dev: TaskSet) -> RunConfiguration:
    return replace(
        configuration,
        task=replace(
            configuration.task,
            split=dev.split,
            selection_seed=dev.selection_seed,
            task_count=dev.task_count,
            task_set_hash=dev.task_set_hash,
        ),
    )


def candidate_faulted(scenario: str, task_id: str) -> bool:
    """Whether the candidate agent issues the wrong call on ``task_id``."""

    if scenario == "regression":
        return True
    if scenario in {"healthy", "production_regression"}:
        return task_id in REGRESS_TASK_IDS
    return False


class ActionJudgedSession(synthetic.FakeSession):
    """A world whose evaluator passes only when every executed call was the right one."""

    def __init__(self, task_id: str) -> None:
        super().__init__(task_id)
        self._actions: list[str] = []

    def execute(self, action: str) -> ToolResult:
        self._actions.append(action)
        return super().execute(action)

    def evaluate(self) -> EvaluationResult:
        success = bool(self._actions) and all(
            action == synthetic._ACTION for action in self._actions
        )
        return EvaluationResult(
            success=success,
            passed_requirements=1 if success else 0,
            total_requirements=1,
            difficulty=1,
        )


class ScenarioAgent(synthetic.FakeAgent):
    """The execute agent: the right call, or ``WRONG_ACTION`` on a faulted task."""

    def __init__(self, clock: Callable, *, faulted: Callable[[str], bool]) -> None:
        super().__init__([], clock=clock)
        self._faulted = faulted

    def begin(self, context: TaskContext, config: RunConfiguration) -> None:
        super().begin(context, config)
        action = WRONG_ACTION if self._faulted(context.task_id) else synthetic._ACTION
        self._turns = [
            AgentTurn(
                prompt_text=synthetic._PROMPT,
                output_text=action,
                top_k_logprobs=synthetic._LOGPROBS,
                generated_token_count=len(synthetic._LOGPROBS),
                latency_seconds=0.1,
                started_at=synthetic._START,
                action=action,
                app_name="calendar",
                api_name=action.split(".", 1)[1].split("(", 1)[0],
            )
        ]
        self._index = 0


def _world(
    clock: Callable,
    *,
    candidate_hash: str,
    scenario: str,
) -> Callable[[RunConfiguration], RuntimeDependencies]:
    def factory(configuration: RunConfiguration) -> RuntimeDependencies:
        is_candidate = run_configuration_hash(configuration) == candidate_hash
        return RuntimeDependencies(
            session_factory=ActionJudgedSession,
            agent=ScenarioAgent(
                clock,
                faulted=lambda task_id: is_candidate and candidate_faulted(scenario, task_id),
            ),
            clock=clock,
        )

    return factory


def _write(root: Path, name: str, document: object) -> str:
    path = root / name
    path.write_text(json.dumps(document), encoding="utf-8")
    return str(path)


@dataclass(frozen=True)
class SyntheticLifecycle:
    gate: GateInputs
    traffic: DevTraffic
    store_path: Path
    dependencies: ServiceDependencies


def build_synthetic(scenario: str, root: Path) -> SyntheticLifecycle:
    """Every input of one synthetic scenario, with the service built by ``serve.py``."""

    if scenario not in SCENARIOS:
        raise ConfigError(f"scenario must be one of: {', '.join(SCENARIOS)}")
    train_set = synthetic._task_set()
    train_reference = synthetic._config(train_set, run_seed=7)
    train_candidate = synthetic._candidate(train_reference)
    dev = _dev_task_set()
    production = _on_dev(train_reference, dev)
    candidate = _on_dev(train_candidate, dev)
    production_hash = run_configuration_hash(production)

    plan_clock = synthetic._clock()
    candidate_agent = (
        EmptyPlanAgent(plan_clock) if scenario == "blocked" else synthetic.PlanAgent(plan_clock)
    )
    gate = GateInputs(
        reference=train_reference,
        candidate=train_candidate,
        task_set=train_set,
        settings=GATE_SETTINGS,
        plan_evidence=synthetic._plan_evidence(),
        runtime_factory=StaticRuntimeFactory(
            reference=synthetic._plan_runtime(plan_clock),
            candidate=RuntimeDependencies(
                session_factory=lambda task_id: synthetic.FakeSession(task_id),
                agent=candidate_agent,
                clock=plan_clock,
            ),
        ),
    )

    store_path = root / "lifecycle.sqlite"
    canary_settings = CanarySettings(
        fraction=1.0,
        outcome_delay_seconds=0.0,
        harm_margin=0.1,
        stopping_rule=StoppingRule(
            name="fixed_window",
            alpha=0.05,
            horizon_episodes=len(CANARY_TASK_IDS),
        ),
        metric_orientation="higher_is_better",
        promotion_policy="horizon_reached_without_harm",
    )
    monitor_settings = MonitorSettings(
        reference_configuration_hash=production_hash,
        outcome_delay_seconds=0.0,
        signals=("task_success",),
        stopping_rules=(
            StoppingRule(name="cusum", alpha=0.1, horizon_episodes=50, threshold=2.0),
        ),
    )
    args = argparse.Namespace(
        production_config=_write(root, "production.json", production.to_dict()),
        candidate_config=_write(root, "candidate.json", candidate.to_dict()),
        store=str(store_path),
        max_in_flight=4,
        shutdown_timeout_seconds=1.0,
        canary_settings=_write(root, "canary.json", canary_settings.to_dict()),
        monitor_settings=_write(root, "monitor.json", monitor_settings.to_dict()),
        frozen_reference=_write(
            root,
            "frozen_reference.json",
            {"configuration_hash": production_hash, "baselines": [["task_success", 0.9]]},
        ),
        dedup_seconds=0.0,
        canary_assignment_seed=0,
        production_base_url="http://production.invalid",
        candidate_base_url="http://candidate.invalid",
        task_metadata=None,
        distributional_monitors=None,
        slice_attribution=False,
        task_selection_allowance=_write(
            root,
            "task_selection_allowance.json",
            task_selection_allowance_document(task_selection_allowance_for(train_reference)),
        ),
        post_promotion_rollback=(
            "rollback_on_alert" if scenario == "production_regression" else "record_only"
        ),
    )
    built = serve._build_dependencies(args)
    built.store.close()
    service_clock = synthetic._clock()
    dependencies = replace(
        built,
        runtime_factory=_world(
            service_clock,
            candidate_hash=run_configuration_hash(candidate),
            scenario=scenario,
        ),
        store=synthetic.LazyStore(store_path),
        clock=service_clock,
        admission_mode="test",
    )
    traffic = DevTraffic(
        production=production,
        candidate=candidate,
        canary_task_ids=CANARY_TASK_IDS,
        production_task_ids=PRODUCTION_TASK_IDS,
    )
    return SyntheticLifecycle(
        gate=gate,
        traffic=traffic,
        store_path=store_path,
        dependencies=dependencies,
    )


def run_synthetic(scenario: str, root: Path) -> dict[str, object]:
    from fastapi.testclient import TestClient

    from llm_behavior_ci.service import create_app

    lifecycle = build_synthetic(scenario, root)
    with TestClient(create_app(lifecycle.dependencies)) as client:
        summary = run_three_tier_dev(
            lifecycle.gate,
            lifecycle.traffic,
            store_path=lifecycle.store_path,
            service=client,
        )
    return {
        "scenario": scenario,
        "provenance": "synthetic_fixture",
        **summary,
        "checks": synthetic_checks(scenario, summary, lifecycle.traffic),
    }


def synthetic_checks(
    scenario: str, summary: dict[str, object], traffic: DevTraffic
) -> dict[str, bool]:
    """Whether a synthetic run reached its scenario's state, serving
    configuration, stored decisions, and persisted evidence."""

    status, state, serving_role, decisions = EXPECTED_SYNTHETIC[scenario]
    hashes = {
        "production": run_configuration_hash(traffic.production),
        "candidate": run_configuration_hash(traffic.candidate),
    }
    deployment = summary.get("deployment")
    evidence = summary.get("evidence")
    tier1 = summary.get("tier1")
    if not isinstance(deployment, dict) or not isinstance(evidence, dict):
        return {"summary": False}
    if not isinstance(tier1, dict):
        return {"summary": False}
    stored = [
        item.get("decision")
        for item in evidence.get("deployment_decisions", [])
        if item.get("decision") in {"admit", "promote", "rollback", "production_rollback"}
    ]
    serving = deployment.get("serving_configuration_hash") or deployment.get(
        "production_configuration_hash"
    )
    episodes = evidence.get("episodes_by_configuration_hash", {})
    artifact = evidence.get("validation_artifact", {})
    return {
        "status": summary.get("status") == status,
        "deployment_state": deployment.get("state") == state,
        "serving_configuration": serving == hashes[serving_role],
        "stored_decisions": stored == decisions,
        "gate_artifact_persisted": artifact.get("artifact_id") == tier1.get("artifact_id"),
        "episodes_persisted": (
            evidence.get("finished_episodes") == 0
            if status == "blocked"
            else bool(episodes) and set(episodes) <= set(hashes.values())
        ),
    }


def _under_results(path: Path) -> bool:
    resolved = path.resolve()
    return any(parent.name == "results" for parent in (resolved, *resolved.parents))


def _load_configuration(raw: str) -> RunConfiguration:
    return RunConfiguration.from_dict(json.loads(Path(raw).read_text(encoding="utf-8")))


def live_inputs(args: argparse.Namespace) -> tuple[GateInputs, DevTraffic]:
    """Load and check every live input; any error here is an invalid invocation.

    Commit provenance is ``release_inputs``'s: production may be carried
    from an earlier commit only through a verified release lineage.
    """

    from llm_behavior_ci.experiments.run_config import load_local_task_manifest
    from llm_behavior_ci.lifecycle.offline_gate import (
        plan_evidence_from_dict,
        require_gate_capabilities,
    )
    from llm_behavior_ci.runtime.factory import LiveRuntimeFactory

    if _under_results(Path(args.store)):
        raise ConfigError("store must not be under results/")
    if not Path(args.store).is_file():
        raise ConfigError("--store must be the running service's existing store")
    require_fresh_store(Path(args.store))
    gate_reference = _load_configuration(args.gate_reference)
    gate_candidate = _load_configuration(args.gate_candidate)
    production = _load_configuration(args.production_config)
    candidate = _load_configuration(args.candidate_config)
    train = load_local_task_manifest(Path(args.task_set)).task_set
    dev = load_local_task_manifest(Path(args.dev_task_set)).task_set
    if dev.split != "dev" or dev.task_set_hash != production.task.task_set_hash:
        raise ConfigError("--dev-task-set must be the dev task set production serves")
    canary_count = int(args.canary_arrivals)
    production_count = int(args.production_arrivals)
    if canary_count < 1 or production_count < 0:
        raise ConfigError("arrival counts must be positive")
    if canary_count + production_count > len(dev.task_ids):
        raise ConfigError("arrival counts exceed the dev task set")
    plan_evidence = plan_evidence_from_dict(
        json.loads(Path(args.plan_evidence).read_text(encoding="utf-8"))
    )
    require_gate_capabilities(
        gate_reference, gate_candidate, plan_evidence.required_statistics
    )
    gate = GateInputs(
        reference=gate_reference,
        candidate=gate_candidate,
        task_set=train,
        settings=GateSettings.from_dict(
            json.loads(Path(args.gate_settings).read_text(encoding="utf-8"))
        ),
        plan_evidence=plan_evidence,
        runtime_factory=LiveRuntimeFactory.from_endpoints(
            reference=args.reference_endpoint,
            candidate=args.candidate_endpoint,
        ),
    )
    traffic = DevTraffic(
        production=production,
        candidate=candidate,
        canary_task_ids=tuple(dev.task_ids[:canary_count]),
        production_task_ids=tuple(
            dev.task_ids[canary_count : canary_count + production_count]
        ),
    )
    return gate, traffic


@dataclass(frozen=True)
class ReleaseContext:
    lineage: dict[str, object] | None
    transition: RuntimeTransition


def release_inputs(
    args: argparse.Namespace,
    gate: GateInputs,
    traffic: DevTraffic,
    *,
    state: RepositoryState | None = None,
) -> ReleaseContext:
    """Check the release-summary path, the lineage, commit provenance, and the runtime transition.

    With --previous-release, --previous-store must be release N's store;
    the record is re-derived from it before the lineage is accepted.
    """

    output = Path(args.release_summary)
    if output.exists():
        raise ConfigError("--release-summary must name a new file")
    if _under_results(output):
        raise ConfigError("--release-summary must not be under results/")
    lineage = None
    if args.previous_release is not None:
        if getattr(args, "previous_store", None) is None:
            raise ConfigError("--previous-release needs --previous-store")
        previous = json.loads(Path(args.previous_release).read_text(encoding="utf-8"))
        serving = verify_release_record(previous, Path(args.previous_store))
        lineage = require_release_lineage(previous, traffic.production, serving=serving)
    resolved = state if state is not None else read_repository_state()
    require_release_provenance(
        release_configurations=(gate.candidate, traffic.candidate),
        gate_reference=gate.reference,
        production=traffic.production,
        state=resolved,
        carried=lineage is not None,
    )
    transition = require_runtime_transition(traffic.production.git_commit, resolved.head)
    return ReleaseContext(lineage=lineage, transition=transition)


def apply_budgets(args: argparse.Namespace, gate: GateInputs, traffic: DevTraffic) -> GateInputs:
    """Refuse a run that can start more episodes than --max-episodes, and cap gate model calls."""

    max_episodes = getattr(args, "max_episodes", None)
    if max_episodes is not None:
        bound = (
            2 * len(gate.task_set.task_ids)
            + 2 * len(traffic.canary_task_ids)
            + len(traffic.production_task_ids)
        )
        if bound > int(max_episodes):
            raise ConfigError(
                f"this run can start {bound} episodes, above --max-episodes {max_episodes}"
            )
    max_model_calls = getattr(args, "max_model_calls", None)
    if max_model_calls is None:
        return gate
    return replace(
        gate,
        runtime_factory=BudgetedRuntimeFactory(
            gate.runtime_factory, ModelCallBudget(limit=int(max_model_calls))
        ),
    )


def write_release_summary(path: Path, document: dict[str, object]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(document, sort_keys=True) + "\n")


def run_live(
    gate: GateInputs,
    traffic: DevTraffic,
    *,
    store_path: Path,
    service_url: str,
) -> dict[str, object]:
    import httpx

    with httpx.Client(base_url=service_url, timeout=None) as client:
        return run_three_tier_dev(gate, traffic, store_path=store_path, service=client)


LIVE_EXIT_CODES = {
    "promoted": 0,
    "blocked": 1,
    "rolled_back": 1,
    "admission_refused": 1,
    "canary_incomplete": 3,
    "production_rolled_back": 5,
}
EXPECTED_SYNTHETIC = {
    "blocked": ("blocked", None, "production", []),
    "healthy": ("promoted", "PROMOTED", "candidate", ["admit", "promote"]),
    "regression": ("rolled_back", "ROLLED_BACK", "production", ["admit", "rollback"]),
    "production_regression": (
        "production_rolled_back",
        "PRODUCTION_ROLLED_BACK",
        "production",
        ["admit", "promote", "production_rollback"],
    ),
}
EXPECTED_SYNTHETIC_STATUS = {
    scenario: expected[0] for scenario, expected in EXPECTED_SYNTHETIC.items()
}
EXIT_INVALID = 2
EXIT_FAILURE = 4


def _fail(error: BaseException, code: int) -> int:
    print(str(error) or type(error).__name__, file=sys.stderr)
    return code


def _main_synthetic(scenario: str) -> int:
    try:
        with TemporaryDirectory() as temporary:
            summary = run_synthetic(scenario, Path(temporary))
    except Exception as error:
        return _fail(error, EXIT_FAILURE)
    expected = EXPECTED_SYNTHETIC_STATUS[scenario]
    checks = summary.get("checks")
    verified = (
        summary.get("status") == expected
        and isinstance(checks, dict)
        and bool(checks)
        and all(checks.values())
    )
    print(
        json.dumps(
            {**summary, "expected_status": expected, "verified": verified},
            sort_keys=True,
        )
    )
    return 0 if verified else 1


def _main_live(args: argparse.Namespace, *, state: RepositoryState | None = None) -> int:
    try:
        gate, traffic = live_inputs(args)
        release = release_inputs(args, gate, traffic, state=state)
        gate = apply_budgets(args, gate, traffic)
    except (
        ConfigError,
        ConnectedLifecycleError,
        GateExecutionError,
        OSError,
        ValueError,
        KeyError,
        TypeError,
    ) as error:
        return _fail(error, EXIT_INVALID)
    try:
        summary = run_live(
            gate, traffic, store_path=Path(args.store), service_url=args.service_url
        )
    except Exception as error:
        return _fail(error, EXIT_FAILURE)
    code = LIVE_EXIT_CODES.get(str(summary.get("status")))
    if code is None:
        return _fail(RuntimeError(f"unknown lifecycle status {summary.get('status')}"), EXIT_FAILURE)
    try:
        write_release_summary(
            Path(args.release_summary),
            release_summary(
                summary,
                traffic,
                lineage=release.lineage,
                release_commit=release.transition.release_commit,
                runtime=release.transition.to_dict(),
            ),
        )
    except Exception as error:
        return _fail(error, EXIT_FAILURE)
    print(json.dumps(summary, sort_keys=True))
    return code


def release_namespace(
    release_dir: Path, manifest: dict[str, object], previous_summary: Path | None
) -> argparse.Namespace:
    """The ``live`` arguments of a provisioned release, read only from its manifest."""

    budget = manifest["budget"]
    endpoints = manifest["gate_endpoints"]
    assert isinstance(budget, dict) and isinstance(endpoints, dict)
    return argparse.Namespace(
        gate_reference=str(release_dir / "gate_reference.json"),
        gate_candidate=str(release_dir / "gate_candidate.json"),
        task_set=str(release_dir / "train_task_set.json"),
        gate_settings=str(release_dir / "gate_settings.json"),
        plan_evidence=str(release_dir / "plan_evidence.json"),
        production_config=str(release_dir / "production.json"),
        candidate_config=str(release_dir / "candidate.json"),
        dev_task_set=str(release_dir / "dev_task_set.json"),
        canary_arrivals=str(budget["canary_arrivals"]),
        production_arrivals=str(budget["production_arrivals"]),
        service_url=service_url(manifest),
        store=str(release_dir / "episodes.sqlite"),
        release_summary=str(release_dir / "release_summary.json"),
        previous_release=None if previous_summary is None else str(previous_summary),
        previous_store=(
            None if previous_summary is None else str(previous_summary.parent / "episodes.sqlite")
        ),
        reference_endpoint=endpoints.get("reference"),
        candidate_endpoint=endpoints.get("candidate"),
        max_model_calls=budget["max_gate_model_calls"],
        max_episodes=budget["max_episodes"],
    )


def _main_release(release_dir: Path, *, state: RepositoryState | None = None) -> int:
    try:
        resolved = state if state is not None else read_repository_state()
        manifest = verify_release_manifest(release_dir, head=resolved.head)
        require_current_lineage(release_dir.parent, manifest)
        previous_summary = require_previous_summary(release_dir.parent, manifest)
        args = release_namespace(release_dir, manifest, previous_summary)
    except (ConfigError, ConnectedLifecycleError, ReleaseError, OSError, ValueError, KeyError) as error:
        return _fail(error, EXIT_INVALID)
    summary_path = Path(args.release_summary)
    existed = summary_path.exists()
    code = _main_live(args, state=resolved)
    if not existed and summary_path.is_file():
        try:
            write_latest(release_dir.parent, release_dir.name)
        except Exception as error:
            return _fail(error, EXIT_FAILURE)
    return code


def main(argv: Sequence[str] | None = None) -> int:
    """Exit codes: synthetic 0 when the scenario's expected outcome is verified,
    1 when it is not; live and release 0 promoted, 1 blocked, rolled back, or admission
    refused, 3 canary incomplete after a confirmed cleanup rollback, 5 promoted
    and then rolled back to the previous production during Tier 3; 2 for an
    invalid invocation or input, including a broken release lineage; 4 for an
    execution or infrastructure failure, or a release summary that could not
    be written.
    """

    args_list = list(sys.argv[1:] if argv is None else argv)
    if not args_list:
        return EXIT_INVALID
    parser = argparse.ArgumentParser(description="Run the connected three-tier dev lifecycle.")
    commands = parser.add_subparsers(dest="command", required=True)
    synthetic_parser = commands.add_parser("synthetic")
    synthetic_parser.add_argument("--scenario", required=True, choices=SCENARIOS)
    live = commands.add_parser("live")
    for name in (
        "--gate-reference",
        "--gate-candidate",
        "--task-set",
        "--gate-settings",
        "--plan-evidence",
        "--production-config",
        "--candidate-config",
        "--dev-task-set",
        "--canary-arrivals",
        "--production-arrivals",
        "--service-url",
        "--store",
        "--release-summary",
    ):
        live.add_argument(name, required=True)
    live.add_argument("--previous-release", default=None)
    live.add_argument("--previous-store", default=None)
    live.add_argument("--reference-endpoint", default=None)
    live.add_argument("--candidate-endpoint", default=None)
    live.add_argument("--max-model-calls", default=None, type=int)
    live.add_argument("--max-episodes", default=None, type=int)
    release = commands.add_parser("release")
    release.add_argument("--release-dir", required=True)
    try:
        args = parser.parse_args(args_list)
    except SystemExit as error:
        return EXIT_INVALID if error.code is None else int(error.code)
    if args.command == "synthetic":
        return _main_synthetic(args.scenario)
    if args.command == "release":
        return _main_release(Path(args.release_dir))
    return _main_live(args)


if __name__ == "__main__":
    raise SystemExit(main())
