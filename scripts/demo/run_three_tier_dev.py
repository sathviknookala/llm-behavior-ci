"""Run the connected three-tier dev lifecycle and print its JSON summary.

``synthetic --scenario blocked|healthy|regression`` runs
``lifecycle.connected.run_three_tier_dev`` in process. The gate uses the
synthetic plan-evidence fixture and the fake plan agents of
``scripts/synthetic/run_connected_lifecycle.py``. The service is built by
``serve.py``'s own ``_build_dependencies`` from written settings files and
a train-to-dev task-selection allowance, then given injected execute
worlds, a lazily opened store, and test admission, because the artifact is
``synthetic_fixture``. No network, GPU, AppWorld, or provider is used.

``live`` runs the same function against a ``serve.py`` already listening
on --service-url with --store, and builds plan runtimes with the
``LiveRuntimeFactory``. It refuses a dirty tree, a non-train gate, a
non-dev service, and a store under results/. Exit codes are in ``main``.
Bare invocation exits 2.
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
    run_three_tier_dev,
)
from llm_behavior_ci.lifecycle.offline_gate import GateExecutionError
from llm_behavior_ci.runtime.agent import AgentTurn
from llm_behavior_ci.runtime.appworld import EvaluationResult, TaskContext, ToolResult
from llm_behavior_ci.runtime.episode import RuntimeDependencies
from llm_behavior_ci.runtime.factory import StaticRuntimeFactory
from llm_behavior_ci.service import ServiceDependencies
from llm_behavior_ci.tasks.selection import (
    TaskSet,
    canonical_task_set_bytes,
    task_set_hash_from_bytes,
)

_REPO = Path(__file__).resolve().parents[2]
SCENARIOS = ("blocked", "healthy", "regression")
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
    if scenario == "healthy":
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
    return {"scenario": scenario, "provenance": "synthetic_fixture", **summary}


def _under_results(path: Path) -> bool:
    resolved = path.resolve()
    return any(parent.name == "results" for parent in (resolved, *resolved.parents))


def _load_configuration(raw: str) -> RunConfiguration:
    return RunConfiguration.from_dict(json.loads(Path(raw).read_text(encoding="utf-8")))


def live_inputs(args: argparse.Namespace) -> tuple[GateInputs, DevTraffic]:
    """Load and check every live input; any error here is an invalid invocation."""

    from llm_behavior_ci.experiments.run_config import load_local_task_manifest
    from llm_behavior_ci.lifecycle.offline_gate import (
        plan_evidence_from_dict,
        require_gate_capabilities,
    )
    from llm_behavior_ci.runtime.factory import LiveRuntimeFactory
    from llm_behavior_ci.runtime.provenance import enforce_committed_provenance

    if _under_results(Path(args.store)):
        raise ConfigError("store must not be under results/")
    gate_reference = _load_configuration(args.gate_reference)
    gate_candidate = _load_configuration(args.gate_candidate)
    production = _load_configuration(args.production_config)
    candidate = _load_configuration(args.candidate_config)
    enforce_committed_provenance(gate_reference, gate_candidate, production, candidate)
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
}
EXPECTED_SYNTHETIC_STATUS = {
    "blocked": "blocked",
    "healthy": "promoted",
    "regression": "rolled_back",
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
    verified = summary.get("status") == expected
    print(
        json.dumps(
            {**summary, "expected_status": expected, "verified": verified},
            sort_keys=True,
        )
    )
    return 0 if verified else 1


def _main_live(args: argparse.Namespace) -> int:
    try:
        gate, traffic = live_inputs(args)
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
    print(json.dumps(summary, sort_keys=True))
    return code


def main(argv: Sequence[str] | None = None) -> int:
    """Exit codes: synthetic 0 when the scenario's expected outcome is verified,
    1 when it is not; live 0 promoted, 1 blocked, rolled back, or admission
    refused, 3 canary incomplete after a confirmed cleanup rollback; 2 for an
    invalid invocation or input; 4 for an execution or infrastructure failure.
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
    ):
        live.add_argument(name, required=True)
    live.add_argument("--reference-endpoint", default=None)
    live.add_argument("--candidate-endpoint", default=None)
    try:
        args = parser.parse_args(args_list)
    except SystemExit as error:
        return EXIT_INVALID if error.code is None else int(error.code)
    if args.command == "synthetic":
        return _main_synthetic(args.scenario)
    return _main_live(args)


if __name__ == "__main__":
    raise SystemExit(main())
