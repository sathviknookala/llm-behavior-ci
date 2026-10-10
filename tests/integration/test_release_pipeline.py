from __future__ import annotations

import importlib.util
import io
import json
import os
import subprocess
import sys
import unittest
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

try:
    from fastapi.testclient import TestClient
except ImportError as error:
    raise ImportError(
        "fastapi is required for tests.integration.test_release_pipeline"
    ) from error

from llm_behavior_ci.config import RunConfiguration, run_configuration_hash
from llm_behavior_ci.experiments.run_config import load_local_task_manifest
from llm_behavior_ci.lifecycle.connected import (
    DevTraffic,
    GateInputs,
    ReleaseLineageError,
    release_summary,
    require_release_lineage,
    run_three_tier_dev,
)
from llm_behavior_ci.lifecycle.offline_gate import plan_evidence_from_dict
from llm_behavior_ci.lifecycle.release import (
    MANIFEST_FILE,
    STORE_FILE,
    SUMMARY_FILE,
    ReleaseError,
    load_release_spec,
    provision_release,
    read_latest,
    release_spec_from_dict,
    require_current_lineage,
    require_deployed,
    require_previous_summary,
    require_release_provenance,
    require_runtime_transition,
    service_argv,
    verify_release_manifest,
    verify_release_record,
    write_latest,
)
from llm_behavior_ci.runtime.budget import BudgetedRuntimeFactory, ModelCallBudget
from llm_behavior_ci.runtime.episode import RuntimeDependencies
from llm_behavior_ci.runtime.factory import StaticRuntimeFactory
from llm_behavior_ci.runtime.provenance import (
    RepositoryState,
    execution_fingerprint,
    is_ancestor,
)
from llm_behavior_ci.service import create_app
from llm_behavior_ci.storage import DeploymentDecisionRecord, EpisodeStore
from llm_behavior_ci.tasks.selection import TaskSet

_ROOT = Path(__file__).resolve().parents[2]


def _load(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, _ROOT / relative)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


demo = _load("run_three_tier_dev_release_tests", "scripts/demo/run_three_tier_dev.py")
synthetic = demo.synthetic
serve = demo.serve
release_ctl = _load("release_ctl_tests", "scripts/ci/release_ctl.py")

_NOW = datetime(2026, 10, 10, tzinfo=timezone.utc)


def _git(repo: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-c", "user.name=release-test", "-c", "user.email=release@test.invalid", *args],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    )
    return completed.stdout.strip()


def _commit(repo: Path, message: str, files: dict[str, str]) -> str:
    for relative, content in files.items():
        path = repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", message)
    return _git(repo, "rev-parse", "HEAD")


def _template(configuration: RunConfiguration) -> str:
    document = configuration.to_dict()
    return json.dumps({"model": document["model"], "agent": document["agent"]}, indent=2)


def _manifest_document(task_set: TaskSet) -> dict[str, object]:
    return {
        "appworld_version": task_set.appworld_version,
        "split": task_set.split,
        "selection_rule": task_set.selection_rule,
        "selection_seed": task_set.selection_seed,
        "task_count": task_set.task_count,
        "scenario_count": task_set.scenario_count,
        "task_ids": list(task_set.task_ids),
        "scenario_ids": list(task_set.scenario_ids),
        "task_set_hash": task_set.task_set_hash,
    }


def _with_prompt(configuration: RunConfiguration, version: str) -> RunConfiguration:
    return replace(
        configuration,
        agent=replace(
            configuration.agent,
            prompt=replace(configuration.agent.prompt, prompt_version=version),
        ),
    )


class ReleaseRepository:
    """A throwaway git repository holding the release inputs, plus local private files."""

    def __init__(self, root: Path) -> None:
        self.repo = root / "repo"
        self.repo.mkdir()
        self.local = root / "local"
        self.local.mkdir()
        self.releases = root / "releases"
        self.releases.mkdir()
        _git(self.repo, "init", "-q", "-b", "main")
        base = synthetic._config(synthetic._task_set())
        self.commits: dict[str, str] = {}
        self.commits["A"] = _commit(
            self.repo,
            "A",
            {
                "README.md": "release test\n",
                "requirements.txt": "-r environments/service-requirements.txt\n",
                "environments/service-requirements.txt": "smolagents==1.22.0\n",
                "src/llm_behavior_ci/config.py": "CONFIG = 1\n",
                "src/llm_behavior_ci/records.py": "RECORDS = 1\n",
                "src/llm_behavior_ci/runtime/agent.py": "AGENT = 1\n",
                "src/llm_behavior_ci/tasks/selection.py": "SELECTION = 1\n",
                "scripts/ci/release_ctl.py": "\n",
                "configs/models/production.json": _template(base),
                "configs/models/candidate.json": _template(_with_prompt(base, "prompt-v2")),
                "configs/models/candidate_v3.json": _template(_with_prompt(base, "prompt-v3")),
            },
        )
        train = synthetic._task_set()
        dev = demo._dev_task_set()
        (self.local / "train.json").write_text(json.dumps(_manifest_document(train)))
        (self.local / "dev.json").write_text(json.dumps(_manifest_document(dev)))
        (self.local / "plan_evidence.json").write_text(
            (_ROOT / "configs" / "fixtures" / "synthetic_plan_evidence.json").read_text()
        )

    def commit(self, label: str, files: dict[str, str]) -> str:
        self.commits[label] = _commit(self.repo, label, files)
        return self.commits[label]

    def spec(self, **overrides: object) -> dict[str, object]:
        document: dict[str, object] = {
            "version": "release-spec-v1",
            "candidate": {"template": "configs/models/candidate.json", "fault": None},
            "initial_production": {"template": "configs/models/production.json", "fault": None},
            "run_seed": 7,
            "protocol_hash": "e" * 64,
            "train_manifest": "train.json",
            "dev_manifest": "dev.json",
            "gate_settings": demo.GATE_SETTINGS.to_dict(),
            "plan_evidence": "plan_evidence.json",
            "canary_settings": {
                "fraction": 1.0,
                "outcome_delay_seconds": 0.0,
                "harm_margin": 0.1,
                "stopping_rule": {
                    "name": "fixed_window",
                    "alpha": 0.05,
                    "horizon_episodes": len(demo.CANARY_TASK_IDS),
                },
                "metric_orientation": "higher_is_better",
                "promotion_policy": "horizon_reached_without_harm",
            },
            "monitor_settings": {
                "outcome_delay_seconds": 0.0,
                "signals": ["task_success"],
                "stopping_rules": [
                    {"name": "cusum", "alpha": 0.1, "horizon_episodes": 50, "threshold": 2.0}
                ],
            },
            "frozen_reference": {
                "baselines": [["task_success", 0.9]],
                "slice_baselines": [],
                "source": None,
            },
            "service": {
                "host": "127.0.0.1",
                "port": 8765,
                "max_in_flight": 4,
                "shutdown_timeout_seconds": 1.0,
                "dedup_seconds": 0.0,
                "canary_assignment_seed": 0,
                "production_base_url": "http://production.invalid",
                "candidate_base_url": "http://candidate.invalid",
                "post_promotion_rollback": "record_only",
            },
            "gate_endpoints": {"reference": None, "candidate": None},
            "budget": {
                "canary_arrivals": len(demo.CANARY_TASK_IDS),
                "production_arrivals": len(demo.PRODUCTION_TASK_IDS),
                "max_episodes": 40,
                "max_gate_model_calls": 100,
                "max_service_model_calls": 100,
            },
        }
        document.update(overrides)
        return document

    def write_spec(self, **overrides: object) -> Path:
        path = self.local / "spec.json"
        path.write_text(json.dumps(self.spec(**overrides)))
        return path

    def provision(
        self,
        release_id: str,
        commit: str,
        *,
        deployment: dict[str, object] | None = None,
        **overrides: object,
    ) -> dict[str, object]:
        return provision_release(
            load_release_spec(self.write_spec(**overrides)),
            release_root=self.releases,
            release_id=release_id,
            state=RepositoryState(head=self.commits[commit], dirty_paths=()),
            deployment=deployment,
            now=_NOW,
            root=self.repo,
        )


def _configuration(path: Path) -> RunConfiguration:
    return RunConfiguration.from_dict(json.loads(path.read_text()))


def release_dependencies(
    release_dir: Path, *, scenario: str, model_calls: int | None = None
):
    """The service ``serve.py`` builds from the release manifest, on injected worlds.

    ``model_calls`` replaces the manifest's service budget limit, keeping
    the calls already persisted in the store as spent.
    """

    manifest = verify_release_manifest(release_dir)
    args = serve.build_parser().parse_args(service_argv(release_dir, manifest))
    built = serve._build_dependencies(args)
    built.store.close()
    if model_calls is not None:
        built = replace(
            built,
            model_call_budget=ModelCallBudget(
                limit=model_calls, spent=built.model_call_budget.spent
            ),
        )
    clock = synthetic._clock()
    candidate_hash = manifest["configurations"]["candidate"]["hash"]
    world = demo._world(clock, candidate_hash=candidate_hash, scenario=scenario)
    factory = (
        world
        if built.model_call_budget is None
        else BudgetedRuntimeFactory(world, built.model_call_budget)
    )
    return replace(
        built,
        runtime_factory=factory,
        store=synthetic.LazyStore(release_dir / STORE_FILE),
        clock=clock,
        admission_mode="test",
    )


def release_gate(release_dir: Path, *, blocked: bool = False) -> GateInputs:
    clock = synthetic._clock()
    candidate_agent = demo.EmptyPlanAgent(clock) if blocked else synthetic.PlanAgent(clock)
    return GateInputs(
        reference=_configuration(release_dir / "gate_reference.json"),
        candidate=_configuration(release_dir / "gate_candidate.json"),
        task_set=load_local_task_manifest(release_dir / "train_task_set.json").task_set,
        settings=demo.GATE_SETTINGS,
        plan_evidence=plan_evidence_from_dict(
            json.loads((release_dir / "plan_evidence.json").read_text())
        ),
        runtime_factory=StaticRuntimeFactory(
            reference=synthetic._plan_runtime(clock),
            candidate=RuntimeDependencies(
                session_factory=lambda task_id: synthetic.FakeSession(task_id),
                agent=candidate_agent,
                clock=clock,
            ),
        ),
    )


def release_traffic(release_dir: Path) -> DevTraffic:
    dev = load_local_task_manifest(release_dir / "dev_task_set.json").task_set
    canary = len(demo.CANARY_TASK_IDS)
    return DevTraffic(
        production=_configuration(release_dir / "production.json"),
        candidate=_configuration(release_dir / "candidate.json"),
        canary_task_ids=tuple(dev.task_ids[:canary]),
        production_task_ids=tuple(dev.task_ids[canary:]),
    )


def run_release(
    release_dir: Path, *, scenario: str = "healthy", blocked: bool = False
) -> tuple[dict[str, object], dict[str, object]]:
    """Run one provisioned release end to end and record it as ``LATEST``."""

    manifest = verify_release_manifest(release_dir)
    traffic = release_traffic(release_dir)
    with TestClient(create_app(release_dependencies(release_dir, scenario=scenario))) as client:
        summary = run_three_tier_dev(
            release_gate(release_dir, blocked=blocked),
            traffic,
            store_path=release_dir / STORE_FILE,
            service=client,
        )
        deployment = client.get("/deployment").json()
    previous = manifest["previous"]
    record = release_summary(
        summary,
        traffic,
        lineage=None if previous is None else previous["lineage"],
        release_commit=manifest["release_commit"],
        runtime=manifest["runtime"],
    )
    (release_dir / SUMMARY_FILE).write_text(json.dumps(record, sort_keys=True) + "\n")
    write_latest(release_dir.parent, release_dir.name)
    return record, deployment


class GitFactsTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = TemporaryDirectory()
        self.addCleanup(self._tmpdir.cleanup)
        self.repository = ReleaseRepository(Path(self._tmpdir.name))
        self.repo = self.repository.repo

    def test_execution_fingerprint_follows_execution_code_only(self) -> None:
        a = self.repository.commits["A"]
        b = self.repository.commit("B", {"README.md": "docs only\n"})
        c = self.repository.commit("C", {"src/llm_behavior_ci/runtime/agent.py": "AGENT = 2\n"})
        self.assertEqual(execution_fingerprint(a, self.repo), execution_fingerprint(b, self.repo))
        self.assertNotEqual(execution_fingerprint(b, self.repo), execution_fingerprint(c, self.repo))
        self.assertTrue(is_ancestor(a, c, self.repo))
        self.assertFalse(is_ancestor(c, a, self.repo))

    def test_runtime_transitions(self) -> None:
        a = self.repository.commits["A"]
        b = self.repository.commit("B", {"README.md": "docs only\n"})
        same = require_runtime_transition(a, a, self.repo)
        self.assertEqual(same.basis, "same_commit")
        identical = require_runtime_transition(a, b, self.repo)
        self.assertEqual(identical.basis, "identical_execution")
        c = self.repository.commit("C", {"src/llm_behavior_ci/config.py": "CONFIG = 2\n"})
        with self.assertRaisesRegex(ReleaseError, "execution code changed"):
            require_runtime_transition(a, c, self.repo)
        with self.assertRaisesRegex(ReleaseError, "not an ancestor"):
            require_runtime_transition(c, a, self.repo)
        declared = {
            "version": "runtime-compatibility-v1",
            "declarations": [
                {
                    "execution_fingerprint": execution_fingerprint(a, self.repo),
                    "reason": "config.py change touches only new prompt versions",
                }
            ],
        }
        d = self.repository.commit(
            "D", {"configs/release/runtime_compatibility.json": json.dumps(declared)}
        )
        self.assertEqual(require_runtime_transition(a, d, self.repo).basis, "declared_compatible")
        _git(self.repo, "checkout", "-q", "-b", "side", a)
        side = self.repository.commit("side", {"README.md": "side branch\n"})
        _git(self.repo, "checkout", "-q", "main")
        with self.assertRaisesRegex(ReleaseError, "not an ancestor"):
            require_runtime_transition(side, d, self.repo)
        with self.assertRaisesRegex(ReleaseError, "repository"):
            require_runtime_transition("f" * 40, d, self.repo)
        with self.assertRaisesRegex(ReleaseError, "repository"):
            require_runtime_transition(d, "f" * 40, self.repo)
        for bad in ("short", "A" * 40, 7):
            with self.assertRaises(ReleaseError):
                require_runtime_transition(bad, d, self.repo)

    def test_a_malformed_compatibility_file_is_refused(self) -> None:
        a = self.repository.commits["A"]
        for document in (
            {"version": "runtime-compatibility-v0", "declarations": []},
            {"version": "runtime-compatibility-v1", "declarations": [{"execution_fingerprint": "x"}]},
            {
                "version": "runtime-compatibility-v1",
                "declarations": [{"execution_fingerprint": "0" * 64, "reason": " "}],
            },
        ):
            with self.subTest(document=document):
                commit = self.repository.commit(
                    f"bad-{len(self.repository.commits)}",
                    {
                        "configs/release/runtime_compatibility.json": json.dumps(document),
                        "src/llm_behavior_ci/tasks/selection.py": (
                            f"SELECTION = {100 + len(self.repository.commits)}\n"
                        ),
                    },
                )
                with self.assertRaises(ReleaseError):
                    require_runtime_transition(a, commit, self.repo)


class ReleaseRecordTests(unittest.TestCase):
    """A release summary is only trusted where its own store confirms it."""

    def setUp(self) -> None:
        self._tmpdir = TemporaryDirectory()
        self.addCleanup(self._tmpdir.cleanup)
        self.root = Path(self._tmpdir.name)

    def _release(self, scenario: str):
        root = self.root / scenario
        root.mkdir()
        lifecycle = demo.build_synthetic(scenario, root)
        with TestClient(create_app(lifecycle.dependencies)) as client:
            summary = run_three_tier_dev(
                lifecycle.gate,
                lifecycle.traffic,
                store_path=lifecycle.store_path,
                service=client,
            )
        return release_summary(summary, lifecycle.traffic), lifecycle

    def test_each_outcome_is_re_derived_from_its_store(self) -> None:
        for scenario, serving_role in (
            ("blocked", "production"),
            ("healthy", "candidate"),
            ("regression", "production"),
            ("production_regression", "production"),
        ):
            with self.subTest(scenario=scenario):
                record, lifecycle = self._release(scenario)
                expected = run_configuration_hash(getattr(lifecycle.traffic, serving_role))
                self.assertEqual(verify_release_record(record, lifecycle.store_path), expected)

    def test_a_tampered_or_mismatched_summary_is_refused(self) -> None:
        record, lifecycle = self._release("healthy")
        other, other_lifecycle = self._release("regression")
        production = run_configuration_hash(lifecycle.traffic.production)
        tampered = {
            "serving": {
                **record,
                "configurations": {**record["configurations"], "serving": production},
            },
            "status": {**record, "status": "rolled_back"},
            "decisions": {
                **record,
                "deployment": {**record["deployment"], "decisions": ["admit", "rollback"]},
            },
            "gate outcome": {**record, "gate": {**record["gate"], "outcome": "BLOCK"}},
            "gate artifact": {**record, "gate": {**record["gate"], "artifact_id": "0" * 64}},
            "version": {**record, "version": "release-summary-v9"},
        }
        for name, document in tampered.items():
            with self.subTest(case=name):
                with self.assertRaises(ReleaseLineageError):
                    verify_release_record(document, lifecycle.store_path)
        with self.assertRaises(ReleaseLineageError):
            verify_release_record(record, other_lifecycle.store_path)
        with self.assertRaises(ReleaseLineageError):
            verify_release_record(record, self.root / "absent.sqlite")
        verify_release_record(other, other_lifecycle.store_path)

    def test_a_rollback_after_the_summary_continues_from_the_known_good(self) -> None:
        record, lifecycle = self._release("healthy")
        production = lifecycle.traffic.production
        candidate_hash = run_configuration_hash(lifecycle.traffic.candidate)
        production_hash = run_configuration_hash(production)
        store = EpisodeStore(lifecycle.store_path)
        try:
            store.append_deployment_decision(
                DeploymentDecisionRecord(
                    configuration_hash=candidate_hash,
                    reference_configuration_hash=production_hash,
                    signal="deployment",
                    slice_name="production",
                    decision="production_rollback",
                    method="manual_rollback",
                    estimate=0.0,
                    boundary=None,
                    sample_size=0,
                    decided_at=_NOW,
                    evidence_source="operator",
                )
            )
        finally:
            store.close()
        serving = verify_release_record(record, lifecycle.store_path)
        self.assertEqual(serving, production_hash)
        lineage = require_release_lineage(record, production, serving=serving)
        self.assertTrue(lineage["rolled_back_after_release"])
        with self.assertRaises(ReleaseLineageError):
            require_release_lineage(record, lifecycle.traffic.candidate, serving=serving)
        regression, regression_lifecycle = self._release("regression")
        with self.assertRaises(ReleaseLineageError):
            require_release_lineage(
                regression,
                regression_lifecycle.traffic.production,
                serving=run_configuration_hash(regression_lifecycle.traffic.candidate),
            )

    def test_v1_and_v2_summaries_still_continue_a_lineage(self) -> None:
        record, lifecycle = self._release("healthy")
        candidate = lifecycle.traffic.candidate
        v2 = {
            key: value for key, value in record.items() if key not in {"commits", "runtime"}
        }
        v2.update(version="release-summary-v2", git_commit=lifecycle.traffic.production.git_commit)
        v1 = dict(v2)
        v1["version"] = "release-summary-v1"
        v1["configurations"] = {
            key: value
            for key, value in v2["configurations"].items()
            if key != "rolled_back"
        }
        v1["deployment"] = {
            key: value for key, value in v2["deployment"].items() if key != "production_rollback"
        }
        for document in (v1, v2):
            with self.subTest(version=document["version"]):
                serving = verify_release_record(document, lifecycle.store_path)
                lineage = require_release_lineage(document, candidate, serving=serving)
                self.assertEqual(lineage["previous_version"], document["version"])
                self.assertEqual(lineage["previous_serving_commit"], candidate.git_commit)
                with self.assertRaises(ReleaseLineageError):
                    require_release_lineage(
                        document, replace(candidate, git_commit="f" * 40), serving=serving
                    )

    def test_cross_commit_provenance_needs_a_verified_lineage(self) -> None:
        record, lifecycle = self._release("healthy")
        carried = lifecycle.traffic.candidate
        carried_gate = lifecycle.gate.candidate
        new_commit = "b" * 40
        candidate = _with_prompt(replace(carried, git_commit=new_commit), "prompt-v3")
        gate_candidate = _with_prompt(replace(carried_gate, git_commit=new_commit), "prompt-v3")
        state = RepositoryState(head=new_commit, dirty_paths=())
        require_release_lineage(record, carried)
        require_release_provenance(
            release_configurations=(gate_candidate, candidate),
            gate_reference=carried_gate,
            production=carried,
            state=state,
            carried=True,
        )
        for name, kwargs in {
            "no lineage": dict(carried=False),
            "candidate at an old commit": dict(
                release_configurations=(gate_candidate, carried), carried=True
            ),
            "gate reference at another commit": dict(
                gate_reference=replace(carried_gate, git_commit=new_commit), carried=True
            ),
            "dirty tree": dict(
                state=RepositoryState(head=new_commit, dirty_paths=("src/x.py",)), carried=True
            ),
        }.items():
            with self.subTest(case=name):
                arguments = dict(
                    release_configurations=(gate_candidate, candidate),
                    gate_reference=carried_gate,
                    production=carried,
                    state=state,
                )
                arguments.update(kwargs)
                with self.assertRaises(ReleaseError):
                    require_release_provenance(**arguments)
        rebuilt = replace(carried, git_commit=new_commit)
        with self.assertRaises(ReleaseLineageError):
            require_release_lineage(record, rebuilt)


class ProvisioningTests(unittest.TestCase):
    """Provision consecutive releases from a git repository and run them end to end."""

    def setUp(self) -> None:
        self._tmpdir = TemporaryDirectory()
        self.addCleanup(self._tmpdir.cleanup)
        self.repository = ReleaseRepository(Path(self._tmpdir.name))
        self.releases = self.repository.releases

    def test_a_first_release_pins_every_file_and_service_setting(self) -> None:
        manifest = self.repository.provision("r1", "A")
        release = self.releases / "r1"
        a = self.repository.commits["A"]
        self.assertEqual(manifest["release_commit"], a)
        self.assertIsNone(manifest["previous"])
        self.assertEqual(manifest["runtime"]["basis"], "same_commit")
        self.assertEqual(
            {entry["git_commit"] for entry in manifest["configurations"].values()}, {a}
        )
        self.assertEqual(verify_release_manifest(release, head=a)["release_id"], "r1")
        self.assertEqual((release / "code" / "RELEASE_COMMIT").read_text().strip(), a)
        self.assertTrue((release / "code" / "configs" / "models" / "candidate.json").is_file())
        self.assertFalse(any(path.name.startswith(".") for path in self.releases.iterdir()))
        self.assertNotIn("dev-canary-0", (release / MANIFEST_FILE).read_text())
        self.assertFalse(os.access(release / "production.json", os.W_OK))
        self.assertTrue(os.access(release / STORE_FILE, os.W_OK))
        monitor = json.loads((release / "monitor_settings.json").read_text())
        frozen = json.loads((release / "frozen_reference.json").read_text())
        production_hash = manifest["configurations"]["production"]["hash"]
        self.assertEqual(monitor["reference_configuration_hash"], production_hash)
        self.assertEqual(frozen["configuration_hash"], production_hash)
        argv = service_argv(release, manifest)
        self.assertEqual(argv[argv.index("--post-promotion-rollback") + 1], "record_only")
        self.assertEqual(argv[argv.index("--max-model-calls") + 1], "100")
        dependencies = release_dependencies(release, scenario="healthy")
        self.assertEqual(
            run_configuration_hash(dependencies.registry.production), production_hash
        )
        self.assertEqual(
            run_configuration_hash(dependencies.registry.candidate),
            manifest["configurations"]["candidate"]["hash"],
        )
        self.assertEqual(dependencies.post_promotion_rollback, "record_only")
        self.assertEqual(dependencies.model_call_budget.limit, 100)
        with self.assertRaises(ValueError):
            verify_release_manifest(release, head="b" * 40)

    def test_releases_are_never_overwritten_and_failures_leave_nothing(self) -> None:
        self.repository.provision("r1", "A")
        before = (self.releases / "r1" / MANIFEST_FILE).read_bytes()
        with self.assertRaisesRegex(ReleaseError, "never overwritten"):
            self.repository.provision("r1", "A")
        self.assertEqual((self.releases / "r1" / MANIFEST_FILE).read_bytes(), before)
        with self.assertRaises(ReleaseError):
            self.repository.provision(
                "r2",
                "A",
                budget={**self.repository.spec()["budget"], "max_episodes": 3},
            )
        self.assertFalse((self.releases / "r2").exists())
        self.assertEqual(
            sorted(path.name for path in self.releases.iterdir()), ["r1"]
        )

    def test_a_changed_release_file_is_refused(self) -> None:
        self.repository.provision("r1", "A")
        target = self.releases / "r1" / "canary_settings.json"
        target.chmod(0o644)
        document = json.loads(target.read_text())
        document["harm_margin"] = 0.5
        target.write_text(json.dumps(document))
        with self.assertRaisesRegex(ReleaseError, "changed after provisioning"):
            verify_release_manifest(self.releases / "r1")

    def test_consecutive_releases_carry_production_across_commits(self) -> None:
        self.repository.provision("r1", "A")
        record1, deployment1 = run_release(self.releases / "r1", scenario="none")
        self.assertEqual(record1["status"], "promoted")
        self.assertEqual(read_latest(self.releases), "r1")
        r1_candidate = record1["configurations"]["candidate"]
        a = self.repository.commits["A"]

        self.repository.commit("B", {"README.md": "docs only\n"})
        b = self.repository.commits["B"]
        with self.assertRaisesRegex(ReleaseError, "must be running"):
            self.repository.provision(
                "r2", "B", candidate={"template": "configs/models/candidate_v3.json", "fault": None}
            )
        manifest2 = self.repository.provision(
            "r2",
            "B",
            deployment=deployment1,
            candidate={"template": "configs/models/candidate_v3.json", "fault": None},
        )
        configurations = manifest2["configurations"]
        self.assertEqual(configurations["production"]["hash"], r1_candidate)
        self.assertEqual(configurations["production"]["git_commit"], a)
        self.assertEqual(configurations["gate_reference"]["git_commit"], a)
        self.assertEqual(configurations["candidate"]["git_commit"], b)
        self.assertEqual(manifest2["runtime"]["basis"], "identical_execution")
        self.assertEqual(manifest2["previous"]["release_id"], "r1")
        self.assertEqual(manifest2["previous"]["lineage"]["previous_serving_commit"], a)
        self.assertEqual(
            (self.releases / "r2" / "production.json").read_bytes(),
            (self.releases / "r1" / "candidate.json").read_bytes(),
        )
        require_current_lineage(self.releases, manifest2)
        self.assertEqual(
            require_previous_summary(self.releases, manifest2),
            self.releases / "r1" / SUMMARY_FILE,
        )

        record2, deployment2 = run_release(self.releases / "r2", scenario="regression")
        self.assertEqual(record2["version"], "release-summary-v3")
        self.assertEqual(record2["status"], "rolled_back")
        self.assertEqual(record2["git_commit"], b)
        self.assertEqual(record2["commits"], {"production": a, "candidate": b, "serving": a})
        self.assertEqual(record2["runtime"]["basis"], "identical_execution")
        self.assertEqual(record2["configurations"]["serving"], r1_candidate)
        self.assertEqual(record2["lineage"]["previous_serving_configuration_hash"], r1_candidate)

        self.repository.commit("C", {"src/llm_behavior_ci/runtime/agent.py": "AGENT = 2\n"})
        with self.assertRaisesRegex(ReleaseError, "execution code changed"):
            self.repository.provision("r3", "C", deployment=deployment2)
        self.assertFalse((self.releases / "r3").exists())
        declaration = {
            "version": "runtime-compatibility-v1",
            "declarations": [
                {
                    "execution_fingerprint": execution_fingerprint(a, self.repository.repo),
                    "reason": "agent change does not alter prompt-v2 execution",
                }
            ],
        }
        self.repository.commit(
            "D", {"configs/release/runtime_compatibility.json": json.dumps(declaration)}
        )
        manifest3 = self.repository.provision("r3", "D", deployment=deployment2)
        self.assertEqual(manifest3["runtime"]["basis"], "declared_compatible")
        self.assertEqual(manifest3["configurations"]["production"]["git_commit"], a)

    def test_the_deployed_service_and_the_lineage_must_agree(self) -> None:
        self.repository.provision("r1", "A")
        record1, deployment1 = run_release(self.releases / "r1", scenario="none")
        candidate_v3 = {"template": "configs/models/candidate_v3.json", "fault": None}
        for name, deployment in {
            "serves another configuration": {
                **deployment1,
                "serving_configuration_hash": record1["configurations"]["production"],
            },
            "canary in flight": {**deployment1, "state": "CANARY_ACTIVE"},
            "gate passed, canary not started": {**deployment1, "state": "GATE_PASSED"},
            "shutting down": {**deployment1, "admission": "shutting_down"},
        }.items():
            with self.subTest(case=name):
                with self.assertRaises(ReleaseError):
                    self.repository.provision(
                        "r2", "A", deployment=deployment, candidate=candidate_v3
                    )
        summary = self.releases / "r1" / SUMMARY_FILE
        original = summary.read_text()
        tampered = json.loads(original)
        tampered["configurations"]["serving"] = record1["configurations"]["production"]
        summary.write_text(json.dumps(tampered))
        with self.assertRaises(ReleaseLineageError):
            self.repository.provision("r2", "A", deployment=deployment1, candidate=candidate_v3)
        summary.write_text(original)
        manifest2 = self.repository.provision(
            "r2", "A", deployment=deployment1, candidate=candidate_v3
        )
        summary.write_text(original + " ")
        with self.assertRaisesRegex(ReleaseLineageError, "changed after provisioning"):
            require_previous_summary(self.releases, manifest2)
        summary.write_text(original)
        (self.releases / "LATEST").write_text("r2\n")
        with self.assertRaisesRegex(ReleaseError, "provision it again"):
            require_current_lineage(self.releases, manifest2)

    def test_a_first_release_refuses_an_unrecorded_deployment(self) -> None:
        with self.assertRaisesRegex(ReleaseError, "unrecorded deployment"):
            self.repository.provision("r1", "A", deployment={"production_configuration_hash": "0"})
        with self.assertRaisesRegex(ReleaseError, "dirty"):
            provision_release(
                load_release_spec(self.repository.write_spec()),
                release_root=self.releases,
                release_id="r1",
                state=RepositoryState(
                    head=self.repository.commits["A"], dirty_paths=("src/x.py",)
                ),
                deployment=None,
                now=_NOW,
                root=self.repository.repo,
            )

    def test_a_blocked_gate_admits_nothing_and_production_continues(self) -> None:
        self.repository.provision("r1", "A")
        record, deployment = run_release(self.releases / "r1", scenario="none", blocked=True)
        self.assertEqual(record["status"], "blocked")
        self.assertEqual(record["deployment"]["decisions"], [])
        self.assertNotIn("state", deployment)
        self.assertEqual(
            deployment["production_configuration_hash"], record["configurations"]["production"]
        )
        manifest2 = self.repository.provision(
            "r2",
            "A",
            deployment=deployment,
            candidate={"template": "configs/models/candidate_v3.json", "fault": None},
        )
        self.assertEqual(
            manifest2["configurations"]["production"]["hash"],
            record["configurations"]["production"],
        )

    def test_a_post_promotion_rollback_is_served_after_restart(self) -> None:
        self.repository.provision(
            "r1",
            "A",
            service={
                **self.repository.spec()["service"],
                "post_promotion_rollback": "rollback_on_alert",
            },
        )
        release = self.releases / "r1"
        record, deployment = run_release(release, scenario="production_regression")
        self.assertEqual(record["status"], "production_rolled_back")
        self.assertEqual(demo.LIVE_EXIT_CODES[record["status"]], 5)
        with TestClient(create_app(release_dependencies(release, scenario="none"))) as client:
            restarted = client.get("/deployment").json()
        self.assertEqual(restarted["state"], "PRODUCTION_ROLLED_BACK")
        self.assertEqual(
            restarted["serving_configuration_hash"], record["configurations"]["production"]
        )
        self.assertEqual(restarted["post_promotion_rollback"], "rollback_on_alert")

    def test_a_promotion_is_served_after_restart_and_the_budget_survives_it(self) -> None:
        self.repository.provision("r1", "A")
        release = self.releases / "r1"
        record, _ = run_release(release, scenario="none")
        self.assertEqual(record["status"], "promoted")
        spent = release_dependencies(release, scenario="none").model_call_budget.spent
        store = EpisodeStore(release / STORE_FILE)
        try:
            episodes = store.load_finished_episodes()
        finally:
            store.close()
        self.assertEqual(
            len(episodes), 2 * len(demo.CANARY_TASK_IDS) + len(demo.PRODUCTION_TASK_IDS)
        )
        self.assertEqual(spent, sum(len(episode.model_steps) for episode in episodes))
        turn_limit = release_traffic(release).candidate.agent.execute_turn_limit
        dependencies = release_dependencies(
            release, scenario="none", model_calls=spent + turn_limit
        )
        with TestClient(create_app(dependencies)) as client:
            deployment = client.get("/deployment").json()
            self.assertEqual(deployment["state"], "PROMOTED")
            self.assertEqual(
                deployment["serving_configuration_hash"], record["configurations"]["candidate"]
            )
            served = client.post(
                "/episodes", json={"task_id": "dev-steady-0", "mode": "execute"}
            )
            self.assertEqual(served.status_code, 200)
            self.assertEqual(
                served.json()["configuration_hash"], record["configurations"]["candidate"]
            )
            refused = client.post(
                "/episodes", json={"task_id": "dev-steady-1", "mode": "execute"}
            )
            self.assertEqual(refused.status_code, 503)
            self.assertIn("budget", refused.json()["detail"])
        store = EpisodeStore(release / STORE_FILE)
        try:
            persisted = store.count_model_steps()
        finally:
            store.close()
        self.assertGreater(persisted, spent)
        self.assertEqual(dependencies.model_call_budget.spent, persisted)


class ReleaseSpecTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = TemporaryDirectory()
        self.addCleanup(self._tmpdir.cleanup)
        self.repository = ReleaseRepository(Path(self._tmpdir.name))

    def test_malformed_specs_are_refused(self) -> None:
        spec = self.repository.spec()
        base = self.repository.local
        cases = {
            "unknown key": {**spec, "extra": 1},
            "missing key": {key: value for key, value in spec.items() if key != "budget"},
            "version": {**spec, "version": "release-spec-v0"},
            "template outside configs": {
                **spec,
                "candidate": {"template": "../secrets.json", "fault": None},
            },
            "absolute template": {
                **spec,
                "candidate": {"template": "/etc/passwd", "fault": None},
            },
            "public host": {**spec, "service": {**spec["service"], "host": "0.0.0.0"}},
            "unknown rollback policy": {
                **spec,
                "service": {**spec["service"], "post_promotion_rollback": "always"},
            },
            "monitor binds its own reference": {
                **spec,
                "monitor_settings": {**spec["monitor_settings"], "reference_configuration_hash": "0"},
            },
            "zero budget": {**spec, "budget": {**spec["budget"], "max_gate_model_calls": 0}},
            "missing plan evidence": {**spec, "plan_evidence": "absent.json"},
        }
        for name, document in cases.items():
            with self.subTest(case=name):
                with self.assertRaises(ReleaseError):
                    release_spec_from_dict(document, base=base)
        parsed = release_spec_from_dict(spec, base=base)
        self.assertEqual(parsed.service["post_promotion_rollback"], "record_only")

    def test_rollback_on_alert_is_only_an_explicit_release_setting(self) -> None:
        spec = self.repository.spec()
        service = {key: value for key, value in spec["service"].items()}
        del service["post_promotion_rollback"]
        with self.assertRaises(ReleaseError):
            release_spec_from_dict({**spec, "service": service}, base=self.repository.local)
        self.assertEqual(
            serve.build_parser().parse_args(
                [
                    "--production-config", "p", "--store", "s", "--max-in-flight", "1",
                    "--shutdown-timeout-seconds", "1", "--canary-settings", "c",
                    "--monitor-settings", "m", "--frozen-reference", "f",
                    "--dedup-seconds", "0", "--canary-assignment-seed", "0",
                ]
            ).post_promotion_rollback,
            "record_only",
        )

    def test_a_train_manifest_cannot_stand_in_for_dev(self) -> None:
        with self.assertRaisesRegex(ReleaseError, "dev manifest"):
            self.repository.provision("r1", "A", dev_manifest="train.json")


class ReleaseCommandTests(unittest.TestCase):
    """``run_three_tier_dev.py release`` and ``release_ctl.py`` on a provisioned release."""

    def setUp(self) -> None:
        self._tmpdir = TemporaryDirectory()
        self.addCleanup(self._tmpdir.cleanup)
        self.repository = ReleaseRepository(Path(self._tmpdir.name))
        self.repository.provision("r1", "A")
        self.release = self.repository.releases / "r1"
        self.state = RepositoryState(head=self.repository.commits["A"], dirty_paths=())

    def _release(self, **patches):
        stdout, stderr = io.StringIO(), io.StringIO()
        transition = require_runtime_transition(
            self.state.head, self.state.head, self.repository.repo
        )
        defaults = {"require_runtime_transition": lambda *args, **kwargs: transition}
        defaults.update(patches)
        with redirect_stdout(stdout), redirect_stderr(stderr):
            with patch.multiple(demo, **defaults):
                code = demo._main_release(self.release, state=self.state)
        return code, stdout.getvalue(), stderr.getvalue()

    def test_a_release_runs_from_its_manifest_and_becomes_latest(self) -> None:
        captured = {}

        def run_live(gate, traffic, *, store_path, service_url):
            captured.update(
                gate=gate, traffic=traffic, store_path=store_path, service_url=service_url
            )
            return {"status": "promoted"}

        code, output, _ = self._release(
            run_live=run_live, release_summary=lambda *args, **kwargs: {"status": "promoted"}
        )
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(output)["status"], "promoted")
        self.assertEqual(captured["service_url"], "http://127.0.0.1:8765")
        self.assertEqual(captured["store_path"], self.release / STORE_FILE)
        self.assertIsInstance(captured["gate"].runtime_factory, BudgetedRuntimeFactory)
        self.assertEqual(captured["gate"].runtime_factory.budget.limit, 100)
        self.assertEqual(len(captured["traffic"].canary_task_ids), len(demo.CANARY_TASK_IDS))
        self.assertEqual(read_latest(self.repository.releases), "r1")

    def test_failed_and_refused_releases_do_not_become_latest(self) -> None:
        code, output, _ = self._release(run_live=lambda *args, **kwargs: (_ for _ in ()).throw(ConnectionError("refused")))
        self.assertEqual(code, 4)
        self.assertEqual(output, "")
        self.assertIsNone(read_latest(self.repository.releases))
        other = RepositoryState(head="b" * 40, dirty_paths=())
        with patch.object(demo, "run_live") as run:
            with redirect_stderr(io.StringIO()):
                self.assertEqual(demo._main_release(self.release, state=other), 2)
            run.assert_not_called()
        self.assertIsNone(read_latest(self.repository.releases))

    def test_a_run_beyond_its_episode_budget_is_refused_before_any_model_call(self) -> None:
        args = demo.release_namespace(
            self.release, verify_release_manifest(self.release), None
        )
        gate, traffic = demo.live_inputs(args)
        with self.assertRaisesRegex(demo.ConfigError, "max-episodes"):
            demo.apply_budgets(
                demo.argparse.Namespace(max_episodes=3, max_model_calls=None), gate, traffic
            )

    def test_release_ctl_refuses_to_serve_from_outside_the_release_code(self) -> None:
        with redirect_stderr(io.StringIO()) as stderr:
            code = release_ctl.main(["serve", "--release-dir", str(self.release)])
        self.assertEqual(code, 2)
        self.assertIn("release's own code", stderr.getvalue())

    def test_release_ctl_reports_an_unreachable_service(self) -> None:
        def unreachable(url: str):
            raise release_ctl.ServiceUnreachable(f"no service answers at {url}")

        with patch.object(release_ctl, "read_repository_state", return_value=self.state):
            with redirect_stdout(io.StringIO()) as stdout:
                code = release_ctl.main(
                    ["check-deployed", "--release-dir", str(self.release)], fetch=unreachable
                )
        self.assertEqual(code, 0)
        self.assertIsNone(json.loads(stdout.getvalue())["deployed"])
        with patch.object(release_ctl, "read_repository_state", return_value=self.state):
            with redirect_stderr(io.StringIO()):
                code = release_ctl.main(
                    ["check-deployed", "--release-dir", str(self.release)],
                    fetch=lambda url: {"production_configuration_hash": "0"},
                )
        self.assertEqual(code, 2)
        with redirect_stderr(io.StringIO()), redirect_stdout(io.StringIO()):
            code = release_ctl.main(
                ["wait-registered", "--release-dir", str(self.release), "--timeout-seconds", "0"],
                fetch=unreachable,
            )
        self.assertEqual(code, 4)

    def test_release_ctl_waits_for_the_release_to_register(self) -> None:
        manifest = verify_release_manifest(self.release)
        registered = {
            "production_configuration_hash": manifest["configurations"]["production"]["hash"],
            "candidate_configuration_hash": manifest["configurations"]["candidate"]["hash"],
        }
        with redirect_stdout(io.StringIO()):
            self.assertEqual(
                release_ctl.main(
                    ["wait-registered", "--release-dir", str(self.release)],
                    fetch=lambda url: registered,
                ),
                0,
            )
        with redirect_stderr(io.StringIO()):
            self.assertEqual(
                release_ctl.main(
                    ["wait-registered", "--release-dir", str(self.release)],
                    fetch=lambda url: {**registered, "candidate_configuration_hash": "0"},
                ),
                4,
            )

    def test_release_ctl_bare_and_malformed_invocations_exit_2(self) -> None:
        with redirect_stderr(io.StringIO()):
            self.assertEqual(release_ctl.main([]), 2)
            self.assertEqual(release_ctl.main(["provision"]), 2)
            self.assertEqual(
                release_ctl.main(
                    [
                        "provision",
                        "--release-root",
                        str(self.repository.releases),
                        "--release-id",
                        "../escape",
                        "--spec",
                        "spec.json",
                    ]
                ),
                2,
            )


if __name__ == "__main__":
    unittest.main()
