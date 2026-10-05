import hashlib
import importlib.util
import io
import json
import os
import re
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from llm_behavior_ci.tasks.selection import (
    canonical_task_set_bytes,
    task_set_hash_from_bytes,
)
from llm_behavior_ci.config import TaskConfiguration
from llm_behavior_ci.tasks.short_horizon import (
    DIAGNOSTIC_RULE,
    DIAGNOSTIC_TASK_COUNT,
    MAX_REFERENCE_AGENT_CALLS,
    MAX_PAGED_CALLS,
    MAX_TASKS_PER_SCENARIO,
    SELECTION_RULE,
    SELECTION_SEED,
    TASK_COUNT,
    ShortHorizonCandidate,
    ShortHorizonSelectionError,
    classify_reference_call,
    reference_shape,
    select_spotify_short_diagnostic,
    select_spotify_short_set,
    selection_audit,
)

_ROOT = Path(__file__).resolve().parents[2]
_BUILDER = _ROOT / "scripts/data/build_spotify_capability_short_set.py"
_SELECTOR = _ROOT / "src/llm_behavior_ci/tasks/short_horizon.py"
_APPWORLD_TASK_ID = re.compile(r"\b[0-9a-f]{7}_[0-9]+\b")
_FORBIDDEN = (
    "results/",
    ".sqlite",
    "evaluator_success",
    "prompt_token",
    "model_steps",
    "top_k_logprobs",
    '["requirement"]',
    "['requirement']",
    'load_task_ids("dev")',
    'load_task_ids("test_normal")',
    'load_task_ids("test_challenge")',
)


def _builder():
    spec = importlib.util.spec_from_file_location(
        "build_spotify_capability_short_set",
        _BUILDER,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _candidate(
    task_id: str,
    scenario_id: str,
    *,
    difficulty: int = 1,
    requirement_count: int = 2,
    reference_paged_calls: int = 1,
    reference_agent_calls: int = 2,
) -> ShortHorizonCandidate:
    return ShortHorizonCandidate(
        task_id=task_id,
        scenario_id=scenario_id,
        difficulty=difficulty,
        requirement_count=requirement_count,
        reference_paged_calls=reference_paged_calls,
        reference_agent_calls=reference_agent_calls,
    )


def _fill_pool() -> tuple[ShortHorizonCandidate, ...]:
    candidates: list[ShortHorizonCandidate] = []
    for scenario_index in range(8):
        scenario_id = f"scenario-{scenario_index}"
        for task_index in range(3):
            candidates.append(
                _candidate(
                    f"task-{scenario_index}-{task_index}",
                    scenario_id,
                )
            )
    return tuple(candidates)


def _select(candidates: tuple[ShortHorizonCandidate, ...], seed: int = SELECTION_SEED):
    return select_spotify_short_set(
        candidates,
        appworld_version="0.1.3.post1",
        selection_seed=seed,
    )


class ShortHorizonSelectionTest(unittest.TestCase):
    def test_selects_exactly_twenty_tasks_from_tier_1(self) -> None:
        selection = _select(_fill_pool())
        self.assertEqual(selection.tier, "tier_1")
        self.assertEqual(selection.task_set.task_count, TASK_COUNT)
        self.assertEqual(len(selection.task_set.task_ids), TASK_COUNT)
        self.assertEqual(len(set(selection.task_set.task_ids)), TASK_COUNT)
        self.assertEqual(selection.task_set.selection_rule, SELECTION_RULE)
        self.assertEqual(selection.task_set.selection_seed, SELECTION_SEED)
        self.assertEqual(selection.task_set.split, "train")
        per_scenario: dict[str, int] = {}
        for candidate in selection.candidates:
            self.assertEqual(candidate.difficulty, 1)
            self.assertLessEqual(candidate.requirement_count, 3)
            self.assertLessEqual(candidate.reference_paged_calls, MAX_PAGED_CALLS)
            self.assertLessEqual(
                candidate.reference_agent_calls,
                MAX_REFERENCE_AGENT_CALLS,
            )
            per_scenario[candidate.scenario_id] = (
                per_scenario.get(candidate.scenario_id, 0) + 1
            )
        self.assertGreaterEqual(len(per_scenario), 7)
        self.assertTrue(all(count <= MAX_TASKS_PER_SCENARIO for count in per_scenario.values()))

    def test_repeat_selection_keeps_ids_and_hash(self) -> None:
        pool = _fill_pool()
        first = _select(pool)
        second = _select(pool)
        self.assertEqual(first.task_set.task_ids, second.task_set.task_ids)
        self.assertEqual(first.task_set.scenario_ids, second.task_set.scenario_ids)
        self.assertEqual(first.task_set.task_set_hash, second.task_set.task_set_hash)
        payload = canonical_task_set_bytes(
            appworld_version=first.task_set.appworld_version,
            split=first.task_set.split,
            selection_rule=first.task_set.selection_rule,
            selection_seed=first.task_set.selection_seed,
            tasks=tuple(zip(first.task_set.task_ids, first.task_set.scenario_ids)),
        )
        self.assertEqual(first.task_set.task_set_hash, task_set_hash_from_bytes(payload))

    def test_seed_changes_the_selected_order(self) -> None:
        pool = _fill_pool()
        self.assertNotEqual(
            _select(pool, seed=17).task_set.task_ids,
            _select(pool, seed=18).task_set.task_ids,
        )

    def test_paginated_tasks_stay_out_of_a_full_structural_pool(self) -> None:
        structural = _fill_pool()
        paginated = tuple(
            _candidate(
                f"paged-{index}",
                f"scenario-{index % 8}",
                reference_paged_calls=10,
                reference_agent_calls=30,
            )
            for index in range(8)
        )
        selection = _select(structural + paginated)
        selected = set(selection.task_set.task_ids)
        self.assertTrue(selected.isdisjoint(candidate.task_id for candidate in paginated))
        self.assertTrue(
            all(candidate.reference_paged_calls <= MAX_PAGED_CALLS for candidate in selection.candidates)
        )

    def test_scenario_cap_limits_a_single_scenario(self) -> None:
        candidates = [
            _candidate(f"fat-{index}", "scenario-fat") for index in range(10)
        ]
        for scenario_index in range(6):
            scenario_id = f"scenario-{scenario_index}"
            for task_index in range(3):
                candidates.append(
                    _candidate(f"task-{scenario_index}-{task_index}", scenario_id)
                )
        selection = _select(tuple(candidates))
        fat = [
            task_id
            for task_id, scenario_id in zip(
                selection.task_set.task_ids,
                selection.task_set.scenario_ids,
            )
            if scenario_id == "scenario-fat"
        ]
        self.assertLessEqual(len(fat), MAX_TASKS_PER_SCENARIO)
        self.assertEqual(selection.task_set.task_count, TASK_COUNT)

    def test_requirement_tier_does_not_admit_paginated_tasks(self) -> None:
        paginated = []
        for scenario_index in range(8):
            for task_index in range(3):
                paginated.append(
                    _candidate(
                        f"paged-{scenario_index}-{task_index}",
                        f"scenario-{scenario_index}",
                        requirement_count=2,
                        reference_paged_calls=10,
                        reference_agent_calls=40,
                    )
                )
        short = tuple(
            _candidate(f"short-{index}", "scenario-short") for index in range(3)
        )
        pool = tuple(paginated) + short
        audit = selection_audit(pool)
        self.assertEqual(audit["requirement_tiers_without_structural_filter"]["tier_1"]["task_count"], 27)
        self.assertEqual(audit["tiers"]["tier_1"]["task_count"], 3)
        with self.assertRaises(ShortHorizonSelectionError) as caught:
            _select(pool)
        self.assertEqual(caught.exception.audit["selected_tier"], None)
        self.assertEqual(caught.exception.audit["tiers"]["tier_1"]["task_count"], 3)
        rendered = json.dumps(caught.exception.audit)
        self.assertIsNone(_APPWORLD_TASK_ID.search(rendered))

    def test_later_tier_is_used_only_after_earlier_tiers_cannot_fill(self) -> None:
        early = tuple(_candidate(f"early-{index}", "scenario-early") for index in range(3))
        later = []
        for scenario_index in range(8):
            for task_index in range(3):
                later.append(
                    _candidate(
                        f"later-{scenario_index}-{task_index}",
                        f"scenario-{scenario_index}",
                        requirement_count=5,
                    )
                )
        selection = _select(early + tuple(later))
        self.assertEqual(selection.tier, "tier_3")
        self.assertTrue(any(candidate.requirement_count == 5 for candidate in selection.candidates))
        self.assertTrue(all(candidate.difficulty == 1 for candidate in selection.candidates))
        self.assertTrue(all(candidate.reference_paged_calls <= 1 for candidate in selection.candidates))

    def test_dev_split_is_rejected(self) -> None:
        with self.assertRaises(ShortHorizonSelectionError):
            select_spotify_short_set(
                _fill_pool(),
                appworld_version="0.1.3.post1",
                split="dev",
            )

    def test_candidate_fields_are_counts_only(self) -> None:
        names = [field.name for field in ShortHorizonCandidate.__dataclass_fields__.values()]
        self.assertEqual(
            names,
            [
                "task_id",
                "scenario_id",
                "difficulty",
                "requirement_count",
                "reference_paged_calls",
                "reference_agent_calls",
            ],
        )

    def test_reference_call_shape_ignores_auth_and_other_apps(self) -> None:
        self.assertIsNone(classify_reference_call("/supervisor/complete_task", False))
        self.assertEqual(classify_reference_call("/spotify/auth/token", True), (False, False))
        self.assertEqual(
            classify_reference_call("/spotify/library/songs?page_index=2", False),
            (True, True),
        )
        self.assertEqual(classify_reference_call("/spotify/songs/7", False), (True, False))
        agent_calls, paged_calls = reference_shape(
            (
                ("/spotify/auth/token", False),
                ("/spotify/library/songs", True),
                ("/spotify/songs/7", False),
                ("/supervisor/complete_task", False),
            )
        )
        self.assertEqual((agent_calls, paged_calls), (2, 1))


class ShortHorizonBuilderTest(unittest.TestCase):
    def test_bare_command_exits_2(self) -> None:
        self.assertEqual(_builder().main([]), 2)

    def test_refuses_a_tracked_manifest_path(self) -> None:
        builder = _builder()
        with redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as caught:
                builder.main(
                    ["--manifest", "configs/tasks/train_spotify_capability.json"],
                    loader=_fill_pool,
                )
        self.assertEqual(caught.exception.code, 2)

    def test_insufficient_pool_writes_nothing_and_prints_counts_only(self) -> None:
        builder = _builder()
        manifest = _ROOT / "data/processed/spotify_capability_short_unit_manifest.json"
        public = _ROOT / "data/processed/spotify_capability_short_unit_public.json"
        tracked = _ROOT / "configs/tasks/train_spotify_capability_short.json"
        manifest.unlink(missing_ok=True)
        public.unlink(missing_ok=True)
        stdout = io.StringIO()
        stderr = io.StringIO()
        try:
            with redirect_stdout(stdout), redirect_stderr(stderr):
                code = builder.main(
                    ["--manifest", str(manifest), "--public", str(public)],
                    loader=lambda: tuple(
                        _candidate(f"only-{index}", "scenario-only") for index in range(3)
                    ),
                )
            self.assertEqual(code, 3)
            self.assertFalse(manifest.exists())
            self.assertFalse(public.exists())
            self.assertFalse(tracked.exists())
            audit = json.loads(stdout.getvalue())
            self.assertEqual(audit["tiers"]["tier_1"]["task_count"], 3)
            self.assertEqual(audit["selected_tier"], None)
            self.assertIsNone(_APPWORLD_TASK_ID.search(stdout.getvalue()))
            self.assertNotIn("requirement_count_by_task", audit)
        finally:
            manifest.unlink(missing_ok=True)
            public.unlink(missing_ok=True)

    def test_rerunning_the_builder_rewrites_the_same_ids_and_hash(self) -> None:
        builder = _builder()
        manifest = _ROOT / "data/processed/spotify_capability_short_unit_manifest.json"
        public = _ROOT / "data/processed/spotify_capability_short_unit_public.json"
        manifest.unlink(missing_ok=True)
        public.unlink(missing_ok=True)
        try:
            first_out = io.StringIO()
            second_out = io.StringIO()
            with redirect_stdout(first_out):
                first = builder.main(
                    ["--manifest", str(manifest), "--public", str(public)],
                    loader=_fill_pool,
                )
            first_manifest = manifest.read_text(encoding="utf-8")
            first_public = public.read_text(encoding="utf-8")
            with redirect_stdout(second_out):
                second = builder.main(
                    ["--manifest", str(manifest), "--public", str(public)],
                    loader=_fill_pool,
                )
            self.assertEqual(first, 0)
            self.assertEqual(second, 0)
            self.assertEqual(manifest.read_text(encoding="utf-8"), first_manifest)
            self.assertEqual(public.read_text(encoding="utf-8"), first_public)
            public_payload = json.loads(first_public)
            local_payload = json.loads(first_manifest)
            self.assertEqual(
                list(public_payload),
                [
                    "appworld_version",
                    "split",
                    "selection_rule",
                    "selection_seed",
                    "task_count",
                    "scenario_count",
                    "task_set_hash",
                    "appworld_setup_profile",
                ],
            )
            self.assertEqual(public_payload["selection_rule"], SELECTION_RULE)
            self.assertEqual(public_payload["task_count"], 20)
            self.assertEqual(public_payload["appworld_setup_profile"], "spotify_authenticated_v1")
            self.assertNotIn("task_ids", public_payload)
            self.assertEqual(local_payload["task_ids"], json.loads(manifest.read_text())["task_ids"])
            self.assertEqual(len(local_payload["task_ids"]), 20)
            self.assertEqual(local_payload["selection_tier"], "tier_1")
            self.assertIsNone(_APPWORLD_TASK_ID.search(first_public))
            self.assertNotIn('["requirement"]', first_manifest)
        finally:
            manifest.unlink(missing_ok=True)
            public.unlink(missing_ok=True)

    def test_sources_do_not_consult_prior_results_or_requirement_text(self) -> None:
        for path in (_SELECTOR, _BUILDER):
            text = path.read_text(encoding="utf-8")
            for token in _FORBIDDEN:
                with self.subTest(path=path.name, token=token):
                    self.assertNotIn(token, text)
            self.assertIsNone(_APPWORLD_TASK_ID.search(text))


def _diagnostic_pool() -> tuple[ShortHorizonCandidate, ...]:
    easy = [
        _candidate(
            f"easy-{index}",
            "scenario-easy",
            difficulty=1,
            requirement_count=2,
            reference_paged_calls=1,
            reference_agent_calls=2,
        )
        for index in range(3)
    ]
    medium = [
        _candidate(
            f"medium-{index}",
            "scenario-medium",
            difficulty=2,
            requirement_count=5,
            reference_paged_calls=0,
            reference_agent_calls=8,
        )
        for index in range(3)
    ]
    rejected = (
        _candidate(
            "paged-long",
            "scenario-long",
            reference_paged_calls=10,
            reference_agent_calls=40,
        ),
        _candidate(
            "calls-long",
            "scenario-long",
            reference_paged_calls=0,
            reference_agent_calls=11,
        ),
    )
    return tuple(easy + medium + list(rejected))


def _diagnose(candidates: tuple[ShortHorizonCandidate, ...]):
    return select_spotify_short_diagnostic(
        candidates,
        appworld_version="0.1.3.post1",
    )


class ShortHorizonDiagnosticTest(unittest.TestCase):
    def test_selects_the_six_shape_limited_tasks_in_task_id_order(self) -> None:
        selection = _diagnose(_diagnostic_pool())
        self.assertEqual(selection.tier, "structural_short")
        self.assertEqual(selection.task_set.selection_rule, DIAGNOSTIC_RULE)
        self.assertEqual(selection.task_set.selection_seed, SELECTION_SEED)
        self.assertEqual(selection.task_set.task_count, DIAGNOSTIC_TASK_COUNT)
        self.assertEqual(selection.task_set.scenario_count, 2)
        self.assertEqual(
            list(selection.task_set.task_ids),
            ["easy-0", "easy-1", "easy-2", "medium-0", "medium-1", "medium-2"],
        )
        self.assertEqual(list(selection.task_set.task_ids), sorted(selection.task_set.task_ids))
        for candidate in selection.candidates:
            self.assertLessEqual(candidate.reference_paged_calls, MAX_PAGED_CALLS)
            self.assertLessEqual(candidate.reference_agent_calls, MAX_REFERENCE_AGENT_CALLS)
        self.assertNotIn("paged-long", selection.task_set.task_ids)
        self.assertNotIn("calls-long", selection.task_set.task_ids)

    def test_diagnostic_hash_is_stable_and_seed_is_locked(self) -> None:
        pool = _diagnostic_pool()
        first = _diagnose(pool)
        second = _diagnose(pool)
        self.assertEqual(first.task_set.task_ids, second.task_set.task_ids)
        self.assertEqual(first.task_set.task_set_hash, second.task_set.task_set_hash)
        payload = canonical_task_set_bytes(
            appworld_version="0.1.3.post1",
            split="train",
            selection_rule=DIAGNOSTIC_RULE,
            selection_seed=SELECTION_SEED,
            tasks=tuple(zip(first.task_set.task_ids, first.task_set.scenario_ids)),
        )
        self.assertEqual(first.task_set.task_set_hash, task_set_hash_from_bytes(payload))
        with self.assertRaises(ShortHorizonSelectionError):
            select_spotify_short_diagnostic(
                pool,
                appworld_version="0.1.3.post1",
                selection_seed=18,
            )

    def test_diagnostic_refuses_a_pool_that_is_not_six_tasks(self) -> None:
        short = _diagnostic_pool()[:5]
        with self.assertRaises(ShortHorizonSelectionError) as caught:
            _diagnose(short)
        self.assertEqual(caught.exception.audit["diagnostic_task_count"], 5)
        self.assertIsNone(_APPWORLD_TASK_ID.search(json.dumps(caught.exception.audit)))

    def test_diagnostic_builder_rewrites_the_same_manifest(self) -> None:
        builder = _builder()
        manifest = _ROOT / "data/processed/spotify_short_horizon_diagnostic_unit.json"
        public = _ROOT / "data/processed/spotify_short_horizon_diagnostic_unit_public.json"
        manifest.unlink(missing_ok=True)
        public.unlink(missing_ok=True)
        try:
            first_out = io.StringIO()
            second_out = io.StringIO()
            with redirect_stdout(first_out):
                first = builder.main(
                    [
                        "--diagnostic",
                        "--manifest",
                        str(manifest),
                        "--public",
                        str(public),
                    ],
                    loader=_diagnostic_pool,
                )
            first_manifest = manifest.read_text(encoding="utf-8")
            first_public = public.read_text(encoding="utf-8")
            with redirect_stdout(second_out):
                second = builder.main(
                    [
                        "--diagnostic",
                        "--manifest",
                        str(manifest),
                        "--public",
                        str(public),
                    ],
                    loader=_diagnostic_pool,
                )
            self.assertEqual(first, 0)
            self.assertEqual(second, 0)
            self.assertEqual(manifest.read_text(encoding="utf-8"), first_manifest)
            self.assertEqual(public.read_text(encoding="utf-8"), first_public)
            public_payload = json.loads(first_public)
            local_payload = json.loads(first_manifest)
            self.assertEqual(public_payload["task_count"], 6)
            self.assertEqual(public_payload["selection_rule"], DIAGNOSTIC_RULE)
            self.assertEqual(public_payload["scenario_count"], 2)
            self.assertNotIn("task_ids", public_payload)
            self.assertEqual(local_payload["selection_tier"], "structural_short")
            self.assertEqual(local_payload["task_ids"], sorted(local_payload["task_ids"]))
            self.assertEqual(len(local_payload["reference_paged_calls_by_task"]), 6)
            self.assertTrue(
                all(count <= 1 for count in local_payload["reference_paged_calls_by_task"].values())
            )
            self.assertTrue(
                all(
                    count <= 10
                    for count in local_payload["reference_agent_calls_by_task"].values()
                )
            )
            self.assertIsNone(_APPWORLD_TASK_ID.search(first_public))
        finally:
            manifest.unlink(missing_ok=True)
            public.unlink(missing_ok=True)

    def test_pilot_keeps_twenty_tasks_for_the_stress_set(self) -> None:
        pilot = _capability_pilot()
        diagnostic = TaskConfiguration(
            appworld_version="0.1.3.post1",
            split="train",
            selection_rule=DIAGNOSTIC_RULE,
            selection_seed=17,
            task_count=6,
            task_set_hash="a" * 64,
        )
        stress = TaskConfiguration(
            appworld_version="0.1.3.post1",
            split="train",
            selection_rule="fixed_spotify_capability",
            selection_seed=17,
            task_count=20,
            task_set_hash="b" * 64,
        )
        self.assertEqual(pilot._required_task_count(diagnostic), 6)
        self.assertEqual(pilot._required_task_count(stress), 20)

    def test_diagnostic_config_preserves_the_locked_32b_stack(self) -> None:
        payload = json.loads(
            (
                _ROOT / "configs/models/qwen3_32b_awq_spotify_short_horizon_diagnostic.json"
            ).read_text(encoding="utf-8")
        )
        serving = payload["model"]["serving"]
        agent = payload["agent"]
        self.assertEqual(payload["model"]["model"]["repository"], "Qwen/Qwen3-32B-AWQ")
        self.assertEqual(
            payload["model"]["model"]["revision"],
            "0499c3ac83fdef8810b907a23894ba91e95eddd8",
        )
        self.assertEqual(serving["max_model_len"], 32768)
        self.assertEqual(serving["gpu_memory_utilization"], 0.97)
        self.assertEqual(serving["cpu_offload_gb"], 5.0)
        self.assertEqual(serving["kv_cache_dtype"], "float16")
        self.assertEqual(serving["dtype"], "float16")
        self.assertEqual(serving["max_num_seqs"], 1)
        self.assertEqual(serving["max_num_batched_tokens"], 4096)
        self.assertFalse(serving["enable_prefix_caching"])
        self.assertTrue(serving["enable_chunked_prefill"])
        self.assertTrue(serving["enforce_eager"])
        self.assertEqual(serving["tensor_parallel_size"], 1)
        self.assertEqual(agent["prompt"]["prompt_version"], "prompt-runtime-auth-v2")
        self.assertEqual(agent["workflow"]["policy"], "plan_progress_v2")
        self.assertEqual(agent["execute_max_model_turns"], 20)
        self.assertEqual(agent["tool_access_profile"], "spotify_capability_v1")
        self.assertEqual(agent["sampling"]["temperature"], 0.0)
        self.assertEqual(agent["sampling"]["seed"], 17)
        self.assertEqual(agent["sampling"]["execute_max_tokens"], 192)


def _capability_pilot():
    path = _ROOT / "scripts/evaluation/run_capability_pilot.py"
    spec = importlib.util.spec_from_file_location("diagnostic_capability_pilot", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _train_data_ready() -> bool:
    root = _ROOT / "data/raw/appworld"
    if not (root / "data/datasets/train.txt").is_file():
        return False
    try:
        import appworld  # noqa: F401
    except ImportError:
        return False
    return True


@unittest.skipUnless(_train_data_ready(), "AppWorld train data is not installed")
class InstalledTrainShortHorizonTest(unittest.TestCase):
    def test_installed_pool_cannot_fill_and_the_audit_is_stable(self) -> None:
        os.environ["APPWORLD_ROOT"] = str(_ROOT / "data/raw/appworld")
        builder = _builder()
        first = builder.load_train_spotify_candidates()
        second = builder.load_train_spotify_candidates()

        def _digest(candidates: tuple[ShortHorizonCandidate, ...]) -> str:
            payload = "\n".join(sorted(item.task_id for item in candidates)).encode()
            return hashlib.sha256(payload).hexdigest()

        self.assertEqual(_digest(first), _digest(second))
        audit = selection_audit(first)
        self.assertEqual(selection_audit(second), audit)
        self.assertEqual(audit["easy"]["task_count"], 24)
        self.assertEqual(audit["easy"]["scenario_count"], 8)
        self.assertEqual(audit["easy"]["paged_calls"], {"1": 3, "10": 18, "30": 3})
        self.assertEqual(
            audit["requirement_tiers_without_structural_filter"]["tier_1"]["task_count"],
            15,
        )
        self.assertEqual(
            audit["requirement_tiers_without_structural_filter"]["tier_3"]["task_count"],
            24,
        )
        self.assertEqual(audit["tiers"]["tier_1"]["task_count"], 3)
        self.assertEqual(audit["tiers"]["tier_1"]["scenario_count"], 1)
        self.assertEqual(audit["tiers"]["tier_3"]["task_count"], 3)
        self.assertEqual(audit["tiers"]["tier_4"]["task_count"], 0)
        self.assertEqual(audit["structural_short"]["task_count"], 6)
        self.assertEqual(audit["structural_short"]["scenario_count"], 2)
        self.assertEqual(audit["structural_short"]["difficulty"], {"1": 3, "2": 3})
        rendered = json.dumps(audit)
        self.assertIsNone(_APPWORLD_TASK_ID.search(rendered))
        with self.assertRaises(ShortHorizonSelectionError):
            select_spotify_short_set(first, appworld_version="0.1.3.post1")
        diagnostic = select_spotify_short_diagnostic(
            first,
            appworld_version="0.1.3.post1",
        )
        self.assertEqual(diagnostic.task_set.task_count, 6)
        self.assertEqual(diagnostic.task_set.scenario_count, 2)
        self.assertEqual(diagnostic.task_set.selection_rule, DIAGNOSTIC_RULE)
        self.assertEqual(
            list(diagnostic.task_set.task_ids),
            sorted(diagnostic.task_set.task_ids),
        )
        difficulties: dict[int, int] = {}
        for candidate in diagnostic.candidates:
            self.assertLessEqual(candidate.reference_paged_calls, 1)
            self.assertLessEqual(candidate.reference_agent_calls, 10)
            difficulties[candidate.difficulty] = difficulties.get(candidate.difficulty, 0) + 1
        self.assertEqual(difficulties, {1: 3, 2: 3})
        rejected = {
            candidate.task_id
            for candidate in first
            if candidate.reference_paged_calls > 1 or candidate.reference_agent_calls > 10
        }
        self.assertTrue(rejected.isdisjoint(diagnostic.task_set.task_ids))
        public_path = _ROOT / "configs/tasks/train_spotify_short_horizon_diagnostic_6.json"
        public_payload = json.loads(public_path.read_text(encoding="utf-8"))
        self.assertEqual(public_payload["task_set_hash"], diagnostic.task_set.task_set_hash)
        self.assertEqual(public_payload["task_count"], 6)
        self.assertEqual(public_payload["scenario_count"], 2)
        self.assertNotIn("task_ids", public_payload)
        self.assertIsNone(_APPWORLD_TASK_ID.search(public_path.read_text(encoding="utf-8")))
        manifest = _ROOT / "data/processed/spotify_capability_short_unit_manifest.json"
        public = _ROOT / "data/processed/spotify_capability_short_unit_public.json"
        tracked = _ROOT / "configs/tasks/train_spotify_capability_short.json"
        manifest.unlink(missing_ok=True)
        public.unlink(missing_ok=True)
        try:
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                code = builder.main(["--manifest", str(manifest), "--public", str(public)])
            self.assertEqual(code, 3)
            self.assertFalse(manifest.exists())
            self.assertFalse(public.exists())
            self.assertFalse(tracked.exists())
        finally:
            manifest.unlink(missing_ok=True)
            public.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
