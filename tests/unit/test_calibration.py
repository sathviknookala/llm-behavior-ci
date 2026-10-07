import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from llm_behavior_ci.config import RunConfiguration, run_configuration_hash
from llm_behavior_ci.experiments.calibration import (
    CalibrationError,
    InventoryCounts,
    ScoredObservation,
    bind_observations,
    build_slots,
    calibration_report,
    checkpoint_identity,
    episode_token_cost,
    load_checkpoint,
    occupied_from_checkpoint,
    read_capture_file,
    read_sqlite_store,
    select_batch,
    write_checkpoint,
)
from llm_behavior_ci.records import assert_public_payload
from llm_behavior_ci.storage import EpisodeStore
from llm_behavior_ci.tasks.selection import (
    TaskSet,
    canonical_task_set_bytes,
    task_set_hash_from_bytes,
)
from llm_behavior_ci.usage import pricing_from_dict

_ROOT = Path(__file__).resolve().parents[2]


def _task_set(split: str = "dev") -> TaskSet:
    tasks = (("task-a", "scenario-a"), ("task-b", "scenario-b"))
    payload = canonical_task_set_bytes(
        appworld_version="0.1.3.post1",
        split=split,
        selection_rule="deterministic_sample",
        selection_seed=17,
        tasks=tasks,
    )
    return TaskSet(
        appworld_version="0.1.3.post1",
        split=split,
        selection_rule="deterministic_sample",
        selection_seed=17,
        task_count=2,
        scenario_count=2,
        task_ids=tuple(task_id for task_id, _scenario in tasks),
        scenario_ids=tuple(scenario for _task_id, scenario in tasks),
        task_set_hash=task_set_hash_from_bytes(payload),
    )


def _configuration(task_set: TaskSet) -> RunConfiguration:
    document = json.loads(
        (_ROOT / "configs/models/glm_5_3_general_experimental.json").read_text(encoding="utf-8")
    )
    document["task"] = {
        "appworld_version": task_set.appworld_version,
        "split": task_set.split,
        "selection_rule": task_set.selection_rule,
        "selection_seed": task_set.selection_seed,
        "task_count": task_set.task_count,
        "task_set_hash": task_set.task_set_hash,
    }
    document["run_seed"] = 17
    document["git_commit"] = "a" * 40
    document["protocol_hash"] = None
    return RunConfiguration.from_dict(document)


def _observation(**overrides: object) -> ScoredObservation:
    values: dict[str, object] = {
        "observation_id": "obs-1",
        "configuration_hash": "b" * 64,
        "task_set_hash": "c" * 64,
        "split": "dev",
        "task_id": "task-a",
        "scenario_id": "scenario-a",
        "mode": "execute",
        "role": "unpaired",
        "repetition": None,
        "pair_id": None,
        "success": False,
        "requirement_fraction": 0.25,
        "termination_reason": "step_limit",
        "latency_seconds": 1.5,
        "input_tokens": 1_000_000,
        "output_tokens": 1_000,
        "cache_read_tokens": 800_000,
        "reasoning_tokens": 100,
        "request_count": 2,
        "provider": "zai",
        "model_id": "glm-5.3",
        "source_name": "fixture.sqlite",
    }
    values.update(overrides)
    return ScoredObservation(**values)  # type: ignore[arg-type]


def _scenarios(task_set: TaskSet) -> dict[str, str]:
    return {
        task_id: str(scenario_id)
        for task_id, scenario_id in zip(task_set.task_ids, task_set.scenario_ids, strict=True)
    }


def _pricing():
    return pricing_from_dict(
        {
            "pricing_version": "test-v1",
            "currency": "USD",
            "entries": [
                {
                    "provider": "zai",
                    "model_id": "glm-5.3",
                    "per_million_tokens": {
                        "input_tokens": 1.4,
                        "cache_read_tokens": 0.26,
                        "output_tokens": 4.4,
                        "reasoning_tokens": 9.0,
                    },
                }
            ],
        }
    )


def _report(task_set, configuration, observations, bound):
    return calibration_report(
        configuration=configuration,
        task_set=task_set,
        observations=observations,
        bound=bound,
        counts=InventoryCounts(),
        repetitions=1,
        confidence_level=0.9,
        resamples=20,
        seed=3,
        max_model_episodes=10,
        pricing=_pricing(),
        provenance={"head_matches": True, "relevant_dirty": False, "run_allowed": True},
    )


class CalibrationPlanTests(unittest.TestCase):
    def test_closed_split_is_refused(self) -> None:
        with self.assertRaises(CalibrationError):
            build_slots(_task_set("test_normal"), repetitions=1, aa_modes=("execute",))

    def test_aa_reference_is_not_also_a_baseline_slot(self) -> None:
        task_set = _task_set()
        configuration = _configuration(task_set)
        digest = run_configuration_hash(configuration)
        slots = build_slots(task_set, repetitions=1, aa_modes=("execute",))
        reference = _observation(
            observation_id="ref",
            configuration_hash=digest,
            role="reference",
            pair_id="pair-1",
            success=False,
        )
        candidate = _observation(
            observation_id="cand",
            configuration_hash=digest,
            role="candidate",
            pair_id="pair-1",
            success=True,
            requirement_fraction=1.0,
        )
        old_reference = _observation(
            observation_id="old-ref",
            configuration_hash="d" * 64,
            role="reference",
            pair_id="old-pair",
            success=True,
            requirement_fraction=1.0,
        )
        old_candidate = _observation(
            observation_id="old-cand",
            configuration_hash="d" * 64,
            role="candidate",
            pair_id="old-pair",
            success=True,
            requirement_fraction=1.0,
        )
        bound = bind_observations(
            slots,
            (reference, candidate, old_reference, old_candidate),
            configuration_hash=digest,
            task_scenarios=_scenarios(task_set),
        )
        self.assertEqual(
            {item.observation_id for item in bound.filled["aa_execute:0:0"]},
            {"ref", "cand"},
        )
        self.assertNotIn("baseline_production:0:0", bound.filled)
        report = _report(
            task_set, configuration, (reference, candidate, old_reference, old_candidate), bound
        )
        assert_public_payload(report)
        self.assertNotIn("task-a", json.dumps(report))
        overlap = {item["configuration_hash"]: item for item in report["reference_baseline_overlap"]}
        target = overlap[digest]
        self.assertEqual(target["exclusive_policy_overlap"], 0)
        self.assertEqual(target["healthy_execute_aa_references"], 1)
        self.assertEqual(target["overlapping_episodes_if_counted_in_both"], 1)
        self.assertTrue(target["supports_baseline_success"])
        self.assertFalse(target["supports_do_nothing_contrast"])
        preserved = overlap["d" * 64]
        self.assertFalse(preserved["fills_target_slots"])
        self.assertFalse(preserved["relabeled_onto_target"])
        self.assertEqual(report["completion"]["model_episodes_filled"], 2)
        self.assertEqual(report["sampling"]["allocation_status"], "provisional")
        self.assertEqual(report["sampling"]["power_based_target"], "unmeasured")
        self.assertIn(
            "repeated_production_draws_of_the_same_tasks",
            report["sampling"]["pilot_evidence_needed"],
        )

    def test_other_configuration_projects_cost_without_filling_slots(self) -> None:
        task_set = _task_set()
        configuration = _configuration(task_set)
        slots = build_slots(task_set, repetitions=1, aa_modes=("execute",))
        prior = _observation(observation_id="prior", configuration_hash="e" * 64, success=True)
        bound = bind_observations(
            slots, (prior,), configuration_hash=run_configuration_hash(configuration), task_scenarios=_scenarios(task_set)
        )
        self.assertEqual(bound.filled, {})
        report = _report(task_set, configuration, (prior,), bound)
        self.assertEqual(report["cost"]["status"], "projected_from_other_configuration")
        self.assertEqual(report["cost"]["prior_configuration_hashes"], ["e" * 64])
        self.assertEqual(report["cost"]["target_priced_episodes"], 0)
        self.assertAlmostEqual(report["cost"]["per_execute_episode"], 0.2 * 1.4 + 0.8 * 0.26 + 0.001 * 4.4)
        self.assertAlmostEqual(
            report["cost"]["next_batch_estimated"],
            report["cost"]["per_execute_episode"] * report["next_batch"]["model_episodes"],
        )

    def test_batch_cap_does_not_split_a_pair(self) -> None:
        slots = build_slots(_task_set(), repetitions=1, aa_modes=("execute",))
        first = select_batch(slots, {}, max_model_episodes=2)
        self.assertEqual(sum(slot.model_episodes for slot in first), 2)
        self.assertNotIn("aa_execute", [slot.kind for slot in first])
        filled = {slot.slot_id: () for slot in slots if slot.kind.startswith("baseline")}
        self.assertEqual(select_batch(slots, filled, max_model_episodes=1), ())

    def test_checkpoint_refuses_a_different_configuration(self) -> None:
        task_set = _task_set()
        configuration = _configuration(task_set)
        digest = run_configuration_hash(configuration)
        slots = build_slots(task_set, repetitions=1, aa_modes=("execute",))
        bound = bind_observations(
            slots,
            (_observation(observation_id="kept", configuration_hash=digest, success=True),),
            configuration_hash=digest,
            task_scenarios=_scenarios(task_set),
        )
        identity = checkpoint_identity(configuration, task_set, repetitions=1, aa_modes=("execute",))
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "checkpoint.json"
            write_checkpoint(path, identity=identity, bound=bound)
            payload = load_checkpoint(path)
            self.assertIn("baseline_production:0:0", occupied_from_checkpoint(payload, identity))
            other = dict(identity)
            other["configuration_hash"] = "f" * 64
            with self.assertRaises(CalibrationError):
                occupied_from_checkpoint(payload, other)

    def test_cost_does_not_add_reasoning_tokens(self) -> None:
        cost = episode_token_cost(_observation(), _pricing())
        self.assertAlmostEqual(cost, 0.2 * 1.4 + 0.8 * 0.26 + 0.001 * 4.4)

    def test_sqlite_and_capture_inventory_skip_closed_splits(self) -> None:
        episode = {
            "episode": {"episode_id": "ep-1", "pair_id": None},
            "run": {"configuration_hash": "e" * 64, "task_set_hash": "f" * 64},
            "task": {"task_id": "task-a", "scenario_id": "scenario-a", "split": "dev"},
            "mode": "execute",
            "role": None,
            "termination_reason": "appworld_completed",
            "evaluator_outcome": {"success": True, "requirement_fraction": 1.0},
            "provider_calls": [
                {
                    "provider": "zai",
                    "model_id": "glm-5.3",
                    "latency_seconds": 0.5,
                    "input_tokens": 10,
                    "output_tokens": 2,
                    "cache_read_tokens": 4,
                    "reasoning_tokens": 1,
                }
            ],
            "model_steps": [],
            "started_at": "2026-10-07T00:00:00+00:00",
            "ended_at": "2026-10-07T00:00:01+00:00",
        }
        closed = json.loads(json.dumps(episode))
        closed["episode"]["episode_id"] = "ep-closed"
        closed["task"]["split"] = "test_normal"
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            store = root / "episodes.sqlite"
            EpisodeStore(store).close()
            connection = sqlite3.connect(store)
            connection.execute(
                "INSERT INTO episodes (episode_id, run_id, task_id, state, identity_json, run_json, result_json) VALUES (?, ?, ?, ?, ?, ?, ?)",
                ("ep-1", "run", "task-a", "finished", "{}", "{}", json.dumps(episode)),
            )
            connection.execute(
                "INSERT INTO episodes (episode_id, run_id, task_id, state, identity_json, run_json, result_json) VALUES (?, ?, ?, ?, ?, ?, ?)",
                ("ep-closed", "run", "hidden", "finished", "{}", "{}", json.dumps(closed)),
            )
            connection.commit()
            connection.close()
            found, counts = read_sqlite_store(store)
            self.assertEqual([item.observation_id for item in found], ["ep-1"])
            self.assertEqual(counts.excluded_closed_split, 1)
            capture = {
                "capture": {
                    "records": [
                        {
                            "schedule": {"repetition": 0},
                            "pair": {
                                "reference": {
                                    **episode,
                                    "episode": {"episode_id": "ep-1", "pair_id": "pair"},
                                    "role": "reference",
                                },
                                "candidate": {
                                    **episode,
                                    "episode": {"episode_id": "ep-2", "pair_id": "pair"},
                                    "role": "candidate",
                                },
                            },
                        }
                    ]
                }
            }
            path = root / "capture.json"
            path.write_text(json.dumps(capture), encoding="utf-8")
            captured, _counts = read_capture_file(path)
            self.assertEqual({item.role for item in captured}, {"reference", "candidate"})


if __name__ == "__main__":
    unittest.main()
