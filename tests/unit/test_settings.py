import json
import unittest
from dataclasses import MISSING, fields
from pathlib import Path

import llm_behavior_ci.config as configuration_module
from llm_behavior_ci.config import (
    HASHED_FIELDS,
    CanarySettings,
    ConfigError,
    EpisodeIdentity,
    GateSettings,
    MonitorSettings,
    RunConfiguration,
    RunIdentity,
    StoppingRule,
    StreamSettings,
)

_HASH = "a" * 64
_TASK_HASH = "b" * 64
_GIT = "c" * 40


def _stopping_rule(**overrides: object) -> StoppingRule:
    values: dict[str, object] = {
        "name": "cusum",
        "alpha": 0.2,
        "horizon_episodes": 10,
    }
    values.update(overrides)
    return StoppingRule(**values)


def _stream() -> StreamSettings:
    return StreamSettings(
        split="train",
        selection_rule="fixed-v1",
        selection_seed=1,
        task_set_hash=_TASK_HASH,
        stream_seed=2,
        arrival_rate_per_second=0.25,
        concurrency=1,
        with_replacement=True,
        task_mix_rule="difficulty-only",
    )


def _gate() -> GateSettings:
    return GateSettings(
        confidence_level=0.9,
        bootstrap_resamples=12,
        score_margin=-0.02,
        kl_limit_nats=0.0,
        mmd_bandwidth=1.5,
        mmd_permutations=19,
        mmd_alpha=0.2,
        plan_format_version="plan-v1",
    )


def _canary() -> CanarySettings:
    return CanarySettings(
        fraction=1.0,
        outcome_delay_seconds=0.0,
        harm_margin=0.1,
        stopping_rule=_stopping_rule(threshold=-0.02),
        metric_orientation="higher_is_better",
        promotion_policy="horizon_reached_without_harm",
    )


def _monitor() -> MonitorSettings:
    return MonitorSettings(
        reference_configuration_hash=_HASH,
        outcome_delay_seconds=0.0,
        signals=("task_success", "requirement_fraction"),
        stopping_rules=(_stopping_rule(), _stopping_rule(name="adwin")),
    )


class SettingsContractTests(unittest.TestCase):
    def test_settings_require_explicit_values(self) -> None:
        for cls in (StreamSettings, GateSettings, CanarySettings, MonitorSettings):
            with self.subTest(cls=cls.__name__):
                with self.assertRaises(TypeError):
                    cls()
                for field in fields(cls):
                    self.assertIs(field.default, MISSING)
                    self.assertIs(field.default_factory, MISSING)
        with self.assertRaises(TypeError):
            StoppingRule(name="cusum", horizon_episodes=10)
        rule = StoppingRule.from_dict(
            {"name": "cusum", "alpha": 0.2, "horizon_episodes": 10}
        )
        self.assertIsNone(rule.threshold)
        self.assertEqual(rule.alpha, 0.2)
        for field in fields(StoppingRule):
            if field.name in ("threshold", "slack"):
                self.assertIsNone(field.default)
            else:
                self.assertIs(field.default, MISSING)

    def test_settings_round_trip_and_reject_unknown_or_invalid_values(self) -> None:
        for original in (_stream(), _gate(), _canary(), _monitor()):
            with self.subTest(cls=type(original).__name__):
                payload = json.loads(json.dumps(original.to_dict()))
                self.assertEqual(type(original).from_dict(payload), original)
        stream_payload = _stream().to_dict()
        self.assertIs(stream_payload["with_replacement"], True)
        self.assertIsInstance(stream_payload["concurrency"], int)
        self.assertNotIsInstance(stream_payload["concurrency"], bool)
        self.assertIsInstance(stream_payload["arrival_rate_per_second"], float)
        stream_payload["task_ids"] = ["local-task"]
        with self.assertRaises(ConfigError):
            StreamSettings.from_dict(stream_payload)
        invalid = (
            (CanarySettings, _canary().to_dict(), "fraction", 0.0),
            (CanarySettings, _canary().to_dict(), "harm_margin", 0.0),
            (CanarySettings, _canary().to_dict(), "outcome_delay_seconds", -1.0),
            (CanarySettings, _canary().to_dict(), "metric_orientation", "sideways"),
            (
                CanarySettings,
                _canary().to_dict(),
                "promotion_policy",
                "superior_evidence",
            ),
            (StreamSettings, _stream().to_dict(), "arrival_rate_per_second", 0.0),
            (StreamSettings, _stream().to_dict(), "concurrency", True),
            (GateSettings, _gate().to_dict(), "mmd_alpha", 1.0),
            (GateSettings, _gate().to_dict(), "kl_limit_nats", -0.1),
        )
        for cls, payload, key, value in invalid:
            with self.subTest(cls=cls.__name__, field=key):
                payload[key] = value
                with self.assertRaises(ConfigError):
                    cls.from_dict(payload)

    def test_boundaries_and_monitor_signals(self) -> None:
        self.assertEqual(_canary().fraction, 1.0)
        self.assertEqual(_canary().outcome_delay_seconds, 0.0)
        self.assertEqual(_gate().score_margin, -0.02)
        self.assertEqual(_gate().kl_limit_nats, 0.0)
        with self.assertRaises(ConfigError):
            MonitorSettings.from_dict(
                {
                    **_monitor().to_dict(),
                    "signals": ["task_success", "task_success"],
                }
            )
        with self.assertRaises(ConfigError):
            MonitorSettings.from_dict({**_monitor().to_dict(), "signals": []})
        with self.assertRaises(ConfigError):
            MonitorSettings.from_dict(
                {**_monitor().to_dict(), "signals": ["latency"]}
            )
        with self.assertRaises(ConfigError):
            MonitorSettings(
                reference_configuration_hash=_HASH,
                outcome_delay_seconds=0.0,
                signals=["task_success"],
                stopping_rules=(_stopping_rule(),),
            )
        restored = MonitorSettings.from_dict(_monitor().to_dict())
        self.assertIsInstance(restored.signals, tuple)
        self.assertEqual(restored.stopping_rules[0].alpha, 0.2)
        self.assertIsNone(restored.stopping_rules[0].threshold)

    def test_settings_stay_outside_the_run_configuration_hash(self) -> None:
        self.assertNotIn("arrival_rate_per_second", HASHED_FIELDS)
        self.assertNotIn("score_margin", HASHED_FIELDS)
        self.assertNotIn("harm_margin", HASHED_FIELDS)
        self.assertFalse(any(field.name == "stream" for field in fields(RunConfiguration)))
        source = Path(configuration_module.__file__).read_text(encoding="utf-8")
        self.assertNotIn("DEMO_", source)
        self.assertNotIn("0.05", source)
        self.assertNotIn("os.environ", source)

    def test_identity_round_trip_preserves_optional_ids(self) -> None:
        run = RunIdentity(
            run_id=f"{_HASH}.{'1' * 32}",
            configuration_hash=_HASH,
            task_set_hash=_TASK_HASH,
            protocol_hash=None,
            git_commit=_GIT,
        )
        self.assertEqual(RunIdentity.from_dict(run.to_dict()), run)
        omitted = run.to_dict()
        del omitted["protocol_hash"]
        self.assertIsNone(RunIdentity.from_dict(omitted).protocol_hash)
        episode = EpisodeIdentity(
            episode_id=f"{run.run_id}.{'2' * 32}",
            run_id=run.run_id,
        )
        self.assertIsNone(episode.pair_id)
        self.assertEqual(EpisodeIdentity.from_dict(episode.to_dict()), episode)
        with self.assertRaises(ConfigError):
            RunIdentity.from_dict({**run.to_dict(), "run_seed": 1})
