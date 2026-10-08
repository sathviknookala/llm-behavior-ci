from __future__ import annotations

import unittest
from unittest.mock import patch

from llm_behavior_ci import usage as usage_module
from llm_behavior_ci.experiments.calibration import ScoredObservation, episode_token_cost
from llm_behavior_ci.records import ProviderCall
from llm_behavior_ci.usage import (
    UsageError,
    aggregate_usage,
    pricing_from_dict,
    token_cost,
    uncached_input_tokens,
)

_ZAI_RATES = {"input_tokens": 1.4, "cache_read_tokens": 0.26, "output_tokens": 4.4}
_ANTHROPIC_RATES = {
    "input_tokens": 3.0,
    "cache_read_tokens": 0.3,
    "cache_write_tokens": 3.75,
    "output_tokens": 15.0,
}


def _pricing(provider: str, model_id: str, rates: dict[str, float]):
    return pricing_from_dict(
        {
            "pricing_version": "test-v1",
            "currency": "USD",
            "entries": [
                {"provider": provider, "model_id": model_id, "per_million_tokens": rates}
            ],
        }
    )


def _call(provider: str = "zai", status: str = "succeeded", **counts: int | None) -> ProviderCall:
    return ProviderCall(
        provider=provider,
        model_id="glm-5.3" if provider == "zai" else "claude-sonnet-5-5",
        mode="execute",
        request_index=0,
        attempts=1,
        status=status,
        latency_seconds=0.5,
        error_kind=None if status == "succeeded" else "network_error",
        **counts,
    )


class _Episode:
    def __init__(self, calls: tuple[ProviderCall, ...]) -> None:
        self.provider_calls = calls


def _aggregate(calls: tuple[ProviderCall, ...], pricing):
    with patch.object(usage_module, "EpisodeResult", _Episode):
        return aggregate_usage(
            (_Episode(calls),), split="dev", configuration_hash="b" * 64, pricing=pricing
        )


class ZaiSemanticsTests(unittest.TestCase):
    def test_cached_prompt_tokens_are_charged_once(self) -> None:
        cost = token_cost(
            "zai",
            _ZAI_RATES,
            input_tokens=17_708,
            output_tokens=155,
            cache_read_tokens=17_664,
            cache_write_tokens=None,
        )
        expected = (44 * 1.4 + 17_664 * 0.26 + 155 * 4.4) / 1_000_000
        self.assertAlmostEqual(cost, expected, places=12)
        self.assertEqual(
            uncached_input_tokens("zai", input_tokens=17_708, cache_read_tokens=17_664), 44
        )

    def test_same_prompt_costs_less_when_cached(self) -> None:
        cold = token_cost(
            "zai", _ZAI_RATES, input_tokens=1000, output_tokens=0,
            cache_read_tokens=0, cache_write_tokens=None,
        )
        warm = token_cost(
            "zai", _ZAI_RATES, input_tokens=1000, output_tokens=0,
            cache_read_tokens=1000, cache_write_tokens=None,
        )
        self.assertAlmostEqual(cold, 1000 * 1.4 / 1_000_000)
        self.assertAlmostEqual(warm, 1000 * 0.26 / 1_000_000)

    def test_cache_above_prompt_is_rejected(self) -> None:
        with self.assertRaises(UsageError):
            token_cost(
                "zai", _ZAI_RATES, input_tokens=10, output_tokens=1,
                cache_read_tokens=11, cache_write_tokens=None,
            )

    def test_unknown_cache_leaves_cost_unset_when_cache_is_priced(self) -> None:
        self.assertIsNone(
            token_cost(
                "zai", _ZAI_RATES, input_tokens=10, output_tokens=1,
                cache_read_tokens=None, cache_write_tokens=None,
            )
        )
        self.assertAlmostEqual(
            token_cost(
                "zai", {"input_tokens": 2.0}, input_tokens=1000, output_tokens=None,
                cache_read_tokens=None, cache_write_tokens=None,
            ),
            0.002,
        )

    def test_aggregate_and_calibration_agree(self) -> None:
        pricing = _pricing("zai", "glm-5.3", _ZAI_RATES)
        calls = (
            _call(input_tokens=17_673, output_tokens=71, cache_read_tokens=0, reasoning_tokens=0),
            _call(input_tokens=17_708, output_tokens=155, cache_read_tokens=17_664, reasoning_tokens=3),
        )
        (row,) = _aggregate(calls, pricing)
        observation = ScoredObservation(
            observation_id="o",
            configuration_hash="b" * 64,
            task_set_hash=None,
            split="dev",
            task_id="t",
            scenario_id=None,
            mode="execute",
            role="reference",
            repetition=None,
            pair_id=None,
            success=None,
            requirement_fraction=None,
            termination_reason="appworld_completed",
            latency_seconds=1.0,
            input_tokens=17_673 + 17_708,
            output_tokens=71 + 155,
            cache_read_tokens=17_664,
            reasoning_tokens=3,
            request_count=2,
            provider="zai",
            model_id="glm-5.3",
            source_name="fixture.sqlite",
        )
        self.assertAlmostEqual(row.cost, episode_token_cost(observation, pricing), places=12)
        double_charged = (17_673 + 17_708) * 1.4 + 17_664 * 0.26 + 226 * 4.4
        self.assertLess(row.cost, double_charged / 1_000_000)

    def test_failed_request_counts_reported_parts(self) -> None:
        pricing = _pricing("zai", "glm-5.3", _ZAI_RATES)
        calls = (
            _call(input_tokens=1000, output_tokens=10, cache_read_tokens=400),
            _call(status="failed", input_tokens=None, output_tokens=None, cache_read_tokens=None),
        )
        (row,) = _aggregate(calls, pricing)
        self.assertAlmostEqual(row.cost, (600 * 1.4 + 400 * 0.26 + 10 * 4.4) / 1_000_000)
        unknown = (_call(input_tokens=1000, output_tokens=None, cache_read_tokens=400),)
        self.assertIsNone(_aggregate(unknown, pricing)[0].cost)


class AnthropicSemanticsTests(unittest.TestCase):
    def test_input_already_excludes_cache(self) -> None:
        cost = token_cost(
            "anthropic",
            _ANTHROPIC_RATES,
            input_tokens=200,
            output_tokens=50,
            cache_read_tokens=5000,
            cache_write_tokens=100,
        )
        expected = (200 * 3.0 + 5000 * 0.3 + 100 * 3.75 + 50 * 15.0) / 1_000_000
        self.assertAlmostEqual(cost, expected, places=12)
        self.assertEqual(
            uncached_input_tokens("anthropic", input_tokens=200, cache_read_tokens=5000), 200
        )

    def test_aggregate_uses_the_same_contract(self) -> None:
        pricing = _pricing("anthropic", "claude-sonnet-5-5", _ANTHROPIC_RATES)
        call = _call(
            provider="anthropic",
            input_tokens=200,
            output_tokens=50,
            cache_read_tokens=5000,
            cache_write_tokens=100,
        )
        (row,) = _aggregate((call,), pricing)
        expected = (200 * 3.0 + 5000 * 0.3 + 100 * 3.75 + 50 * 15.0) / 1_000_000
        self.assertAlmostEqual(row.cost, expected, places=12)


class PricingContractTests(unittest.TestCase):
    def test_reasoning_tokens_cannot_be_priced(self) -> None:
        with self.assertRaises(UsageError):
            _pricing("zai", "glm-5.3", {**_ZAI_RATES, "reasoning_tokens": 9.0})

    def test_provider_without_semantics_is_rejected(self) -> None:
        with self.assertRaises(UsageError):
            _pricing("vllm", "qwen", {"input_tokens": 1.0})
        with self.assertRaises(UsageError):
            token_cost(
                "vllm", {"input_tokens": 1.0}, input_tokens=1, output_tokens=1,
                cache_read_tokens=0, cache_write_tokens=None,
            )


if __name__ == "__main__":
    unittest.main()
