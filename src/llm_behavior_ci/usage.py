"""Hosted usage aggregates and versioned pricing.

``aggregate_usage`` groups the sanitized ``ProviderCall`` rows of local
episodes by provider, model, and mode into public ``UsageAggregate``
records. Cost comes only from a ``PricingTable`` loaded from a versioned
JSON input; without one, or with an unknown count in a priced field, cost
stays ``None``. A pricing file is caller-supplied and is not a measured
bill.

Pricing JSON::

    {"pricing_version": "...", "currency": "USD",
     "entries": [{"provider": "zai", "model_id": "...",
                  "per_million_tokens": {"input_tokens": 0.0,
                                         "output_tokens": 0.0}}]}

Priced fields are any of ``input_tokens``, ``output_tokens``,
``cache_read_tokens``, ``cache_write_tokens``, ``reasoning_tokens``.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from llm_behavior_ci.records import EpisodeResult, ProviderCall, UsageAggregate

USAGE_FIELDS = (
    "input_tokens",
    "output_tokens",
    "cache_read_tokens",
    "cache_write_tokens",
    "reasoning_tokens",
)


class UsageError(ValueError):
    pass


@dataclass(frozen=True)
class PricingEntry:
    provider: str
    model_id: str
    per_million_tokens: tuple[tuple[str, float], ...]


@dataclass(frozen=True)
class PricingTable:
    pricing_version: str
    currency: str
    entries: tuple[PricingEntry, ...]

    def entry_for(self, provider: str, model_id: str) -> PricingEntry | None:
        for entry in self.entries:
            if entry.provider == provider and entry.model_id == model_id:
                return entry
        return None


def pricing_from_dict(payload: object) -> PricingTable:
    if not isinstance(payload, Mapping):
        raise UsageError("pricing must be an object")
    version = payload.get("pricing_version")
    currency = payload.get("currency")
    raw_entries = payload.get("entries")
    if not isinstance(version, str) or not version.strip():
        raise UsageError("pricing_version is required")
    if not isinstance(currency, str) or not currency.strip():
        raise UsageError("currency is required")
    if not isinstance(raw_entries, list) or not raw_entries:
        raise UsageError("pricing entries must be a non-empty list")
    entries: list[PricingEntry] = []
    seen: set[tuple[str, str]] = set()
    for item in raw_entries:
        if not isinstance(item, Mapping):
            raise UsageError("pricing entry must be an object")
        provider = item.get("provider")
        model_id = item.get("model_id")
        rates = item.get("per_million_tokens")
        if not isinstance(provider, str) or not isinstance(model_id, str):
            raise UsageError("pricing entry needs provider and model_id")
        if (provider, model_id) in seen:
            raise UsageError("pricing entry is duplicated")
        seen.add((provider, model_id))
        if not isinstance(rates, Mapping) or not rates:
            raise UsageError("per_million_tokens must be a non-empty object")
        pairs: list[tuple[str, float]] = []
        for name, rate in sorted(rates.items()):
            if name not in USAGE_FIELDS:
                raise UsageError(f"unknown priced field {name}")
            if isinstance(rate, bool) or not isinstance(rate, (int, float)):
                raise UsageError("rates must be numbers")
            if not math.isfinite(float(rate)) or float(rate) < 0.0:
                raise UsageError("rates must be nonnegative and finite")
            pairs.append((name, float(rate)))
        entries.append(PricingEntry(provider, model_id, tuple(pairs)))
    return PricingTable(version, currency, tuple(entries))


def load_pricing(path: Path) -> PricingTable:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise UsageError("pricing file is not readable JSON") from error
    return pricing_from_dict(payload)


def aggregate_usage(
    episodes: Sequence[EpisodeResult],
    *,
    split: str,
    configuration_hash: str,
    pricing: PricingTable | None = None,
) -> tuple[UsageAggregate, ...]:
    """One public aggregate per provider, model, and mode, sorted."""

    groups: dict[tuple[str, str, str], list[ProviderCall]] = {}
    for episode in episodes:
        if not isinstance(episode, EpisodeResult):
            raise UsageError("usage aggregation requires episode results")
        for call in episode.provider_calls:
            groups.setdefault((call.provider, call.model_id, call.mode), []).append(call)
    aggregates: list[UsageAggregate] = []
    for (provider, model_id, mode), calls in sorted(groups.items()):
        totals: dict[str, int | None] = {}
        unknown: dict[str, int] = {}
        succeeded_unknown: dict[str, int] = {}
        for name in USAGE_FIELDS:
            reported = [getattr(call, name) for call in calls if getattr(call, name) is not None]
            totals[name] = sum(reported) if reported else None
            unknown[name] = sum(1 for call in calls if getattr(call, name) is None)
            succeeded_unknown[name] = sum(
                1
                for call in calls
                if call.status == "succeeded" and getattr(call, name) is None
            )
        cost: float | None = None
        version: str | None = None
        currency: str | None = None
        entry = None if pricing is None else pricing.entry_for(provider, model_id)
        if entry is not None:
            version = pricing.pricing_version
            currency = pricing.currency
            if all(succeeded_unknown[name] == 0 for name, _ in entry.per_million_tokens):
                cost = sum(
                    (totals[name] or 0) * rate / 1_000_000.0
                    for name, rate in entry.per_million_tokens
                )
        aggregates.append(
            UsageAggregate(
                split=split,
                configuration_hash=configuration_hash,
                provider=provider,
                model_id=model_id,
                mode=mode,
                request_count=len(calls),
                failed_request_count=sum(1 for call in calls if call.status != "succeeded"),
                attempt_count=sum(call.attempts for call in calls),
                latency_seconds=sum(call.latency_seconds for call in calls),
                **totals,
                **{f"{name}_unknown_requests": unknown[name] for name in USAGE_FIELDS},
                pricing_version=version,
                currency=currency,
                cost=cost,
            )
        )
    return tuple(aggregates)
