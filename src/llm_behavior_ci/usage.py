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
``cache_read_tokens``, ``cache_write_tokens``. The ``input_tokens`` rate is
the uncached-input rate. ``reasoning_tokens`` is never priced: every
supported provider reports reasoning inside ``output_tokens``.

``ProviderCall`` keeps each provider's own counts. ``PROVIDER_USAGE_SEMANTICS``
names how they map to billable tokens, and ``token_cost`` is the one place
that applies the map, for both ``aggregate_usage`` and the calibration
projection:

- ``zai`` (Chat Completions): ``input_tokens`` is ``prompt_tokens``, which
  includes ``cache_read_tokens``. Uncached input is the difference.
- ``anthropic`` (Messages): ``input_tokens`` already excludes cache reads
  and cache writes, which are reported separately.

Each billable token is charged once, at one rate. When a table prices
``input_tokens`` but not ``cache_read_tokens``, cached input is charged at
the input rate. A provider without declared semantics cannot be priced.
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
PRICED_FIELDS = (
    "input_tokens",
    "output_tokens",
    "cache_read_tokens",
    "cache_write_tokens",
)
PROMPT_INCLUDES_CACHE_READ = "prompt_includes_cache_read"
INPUT_EXCLUDES_CACHE = "input_excludes_cache"
PROVIDER_USAGE_SEMANTICS = {
    "zai": PROMPT_INCLUDES_CACHE_READ,
    "anthropic": INPUT_EXCLUDES_CACHE,
}


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
        if provider not in PROVIDER_USAGE_SEMANTICS:
            raise UsageError(f"provider {provider} has no declared usage semantics")
        seen.add((provider, model_id))
        if not isinstance(rates, Mapping) or not rates:
            raise UsageError("per_million_tokens must be a non-empty object")
        pairs: list[tuple[str, float]] = []
        for name, rate in sorted(rates.items()):
            if name == "reasoning_tokens":
                raise UsageError(
                    "reasoning_tokens are part of output_tokens and are not priced separately"
                )
            if name not in PRICED_FIELDS:
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


def token_cost(
    provider: str,
    rates: Mapping[str, float],
    *,
    input_tokens: int | None,
    output_tokens: int | None,
    cache_read_tokens: int | None,
    cache_write_tokens: int | None,
) -> float | None:
    """Price one request or one episode from provider-reported counts.

    ``rates`` are per million tokens. Returns ``None`` when a count that a
    priced field needs was not reported. Raises ``UsageError`` for a provider
    without declared semantics, or for a Chat Completions cache count larger
    than its prompt.
    """

    semantics = PROVIDER_USAGE_SEMANTICS.get(provider)
    if semantics is None:
        raise UsageError(f"provider {provider} has no declared usage semantics")
    unknown = set(rates) - set(PRICED_FIELDS)
    if unknown:
        raise UsageError(f"unknown priced field {sorted(unknown)[0]}")
    total = 0.0
    if "output_tokens" in rates:
        if output_tokens is None:
            return None
        total += output_tokens * rates["output_tokens"]
    if "cache_write_tokens" in rates:
        if cache_write_tokens is None:
            return None
        total += cache_write_tokens * rates["cache_write_tokens"]
    if semantics == PROMPT_INCLUDES_CACHE_READ:
        if "input_tokens" in rates and "cache_read_tokens" in rates:
            if input_tokens is None or cache_read_tokens is None:
                return None
            if cache_read_tokens > input_tokens:
                raise UsageError("cached tokens exceed prompt tokens")
            total += (input_tokens - cache_read_tokens) * rates["input_tokens"]
            total += cache_read_tokens * rates["cache_read_tokens"]
        elif "input_tokens" in rates:
            if input_tokens is None:
                return None
            total += input_tokens * rates["input_tokens"]
        elif "cache_read_tokens" in rates:
            if cache_read_tokens is None:
                return None
            total += cache_read_tokens * rates["cache_read_tokens"]
    else:
        if "input_tokens" in rates:
            if input_tokens is None:
                return None
            total += input_tokens * rates["input_tokens"]
        if "cache_read_tokens" in rates:
            if cache_read_tokens is None:
                return None
            total += cache_read_tokens * rates["cache_read_tokens"]
        elif cache_read_tokens is not None and "input_tokens" in rates:
            total += cache_read_tokens * rates["input_tokens"]
    return total / 1_000_000.0


def uncached_input_tokens(
    provider: str,
    *,
    input_tokens: int | None,
    cache_read_tokens: int | None,
) -> int | None:
    """Input tokens billed at the uncached rate, or ``None`` when unknown."""

    semantics = PROVIDER_USAGE_SEMANTICS.get(provider)
    if semantics is None:
        raise UsageError(f"provider {provider} has no declared usage semantics")
    if input_tokens is None:
        return None
    if semantics == INPUT_EXCLUDES_CACHE:
        return input_tokens
    if cache_read_tokens is None:
        return None
    if cache_read_tokens > input_tokens:
        raise UsageError("cached tokens exceed prompt tokens")
    return input_tokens - cache_read_tokens


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
        for name in USAGE_FIELDS:
            reported = [getattr(call, name) for call in calls if getattr(call, name) is not None]
            totals[name] = sum(reported) if reported else None
            unknown[name] = sum(1 for call in calls if getattr(call, name) is None)
        cost: float | None = None
        version: str | None = None
        currency: str | None = None
        entry = None if pricing is None else pricing.entry_for(provider, model_id)
        if entry is not None:
            version = pricing.pricing_version
            currency = pricing.currency
            rates = dict(entry.per_million_tokens)
            priced: list[float] = []
            for call in calls:
                counts = {name: getattr(call, name) for name in PRICED_FIELDS}
                if call.status != "succeeded":
                    counts = {name: value or 0 for name, value in counts.items()}
                value = token_cost(provider, rates, **counts)
                if value is None:
                    break
                priced.append(value)
            else:
                cost = math.fsum(priced)
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
