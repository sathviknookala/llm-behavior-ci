"""Full-vocabulary teacher-forced plan scoring for truncation checks.

Scores one frozen plan through an injected scorer. ``load_transformers_bundle``
imports Transformers only when a caller asks it to load the configuration's
pinned repository and revision. Top-k slices are cuts of that same dense
distribution. Numeric truncation error goes through ``compare_plan_kl``.
vLLM-versus-dense top-k agreement is a separate report. This module does
not choose full versus top-k, a protocol tolerance, or a threshold.

The collection order is sequential because the dense model is not assumed
to fit beside the running vLLM server: persist the frozen plan locally,
stop vLLM, load Transformers, score the same plan, then compare.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from llm_behavior_ci.config import RunConfiguration, run_configuration_hash
from llm_behavior_ci.experiments.validation import (
    KLApproximationReport,
    KLPositionSample,
    compare_plan_kl,
    default_results_root,
)
from llm_behavior_ci.records import TokenLogprob
from llm_behavior_ci.runtime.scoring import (
    FidelityProofError,
    PositionAlignmentError,
    ScoredPosition,
    ScoringContract,
    SupportAlignmentError,
    _is_normalized,
    fingerprint_messages,
    verify_fidelity_claim,
    verify_scoring_contracts,
)

DEFAULT_TOP_K = 20


class FullVocabError(ValueError):
    pass


class FullVocabScorerResult(Protocol):
    """One scorer return: dense full-vocabulary log-probabilities.

    ``position_log_probabilities[i][token_id]`` is the log-probability of
    ``token_id`` at forced-plan position ``i``. Length of each row equals
    ``vocabulary_size``. ``forced_token_ids[i]`` is the token actually
    forced at that position (the plan token identity used for alignment
    with a vLLM ``teacher_force_plan`` table).
    """

    @property
    def vocabulary_size(self) -> int: ...

    @property
    def forced_token_ids(self) -> tuple[int, ...]: ...

    @property
    def position_log_probabilities(self) -> tuple[tuple[float, ...], ...]: ...


@dataclass(frozen=True)
class DenseFullVocabResult:
    """Concrete scorer payload used by fakes and local collectors."""

    vocabulary_size: int
    forced_token_ids: tuple[int, ...]
    position_log_probabilities: tuple[tuple[float, ...], ...]


FullVocabScorer = Callable[
    ...,
    FullVocabScorerResult,
]


@dataclass(frozen=True)
class FullVocabPlanScore:
    """Aligned full-vocabulary teacher-forced score for one frozen plan."""

    vocabulary_size: int
    input_prefix_fingerprint: str
    forced_token_ids: tuple[int, ...]
    position_log_probabilities: tuple[tuple[float, ...], ...]
    top_k_default: tuple[tuple[TokenLogprob, ...], ...]
    top_k_larger: tuple[tuple[TokenLogprob, ...], ...]
    larger_k: int

    @property
    def position_count(self) -> int:
        return len(self.forced_token_ids)


def score_frozen_plan(
    *,
    messages: Sequence[Mapping[str, str]],
    plan_text: str,
    scorer: FullVocabScorer,
    larger_k: int,
    default_k: int = DEFAULT_TOP_K,
) -> FullVocabPlanScore:
    """Teacher-force one frozen plan through an injected full-vocab scorer.

    ``messages`` and ``plan_text`` match the vLLM ``teacher_force_plan`` /
    ``teacher_force_payload`` contract: the conversation prefix without the
    plan, plus the frozen plan text. Does not call vLLM or load weights.
    """

    if plan_text == "":
        raise FullVocabError("plan_text is required for teacher forcing")
    if (
        not isinstance(default_k, int)
        or isinstance(default_k, bool)
        or default_k <= 0
    ):
        raise FullVocabError("default_k must be a positive int")
    if (
        not isinstance(larger_k, int)
        or isinstance(larger_k, bool)
        or larger_k <= default_k
    ):
        raise FullVocabError(f"larger_k must be an int greater than {default_k}")
    fingerprint = fingerprint_messages(messages)
    raw = scorer(messages=list(messages), plan_text=plan_text)
    vocabulary_size = _positive_int(raw.vocabulary_size, "vocabulary_size")
    forced = tuple(int(token_id) for token_id in raw.forced_token_ids)
    rows = tuple(
        tuple(float(value) for value in row)
        for row in raw.position_log_probabilities
    )
    if not forced or not rows:
        raise FullVocabError("scorer returned no forced-plan positions")
    if len(forced) != len(rows):
        raise FullVocabError(
            "forced_token_ids and position_log_probabilities length differ"
        )
    if larger_k > vocabulary_size or default_k > vocabulary_size:
        raise FullVocabError("requested top-k exceeds vocabulary_size")
    for index, (token_id, row) in enumerate(zip(forced, rows, strict=True)):
        if len(row) != vocabulary_size:
            raise FullVocabError(
                f"position {index} support length differs from vocabulary_size"
            )
        if token_id < 0 or token_id >= vocabulary_size:
            raise FullVocabError(
                f"forced token id out of vocabulary at position {index}"
            )
    return FullVocabPlanScore(
        vocabulary_size=vocabulary_size,
        input_prefix_fingerprint=fingerprint,
        forced_token_ids=forced,
        position_log_probabilities=rows,
        top_k_default=tuple(top_k_slice(row, default_k) for row in rows),
        top_k_larger=tuple(top_k_slice(row, larger_k) for row in rows),
        larger_k=larger_k,
    )


def top_k_slice(
    log_probabilities: Sequence[float],
    k: int,
) -> tuple[TokenLogprob, ...]:
    """The top-k cut of one full-vocabulary position, ranked by log-prob.

    Ties break toward the smaller token id, matching ``compare_plan_kl``.
    """

    if not isinstance(k, int) or isinstance(k, bool) or k <= 0:
        raise FullVocabError("k must be a positive int")
    if k > len(log_probabilities):
        raise FullVocabError("k exceeds the supplied support")
    indexes = _top_indexes(log_probabilities, k)
    return tuple(
        TokenLogprob(
            token_id=token_id,
            logprob=float(log_probabilities[token_id]),
            rank=rank,
        )
        for rank, token_id in enumerate(indexes)
    )


def assert_aligned_with_vllm_topk(
    score: FullVocabPlanScore,
    vllm_positions: Sequence[Sequence[TokenLogprob]],
) -> None:
    """Fail closed when vLLM top-k positions do not match this forced plan.

    Checks position count and the forced token id at each position (vLLM
    rank-0). Does not pad, invent, or reorder log-probabilities.
    """

    if len(score.forced_token_ids) != len(vllm_positions):
        raise PositionAlignmentError(
            "full-vocabulary and vLLM scored a different number of positions"
        )
    for index, (forced_id, position) in enumerate(
        zip(score.forced_token_ids, vllm_positions, strict=True)
    ):
        if not position:
            raise SupportAlignmentError(
                f"vLLM support is empty at position {index}"
            )
        if int(position[0].token_id) != int(forced_id):
            raise SupportAlignmentError(
                f"forced token id differs at position {index}"
            )


def kl_position_sample(
    production: FullVocabPlanScore,
    candidate: FullVocabPlanScore,
) -> KLPositionSample:
    """Build the sample type ``compare_plan_kl`` already scores."""

    _pair_aligned(production, candidate)
    return KLPositionSample(
        production_log_probabilities=production.position_log_probabilities,
        candidate_log_probabilities=candidate.position_log_probabilities,
    )


def compare_plan_kl_sample_document(
    pairs: Sequence[tuple[FullVocabPlanScore, FullVocabPlanScore]],
) -> dict[str, object]:
    """JSON document accepted by ``compare_plan_kl`` / ``_load_kl_samples``."""

    samples: list[dict[str, object]] = []
    for production, candidate in pairs:
        sample = kl_position_sample(production, candidate)
        samples.append(
            {
                "production": [
                    list(row) for row in sample.production_log_probabilities
                ],
                "candidate": [
                    list(row) for row in sample.candidate_log_probabilities
                ],
            }
        )
    return {"samples": samples}


def score_truncation_error(
    production: FullVocabPlanScore,
    candidate: FullVocabPlanScore,
    *,
    top_k: int,
    provenance: str = "supplied_sample",
) -> KLApproximationReport:
    """Record truncation error only through ``compare_plan_kl``."""

    sample = kl_position_sample(production, candidate)
    return compare_plan_kl(
        (sample,),
        top_k=top_k,
        provenance=provenance,
        vocabulary_size=production.vocabulary_size,
    )


def public_full_vocab_aggregate(
    score: FullVocabPlanScore,
    *,
    report: KLApproximationReport | None = None,
) -> dict[str, object]:
    """Public aggregate without plan text or token strings."""

    document: dict[str, object] = {
        "vocabulary_size": score.vocabulary_size,
        "position_count": score.position_count,
        "larger_k": score.larger_k,
        "default_top_k": DEFAULT_TOP_K,
        "input_prefix_fingerprint": score.input_prefix_fingerprint,
    }
    if report is not None:
        document["mean_signed_error"] = report.mean_signed_error
        document["mean_absolute_error"] = report.mean_absolute_error
        document["max_absolute_error"] = report.max_absolute_error
        document["top_k"] = report.top_k
        document["status"] = report.status
        document["approximation"] = report.approximation
        document["gpu_floor_measured"] = report.gpu_floor_measured
    return document


def write_local_kl_sample(
    path: Path,
    pairs: Sequence[tuple[FullVocabPlanScore, FullVocabPlanScore]],
    *,
    results_root: Path | None = None,
) -> None:
    """Write a local sample file for ``compare_plan_kl``. Refuses ``results/``."""

    root = results_root if results_root is not None else default_results_root()
    resolved = path.resolve()
    base = root.resolve()
    if resolved == base or base in resolved.parents:
        raise FullVocabError("refusing to write under results/")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            compare_plan_kl_sample_document(pairs),
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _pair_aligned(
    production: FullVocabPlanScore,
    candidate: FullVocabPlanScore,
) -> None:
    if production.vocabulary_size != candidate.vocabulary_size:
        raise FullVocabError("production and candidate vocabulary_size differ")
    if production.input_prefix_fingerprint != candidate.input_prefix_fingerprint:
        raise PositionAlignmentError(
            "production and candidate were not teacher-forced over the same "
            "input prefix"
        )
    if len(production.forced_token_ids) != len(candidate.forced_token_ids):
        raise PositionAlignmentError(
            "production and candidate scored a different number of positions"
        )
    if production.forced_token_ids != candidate.forced_token_ids:
        raise SupportAlignmentError(
            "production and candidate forced token ids differ"
        )


def _top_indexes(log_probabilities: Sequence[float], top_k: int) -> tuple[int, ...]:
    ranked = sorted(
        range(len(log_probabilities)),
        key=lambda index: (-log_probabilities[index], index),
    )
    return tuple(ranked[:top_k])


def _positive_int(value: object, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise FullVocabError(f"{name} must be a positive int")
    return value


@dataclass(frozen=True)
class TransformersBundle:
    """A loaded tokenizer and a next-token log-probability forward."""

    tokenizer: object
    forward: Callable[[Sequence[int]], Sequence[Sequence[float]]]
    vocabulary_size: int


@dataclass(frozen=True)
class FullVocabEvidence:
    """One full-vocabulary score bound to model, tokenizer, and prefix identity."""

    contract: ScoringContract
    forced_token_ids: tuple[int, ...]
    configuration_hash: str


@dataclass(frozen=True)
class BackendAgreementReport:
    """vLLM top-k versus the dense top-k slice. Not a truncation-error report.

    ``logprob_tolerance`` is a caller-supplied diagnostic. It is not a
    protocol choice.
    """

    logprob_tolerance: float
    position_count: int
    forced_token_mismatches: int
    support_mismatches: int
    logprob_mismatches: int
    max_abs_logprob_delta: float | None
    status: str


def render_plan_token_ids(
    tokenizer: object,
    messages: Sequence[Mapping[str, str]],
    plan_text: str,
    *,
    enable_thinking: bool,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Tokenize the vLLM prefix and the forced plan with the same template.

    The prefix uses ``add_generation_prompt=True``. The forced request
    appends the plan as the assistant message with
    ``add_generation_prompt=False``. The forced ids must start with the
    prefix ids. The returned continuation is every token after that
    boundary, including the template's end-of-turn tokens.
    """

    if plan_text == "":
        raise FullVocabError("plan_text is required for teacher forcing")
    prefix = _apply_template(
        tokenizer,
        list(messages),
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )
    forced = _apply_template(
        tokenizer,
        list(messages) + [{"role": "assistant", "content": plan_text}],
        add_generation_prompt=False,
        enable_thinking=enable_thinking,
    )
    if not prefix or forced[: len(prefix)] != prefix:
        raise FullVocabError(
            "forced prompt does not start with the plan prefix tokens"
        )
    continuation = tuple(forced[len(prefix) :])
    if not continuation:
        raise FullVocabError("forced plan produced no tokens after the prefix")
    return tuple(prefix), continuation


def log_probabilities_for_continuation(
    full_ids: Sequence[int],
    prefix_length: int,
    next_token_log_probabilities: Sequence[Sequence[float]],
    *,
    vocabulary_size: int,
) -> DenseFullVocabResult:
    """Slice next-token rows onto the forced continuation.

    ``next_token_log_probabilities[t]`` is the distribution of
    ``full_ids[t + 1]``. Rows must be normalized and ``vocabulary_size``
    wide. A short or shifted row fails closed.
    """

    vocabulary = _positive_int(vocabulary_size, "vocabulary_size")
    if (
        not isinstance(prefix_length, int)
        or isinstance(prefix_length, bool)
        or prefix_length < 1
        or prefix_length >= len(full_ids)
    ):
        raise FullVocabError("prefix_length does not leave a forced continuation")
    expected_rows = len(full_ids) - 1
    if len(next_token_log_probabilities) != expected_rows:
        raise FullVocabError("next-token rows do not cover the forced token ids")
    start = prefix_length - 1
    continuation = tuple(int(token_id) for token_id in full_ids[prefix_length:])
    rows = tuple(
        tuple(float(value) for value in row)
        for row in next_token_log_probabilities[start : start + len(continuation)]
    )
    if len(rows) != len(continuation):
        raise FullVocabError("continuation log-probabilities were truncated")
    for index, (token_id, row) in enumerate(zip(continuation, rows, strict=True)):
        if len(row) != vocabulary:
            raise FullVocabError(
                f"position {index} support length differs from vocabulary_size"
            )
        if token_id < 0 or token_id >= vocabulary:
            raise FullVocabError(
                f"forced token id out of vocabulary at position {index}"
            )
        if not _is_normalized(row):
            raise FullVocabError(f"position {index} is not a normalized distribution")
    return DenseFullVocabResult(
        vocabulary_size=vocabulary,
        forced_token_ids=continuation,
        position_log_probabilities=rows,
    )


class TransformersFullVocabScorer:
    """Teacher-force one frozen plan with the configuration's pinned checkpoint.

    ``device`` is required. Pass ``cpu`` or ``cuda`` only after the vLLM
    server has been stopped when the device is the shared GPU. The loader
    is injectable so tests do not download or load weights.
    """

    def __init__(
        self,
        configuration: RunConfiguration,
        *,
        device: str,
        loader: Callable[..., TransformersBundle] | None = None,
    ) -> None:
        if not isinstance(configuration, RunConfiguration):
            raise FullVocabError("scorer requires a run configuration")
        if not isinstance(device, str) or device == "":
            raise FullVocabError("device is required")
        self._configuration = configuration
        self._device = device
        self._loader = loader or load_transformers_bundle
        self._bundle: TransformersBundle | None = None

    def __call__(
        self,
        *,
        messages: Sequence[Mapping[str, str]],
        plan_text: str,
    ) -> DenseFullVocabResult:
        if self._bundle is None:
            self._bundle = self._loader(self._configuration, device=self._device)
        prefix, continuation = render_plan_token_ids(
            self._bundle.tokenizer,
            messages,
            plan_text,
            enable_thinking=self._configuration.agent.prompt.thinking_enabled,
        )
        full_ids = prefix + continuation
        rows = self._bundle.forward(full_ids)
        return log_probabilities_for_continuation(
            full_ids,
            len(prefix),
            rows,
            vocabulary_size=self._bundle.vocabulary_size,
        )


def load_transformers_bundle(
    configuration: RunConfiguration,
    *,
    device: str,
) -> TransformersBundle:
    """Load the configuration's model and tokenizer revisions. Not ``main``.

    Transformers and torch are imported here, not when this module loads.
    """

    if not isinstance(configuration, RunConfiguration):
        raise FullVocabError("scorer requires a run configuration")
    if not isinstance(device, str) or device == "":
        raise FullVocabError("device is required")
    model_repository = configuration.model.model.repository
    model_revision = configuration.model.model.revision
    tokenizer_repository = configuration.model.tokenizer.repository
    tokenizer_revision = configuration.model.tokenizer.revision
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as error:
        raise FullVocabError("transformers is not installed") from error
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_repository,
        revision=tokenizer_revision,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_repository,
        revision=model_revision,
    )
    model.eval()
    model.to(device)
    tokenizer_size = len(tokenizer)
    model_size = getattr(getattr(model, "config", None), "vocab_size", None)
    if not isinstance(model_size, int) or isinstance(model_size, bool):
        raise FullVocabError("model vocabulary size is missing")
    if tokenizer_size != model_size:
        raise FullVocabError("tokenizer and model vocabulary sizes differ")

    def forward(token_ids: Sequence[int]) -> list[list[float]]:
        tensor = torch.tensor([list(token_ids)], dtype=torch.long, device=device)
        with torch.inference_mode():
            logits = model(tensor).logits[0, :-1]
            if int(logits.shape[-1]) != tokenizer_size:
                raise FullVocabError("logit width differs from vocabulary size")
            return torch.log_softmax(logits, dim=-1).tolist()

    return TransformersBundle(
        tokenizer=tokenizer,
        forward=forward,
        vocabulary_size=tokenizer_size,
    )


def bind_full_vocab_evidence(
    score: FullVocabPlanScore,
    configuration: RunConfiguration,
) -> FullVocabEvidence:
    """Accept the score as full-vocabulary only after the fidelity proof."""

    if not isinstance(configuration, RunConfiguration):
        raise FullVocabError("evidence requires a run configuration")
    if len(score.position_log_probabilities) != len(score.forced_token_ids):
        raise PositionAlignmentError("forced positions and distributions differ")
    positions: list[ScoredPosition] = []
    for index, row in enumerate(score.position_log_probabilities):
        if len(row) != score.vocabulary_size:
            raise FidelityProofError("incomplete vocabulary support")
        positions.append(
            ScoredPosition(
                position=index,
                support_token_ids=tuple(range(score.vocabulary_size)),
                log_probabilities=row,
            )
        )
    contract = ScoringContract(
        model_repository=configuration.model.model.repository,
        model_revision=configuration.model.model.revision,
        tokenizer_repository=configuration.model.tokenizer.repository,
        tokenizer_revision=configuration.model.tokenizer.revision,
        input_prefix_fingerprint=score.input_prefix_fingerprint,
        positions=tuple(positions),
        fidelity="full",
        declared_vocabulary_size=score.vocabulary_size,
    )
    verify_fidelity_claim(contract)
    return FullVocabEvidence(
        contract=contract,
        forced_token_ids=tuple(score.forced_token_ids),
        configuration_hash=run_configuration_hash(configuration),
    )


def assert_full_vocab_comparable(
    reference: FullVocabEvidence,
    candidate: FullVocabEvidence,
) -> None:
    """Fail closed on tokenizer, prefix, position, support, or forced-token mismatch."""

    verify_scoring_contracts(reference.contract, candidate.contract)
    if len(reference.forced_token_ids) != len(candidate.forced_token_ids):
        raise PositionAlignmentError("forced position counts differ")
    if reference.forced_token_ids != candidate.forced_token_ids:
        raise SupportAlignmentError("forced token ids differ")


def measure_backend_agreement(
    score: FullVocabPlanScore,
    vllm_positions: Sequence[Sequence[TokenLogprob]],
    *,
    logprob_tolerance: float,
) -> BackendAgreementReport:
    """Compare vLLM top-k with the dense top-k slice at a caller tolerance.

    Position count must match or this fails closed. Forced-token, support,
    and log-probability disagreements are counted. They are not truncation
    error.
    """

    if (
        isinstance(logprob_tolerance, bool)
        or not isinstance(logprob_tolerance, (int, float))
        or not math.isfinite(float(logprob_tolerance))
        or float(logprob_tolerance) < 0.0
    ):
        raise FullVocabError("logprob_tolerance must be a nonnegative finite float")
    tolerance = float(logprob_tolerance)
    if len(score.forced_token_ids) != len(vllm_positions):
        raise PositionAlignmentError(
            "full-vocabulary and vLLM scored a different number of positions"
        )
    forced_mismatches = 0
    support_mismatches = 0
    logprob_mismatches = 0
    max_delta: float | None = None
    for index, (forced_id, position, row) in enumerate(
        zip(
            score.forced_token_ids,
            vllm_positions,
            score.position_log_probabilities,
            strict=True,
        )
    ):
        if not position:
            raise SupportAlignmentError(f"vLLM support is empty at position {index}")
        if int(position[0].token_id) != int(forced_id):
            forced_mismatches += 1
        dense_slice = top_k_slice(row, len(position))
        vllm_ids = tuple(int(item.token_id) for item in position)
        dense_ids = tuple(item.token_id for item in dense_slice)
        support_matches = set(vllm_ids) == set(dense_ids) and len(vllm_ids) == len(
            set(vllm_ids)
        )
        if not support_matches:
            support_mismatches += 1
        dense_by_id = {item.token_id: float(item.logprob) for item in dense_slice}
        vllm_by_id = {int(item.token_id): float(item.logprob) for item in position}
        position_disagrees = False
        for token_id in set(dense_by_id) & set(vllm_by_id):
            delta = abs(vllm_by_id[token_id] - dense_by_id[token_id])
            max_delta = delta if max_delta is None else max(max_delta, delta)
            if delta > tolerance:
                position_disagrees = True
        if position_disagrees:
            logprob_mismatches += 1
    disagreements = forced_mismatches + support_mismatches + logprob_mismatches
    return BackendAgreementReport(
        logprob_tolerance=tolerance,
        position_count=len(score.forced_token_ids),
        forced_token_mismatches=forced_mismatches,
        support_mismatches=support_mismatches,
        logprob_mismatches=logprob_mismatches,
        max_abs_logprob_delta=max_delta,
        status="agree" if disagreements == 0 else "disagree",
    )


def public_backend_agreement(report: BackendAgreementReport) -> dict[str, object]:
    """Public counts for one backend-agreement report. No token strings."""

    return {
        "logprob_tolerance": report.logprob_tolerance,
        "position_count": report.position_count,
        "forced_token_mismatches": report.forced_token_mismatches,
        "support_mismatches": report.support_mismatches,
        "logprob_mismatches": report.logprob_mismatches,
        "max_abs_logprob_delta": report.max_abs_logprob_delta,
        "status": report.status,
    }


def write_local_plan_inputs(
    path: Path,
    *,
    messages: Sequence[Mapping[str, str]],
    plan_text: str,
    configuration: RunConfiguration,
    results_root: Path | None = None,
) -> None:
    """Persist one frozen plan for a later Transformers pass. Refuses ``results/``."""

    if not isinstance(configuration, RunConfiguration):
        raise FullVocabError("plan inputs require a run configuration")
    root = results_root if results_root is not None else default_results_root()
    resolved = path.resolve()
    base = root.resolve()
    if resolved == base or base in resolved.parents:
        raise FullVocabError("refusing to write under results/")
    payload = {
        "visibility": "local",
        "model_repository": configuration.model.model.repository,
        "model_revision": configuration.model.model.revision,
        "tokenizer_repository": configuration.model.tokenizer.repository,
        "tokenizer_revision": configuration.model.tokenizer.revision,
        "configuration_hash": run_configuration_hash(configuration),
        "enable_thinking": configuration.agent.prompt.thinking_enabled,
        "input_prefix_fingerprint": fingerprint_messages(messages),
        "messages": [
            {"role": str(message["role"]), "content": str(message["content"])}
            for message in messages
        ],
        "plan_text": plan_text,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _apply_template(
    tokenizer: object,
    messages: list[dict[str, str]],
    *,
    add_generation_prompt: bool,
    enable_thinking: bool,
) -> list[int]:
    apply = getattr(tokenizer, "apply_chat_template", None)
    if not callable(apply):
        raise FullVocabError("tokenizer has no chat template")
    try:
        rendered = apply(
            messages,
            add_generation_prompt=add_generation_prompt,
            tokenize=True,
            enable_thinking=enable_thinking,
        )
    except TypeError as error:
        raise FullVocabError("chat template rejected enable_thinking") from error
    except Exception as error:
        raise FullVocabError("chat template failed") from error
    if hasattr(rendered, "input_ids"):
        rendered = rendered["input_ids"]
    if isinstance(rendered, tuple):
        rendered = list(rendered)
    if (
        isinstance(rendered, list)
        and rendered
        and isinstance(rendered[0], list)
    ):
        rendered = rendered[0]
    if not isinstance(rendered, list) or not all(
        isinstance(token, int) and not isinstance(token, bool) for token in rendered
    ):
        raise FullVocabError("tokenizer did not return token ids")
    return rendered
