from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Literal, Mapping, Sequence

from llm_behavior_ci.config import KL_FIDELITY_MODES
from llm_behavior_ci.stats.kl import next_token_kl

FidelityMode = Literal["full", "top_k"]

_FULL_VOCABULARY_NORMALIZATION_TOLERANCE = 1e-6


class ScoringError(ValueError):
    pass


class PositionAlignmentError(ScoringError):
    pass


class SupportAlignmentError(ScoringError):
    pass


class TokenizerMismatchError(ScoringError):
    pass


class FidelityProofError(ScoringError):
    pass


def _positive_int(value: object, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ScoringError(f"{name} must be a positive int")
    return value


def _nonempty_text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ScoringError(f"{name} must be a non-empty string")
    return value


@dataclass(frozen=True)
class DistributionScore:
    kind: Literal["full", "top_k"]
    position_kl_nats: tuple[float, ...]
    mean_kl_nats: float


@dataclass(frozen=True)
class ScoredPosition:
    """One teacher-forced scored token position.

    ``support_token_ids`` and ``log_probabilities`` are aligned pointwise
    and are the exact support the scorer returned at this position: every
    token id assigned nonzero probability mass in its response, in the
    order returned. An empty or duplicated support is not a valid
    position, since it cannot be renormalized or compared unambiguously.
    """

    position: int
    support_token_ids: tuple[int, ...]
    log_probabilities: tuple[float, ...]

    def __post_init__(self) -> None:
        if (
            not isinstance(self.position, int)
            or isinstance(self.position, bool)
            or self.position < 0
        ):
            raise ScoringError("position must be a nonnegative int")
        if len(self.support_token_ids) != len(self.log_probabilities):
            raise ScoringError(
                "support_token_ids and log_probabilities length differ"
                f" at position {self.position}"
            )
        if not self.support_token_ids:
            raise ScoringError(f"position {self.position} has empty support")
        if len(set(self.support_token_ids)) != len(self.support_token_ids):
            raise ScoringError(
                f"position {self.position} has duplicate support token ids"
            )


@dataclass(frozen=True)
class ScoringContract:
    """The one authoritative record of what was teacher-forced and scored.

    Carries model/revision identity, tokenizer identity, a fingerprint of
    the exact input prefix, the exact scored positions and their support,
    the returned log-probabilities, and the caller's fidelity claim.
    ``declared_vocabulary_size`` is the verified tokenizer's true
    vocabulary size; it is required to even construct a ``"full"``-fidelity
    contract, but constructing one is only a necessary precondition, not a
    proof — ``verify_fidelity_claim`` (or ``verify_scoring_contracts``)
    is what mechanically establishes or rejects the claim from the
    supplied positions. ``fidelity`` is never inferred here; it is always
    what the caller asserts, and every caller of this module is expected
    to route through ``verify_scoring_contracts`` before trusting it.

    ``canonical_json``/``content_hash`` give a stable, reproducible digest
    of exactly what distribution was compared, for downstream provenance.
    """

    model_repository: str
    model_revision: str
    tokenizer_repository: str
    tokenizer_revision: str
    input_prefix_fingerprint: str
    positions: tuple[ScoredPosition, ...]
    fidelity: FidelityMode
    declared_vocabulary_size: int | None

    def __post_init__(self) -> None:
        _nonempty_text(self.model_repository, "model_repository")
        _nonempty_text(self.model_revision, "model_revision")
        _nonempty_text(self.tokenizer_repository, "tokenizer_repository")
        _nonempty_text(self.tokenizer_revision, "tokenizer_revision")
        _nonempty_text(self.input_prefix_fingerprint, "input_prefix_fingerprint")
        if not isinstance(self.positions, tuple) or not self.positions:
            raise ScoringError("positions must be a non-empty tuple")
        for index, position in enumerate(self.positions):
            if not isinstance(position, ScoredPosition):
                raise ScoringError("positions entries must be ScoredPosition")
            if position.position != index:
                raise ScoringError(
                    "positions must be ordered and contiguous from zero"
                )
        if self.fidelity not in KL_FIDELITY_MODES:
            raise ScoringError("fidelity must be full or top_k")
        if self.declared_vocabulary_size is not None:
            _positive_int(self.declared_vocabulary_size, "declared_vocabulary_size")
        if self.fidelity == "full" and self.declared_vocabulary_size is None:
            raise ScoringError("full fidelity requires declared_vocabulary_size")

    def canonical_json(self) -> str:
        payload = {
            "model_repository": self.model_repository,
            "model_revision": self.model_revision,
            "tokenizer_repository": self.tokenizer_repository,
            "tokenizer_revision": self.tokenizer_revision,
            "input_prefix_fingerprint": self.input_prefix_fingerprint,
            "positions": [
                {
                    "position": position.position,
                    "support_token_ids": list(position.support_token_ids),
                    "log_probabilities": list(position.log_probabilities),
                }
                for position in self.positions
            ],
            "fidelity": self.fidelity,
            "declared_vocabulary_size": self.declared_vocabulary_size,
        }
        try:
            return json.dumps(
                payload,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            )
        except (TypeError, ValueError) as error:
            raise ScoringError("scoring contract is not finite JSON") from error

    def content_hash(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()


def fingerprint_messages(messages: Sequence[Mapping[str, str]]) -> str:
    """A stable digest of the exact message sequence teacher-forced as a prefix."""

    payload = [
        {"role": str(message["role"]), "content": str(message["content"])}
        for message in messages
    ]
    try:
        document = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise ScoringError("messages are not finite JSON") from error
    return hashlib.sha256(document.encode("utf-8")).hexdigest()


def _is_normalized(log_probabilities: Sequence[float]) -> bool:
    finite = [value for value in log_probabilities if math.isfinite(value)]
    if not finite:
        return False
    maximum = max(finite)
    total = math.fsum(math.exp(value - maximum) for value in finite)
    log_normalizer = maximum + math.log(total)
    return abs(log_normalizer) <= _FULL_VOCABULARY_NORMALIZATION_TOLERANCE


def establishes_full_vocabulary_coverage(contract: ScoringContract) -> bool:
    """True only when every position provably covers the declared vocabulary.

    This is the only mechanism by which ``"full"`` may be accepted: every
    position's support must be exactly the declared vocabulary's token ids
    (length match and identical id set) and must be normalized within
    tolerance. ``"full"`` is never inferred from configuration alone; a
    caller that has not supplied a matching, normalized, full-length
    support at every position gets ``False`` here regardless of what it
    claims.
    """

    if contract.declared_vocabulary_size is None:
        return False
    expected_ids = frozenset(range(contract.declared_vocabulary_size))
    for position in contract.positions:
        if len(position.support_token_ids) != contract.declared_vocabulary_size:
            return False
        if frozenset(position.support_token_ids) != expected_ids:
            return False
        if not _is_normalized(position.log_probabilities):
            return False
    return True


def verify_fidelity_claim(contract: ScoringContract) -> None:
    if contract.fidelity == "full" and not establishes_full_vocabulary_coverage(
        contract
    ):
        raise FidelityProofError(
            "fidelity is claimed full but the supplied positions do not "
            "establish full-vocabulary coverage at every position"
        )


def verify_tokenizer_compatibility(
    reference: ScoringContract, candidate: ScoringContract
) -> None:
    if (reference.tokenizer_repository, reference.tokenizer_revision) != (
        candidate.tokenizer_repository,
        candidate.tokenizer_revision,
    ):
        raise TokenizerMismatchError(
            "reference and candidate tokenizer identity differ"
        )


def verify_position_alignment(
    reference: ScoringContract, candidate: ScoringContract
) -> None:
    if reference.input_prefix_fingerprint != candidate.input_prefix_fingerprint:
        raise PositionAlignmentError(
            "reference and candidate were not teacher-forced over the same "
            "input prefix"
        )
    if len(reference.positions) != len(candidate.positions):
        raise PositionAlignmentError(
            "reference and candidate scored a different number of positions"
        )
    for reference_position, candidate_position in zip(
        reference.positions, candidate.positions
    ):
        if reference_position.position != candidate_position.position:
            raise PositionAlignmentError(
                "reference and candidate scored positions are not ordered "
                "identically"
            )


def verify_support_alignment(
    reference: ScoringContract, candidate: ScoringContract
) -> None:
    for reference_position, candidate_position in zip(
        reference.positions, candidate.positions
    ):
        if reference_position.support_token_ids != candidate_position.support_token_ids:
            raise SupportAlignmentError(
                "reference and candidate support differ at position "
                f"{reference_position.position}"
            )


def verify_scoring_contracts(
    reference: ScoringContract, candidate: ScoringContract
) -> None:
    """Run every check this module requires before a KL number is trusted.

    Order matters for a useful error message: tokenizer identity, then
    position alignment (same prefix, same position count and ordering),
    then per-position support alignment, then the fidelity proof for each
    side. Raises the first violated ``ScoringError`` subclass; raises
    nothing when every check passes.
    """

    if reference.fidelity != candidate.fidelity:
        raise ScoringError("reference and candidate fidelity modes differ")
    verify_tokenizer_compatibility(reference, candidate)
    verify_position_alignment(reference, candidate)
    verify_support_alignment(reference, candidate)
    verify_fidelity_claim(reference)
    verify_fidelity_claim(candidate)


def score_full(
    production: Sequence[Sequence[float]],
    candidate: Sequence[Sequence[float]],
) -> DistributionScore:
    """Score aligned full-vocabulary next-token log-probabilities.

    This is full-vocabulary KL only when the caller passes aligned
    full-vocabulary log-probabilities. A selected token log-probability is
    not a valid input. This function trusts its caller's arrays; it does
    not itself prove vocabulary coverage. ``score_scoring_contracts`` is
    the entry point that proves the claim before reaching here.
    """

    result = next_token_kl(production, candidate)
    return DistributionScore(
        kind="full",
        position_kl_nats=result.position_kl_nats,
        mean_kl_nats=result.mean_kl_nats,
    )


def score_top_k(
    production: Sequence[Sequence[float]],
    candidate: Sequence[Sequence[float]],
) -> DistributionScore:
    """Score truncated top-k next-token log-probabilities.

    This is a truncated approximation, not exact KL.
    """

    from llm_behavior_ci.stats import kl as kl_module

    result = kl_module.truncated_next_token_kl(production, candidate)
    if result.approximation != "top_k":
        raise ScoringError("truncated score is not labeled top_k")
    return DistributionScore(
        kind="top_k",
        position_kl_nats=result.position_kl_nats,
        mean_kl_nats=result.mean_kl_nats,
    )


def score_scoring_contracts(
    reference: ScoringContract, candidate: ScoringContract
) -> DistributionScore:
    """Verify, then score, one teacher-forced comparison end to end.

    This is the mechanically-enforced entry point: it never computes a KL
    number before ``verify_scoring_contracts`` has passed, and the
    ``DistributionScore.kind`` it returns is always the fidelity that was
    just proven, never a value taken on faith from the caller.
    """

    verify_scoring_contracts(reference, candidate)
    production = tuple(position.log_probabilities for position in reference.positions)
    comparison = tuple(position.log_probabilities for position in candidate.positions)
    if reference.fidelity == "full":
        return score_full(production, comparison)
    return score_top_k(production, comparison)
