from __future__ import annotations

import math
import unittest

from llm_behavior_ci.runtime.scoring import (
    DistributionScore,
    FidelityProofError,
    PositionAlignmentError,
    ScoredPosition,
    ScoringContract,
    ScoringError,
    SupportAlignmentError,
    TokenizerMismatchError,
    establishes_full_vocabulary_coverage,
    fingerprint_messages,
    score_scoring_contracts,
    verify_fidelity_claim,
    verify_scoring_contracts,
)

_MODEL_A = ("Qwen/Qwen3-4B", "0123456789abcdef0123456789abcdef01234567")
_MODEL_B = ("Qwen/Qwen3-4B", "fedcba9876543210fedcba9876543210fedcba98")
_TOKENIZER = ("Qwen/Qwen3-4B-tokenizer", "abcdefabcdefabcdefabcdefabcdefabcdefabcd")
_OTHER_TOKENIZER = ("Other/tokenizer", "1111111111111111111111111111111111111111")


def _messages() -> list[dict[str, str]]:
    return [
        {"role": "system", "content": "system text"},
        {"role": "user", "content": "solve the task"},
    ]


def _fingerprint() -> str:
    return fingerprint_messages(_messages())


def _full_position(index: int, probabilities: tuple[float, float]) -> ScoredPosition:
    return ScoredPosition(
        position=index,
        support_token_ids=(0, 1),
        log_probabilities=(math.log(probabilities[0]), math.log(probabilities[1])),
    )


def _top_k_position(index: int, token_ids: tuple[int, int], logprobs: tuple[float, float]) -> ScoredPosition:
    return ScoredPosition(
        position=index,
        support_token_ids=token_ids,
        log_probabilities=logprobs,
    )


def _contract(
    *,
    model: tuple[str, str] = _MODEL_A,
    tokenizer: tuple[str, str] = _TOKENIZER,
    fingerprint: str | None = None,
    positions: tuple[ScoredPosition, ...],
    fidelity: str = "full",
    declared_vocabulary_size: int | None = 2,
) -> ScoringContract:
    return ScoringContract(
        model_repository=model[0],
        model_revision=model[1],
        tokenizer_repository=tokenizer[0],
        tokenizer_revision=tokenizer[1],
        input_prefix_fingerprint=fingerprint or _fingerprint(),
        positions=positions,
        fidelity=fidelity,
        declared_vocabulary_size=declared_vocabulary_size,
    )


class ScoredPositionTests(unittest.TestCase):
    def test_empty_support_rejected(self) -> None:
        with self.assertRaises(ScoringError):
            ScoredPosition(position=0, support_token_ids=(), log_probabilities=())

    def test_mismatched_lengths_rejected(self) -> None:
        with self.assertRaises(ScoringError):
            ScoredPosition(position=0, support_token_ids=(0, 1), log_probabilities=(0.0,))

    def test_duplicate_support_rejected(self) -> None:
        with self.assertRaises(ScoringError):
            ScoredPosition(
                position=0,
                support_token_ids=(1, 1),
                log_probabilities=(math.log(0.5), math.log(0.5)),
            )


class ScoringContractTests(unittest.TestCase):
    def test_full_requires_vocabulary_size(self) -> None:
        with self.assertRaises(ScoringError):
            _contract(
                positions=(_full_position(0, (0.5, 0.5)),),
                fidelity="full",
                declared_vocabulary_size=None,
            )

    def test_positions_must_be_contiguous_from_zero(self) -> None:
        bad_position = ScoredPosition(
            position=1,
            support_token_ids=(0, 1),
            log_probabilities=(math.log(0.5), math.log(0.5)),
        )
        with self.assertRaises(ScoringError):
            _contract(positions=(bad_position,))

    def test_canonical_json_is_stable_and_hashable(self) -> None:
        contract = _contract(positions=(_full_position(0, (0.5, 0.5)),))
        first = contract.content_hash()
        second = contract.content_hash()
        self.assertEqual(first, second)
        self.assertEqual(len(first), 64)
        rebuilt = _contract(positions=(_full_position(0, (0.5, 0.5)),))
        self.assertEqual(first, rebuilt.content_hash())

    def test_canonical_json_changes_with_content(self) -> None:
        base = _contract(positions=(_full_position(0, (0.5, 0.5)),))
        shifted = _contract(positions=(_full_position(0, (0.25, 0.75)),))
        self.assertNotEqual(base.content_hash(), shifted.content_hash())


class FingerprintMessagesTests(unittest.TestCase):
    def test_deterministic(self) -> None:
        self.assertEqual(fingerprint_messages(_messages()), fingerprint_messages(_messages()))

    def test_differs_on_content_change(self) -> None:
        other = [{"role": "system", "content": "different"}]
        self.assertNotEqual(fingerprint_messages(_messages()), fingerprint_messages(other))


class FullVocabularyCoverageTests(unittest.TestCase):
    def test_correct_full_support_is_accepted(self) -> None:
        contract = _contract(positions=(_full_position(0, (0.5, 0.5)),))
        self.assertTrue(establishes_full_vocabulary_coverage(contract))
        verify_fidelity_claim(contract)

    def test_false_full_claim_wrong_length_is_rejected(self) -> None:
        position = _top_k_position(0, (0, 1, 2), (math.log(0.2), math.log(0.3), math.log(0.5)))
        contract = _contract(
            positions=(position,),
            fidelity="full",
            declared_vocabulary_size=2,
        )
        self.assertFalse(establishes_full_vocabulary_coverage(contract))
        with self.assertRaises(FidelityProofError):
            verify_fidelity_claim(contract)

    def test_false_full_claim_wrong_support_set_is_rejected(self) -> None:
        position = _top_k_position(0, (5, 9), (math.log(0.5), math.log(0.5)))
        contract = _contract(positions=(position,), fidelity="full", declared_vocabulary_size=2)
        self.assertFalse(establishes_full_vocabulary_coverage(contract))
        with self.assertRaises(FidelityProofError):
            verify_fidelity_claim(contract)

    def test_false_full_claim_not_normalized_is_rejected(self) -> None:
        position = ScoredPosition(
            position=0,
            support_token_ids=(0, 1),
            log_probabilities=(math.log(0.5), math.log(0.3)),
        )
        contract = _contract(positions=(position,), fidelity="full", declared_vocabulary_size=2)
        self.assertFalse(establishes_full_vocabulary_coverage(contract))
        with self.assertRaises(FidelityProofError):
            verify_fidelity_claim(contract)

    def test_full_is_never_inferred_from_configuration_alone(self) -> None:
        top_k_shaped = _top_k_position(0, (7, 9), (math.log(0.4), math.log(0.6)))
        contract = _contract(
            positions=(top_k_shaped,),
            fidelity="full",
            declared_vocabulary_size=151936,
        )
        self.assertFalse(establishes_full_vocabulary_coverage(contract))
        with self.assertRaises(FidelityProofError):
            verify_fidelity_claim(contract)


class VerifyScoringContractsTests(unittest.TestCase):
    def test_identical_full_distributions_score_zero(self) -> None:
        reference = _contract(positions=(_full_position(0, (0.5, 0.5)),))
        candidate = _contract(model=_MODEL_B, positions=(_full_position(0, (0.5, 0.5)),))
        score = score_scoring_contracts(reference, candidate)
        self.assertIsInstance(score, DistributionScore)
        self.assertEqual(score.kind, "full")
        self.assertAlmostEqual(score.mean_kl_nats, 0.0, delta=1e-9)

    def test_shifted_full_distributions_score_positive(self) -> None:
        reference = _contract(positions=(_full_position(0, (0.5, 0.5)),))
        candidate = _contract(model=_MODEL_B, positions=(_full_position(0, (0.1, 0.9)),))
        score = score_scoring_contracts(reference, candidate)
        self.assertEqual(score.kind, "full")
        self.assertGreater(score.mean_kl_nats, 0.0)

    def test_top_k_distributions_are_labeled_top_k(self) -> None:
        reference = _contract(
            positions=(_top_k_position(0, (7, 9), (math.log(0.4), math.log(0.6))),),
            fidelity="top_k",
            declared_vocabulary_size=None,
        )
        candidate = _contract(
            model=_MODEL_B,
            positions=(_top_k_position(0, (7, 9), (math.log(0.4), math.log(0.6))),),
            fidelity="top_k",
            declared_vocabulary_size=None,
        )
        score = score_scoring_contracts(reference, candidate)
        self.assertEqual(score.kind, "top_k")
        self.assertAlmostEqual(score.mean_kl_nats, 0.0, delta=1e-9)

    def test_mismatched_positions_raise(self) -> None:
        reference = _contract(
            positions=(
                _full_position(0, (0.5, 0.5)),
                _full_position(1, (0.5, 0.5)),
            )
        )
        candidate = _contract(model=_MODEL_B, positions=(_full_position(0, (0.5, 0.5)),))
        with self.assertRaises(PositionAlignmentError):
            verify_scoring_contracts(reference, candidate)
        with self.assertRaises(PositionAlignmentError):
            score_scoring_contracts(reference, candidate)

    def test_different_prefix_fingerprint_raises_position_alignment_error(self) -> None:
        reference = _contract(
            positions=(_full_position(0, (0.5, 0.5)),),
            fingerprint="a" * 64,
        )
        candidate = _contract(
            model=_MODEL_B,
            positions=(_full_position(0, (0.5, 0.5)),),
            fingerprint="b" * 64,
        )
        with self.assertRaises(PositionAlignmentError):
            verify_scoring_contracts(reference, candidate)

    def test_mismatched_support_raises(self) -> None:
        reference = _contract(
            positions=(_top_k_position(0, (7, 9), (math.log(0.4), math.log(0.6))),),
            fidelity="top_k",
            declared_vocabulary_size=None,
        )
        candidate = _contract(
            model=_MODEL_B,
            positions=(_top_k_position(0, (11, 13), (math.log(0.4), math.log(0.6))),),
            fidelity="top_k",
            declared_vocabulary_size=None,
        )
        with self.assertRaises(SupportAlignmentError):
            verify_scoring_contracts(reference, candidate)
        with self.assertRaises(SupportAlignmentError):
            score_scoring_contracts(reference, candidate)

    def test_mismatched_tokenizer_identity_raises(self) -> None:
        reference = _contract(
            tokenizer=_TOKENIZER,
            positions=(_full_position(0, (0.5, 0.5)),),
        )
        candidate = _contract(
            model=_MODEL_B,
            tokenizer=_OTHER_TOKENIZER,
            positions=(_full_position(0, (0.5, 0.5)),),
        )
        with self.assertRaises(TokenizerMismatchError):
            verify_scoring_contracts(reference, candidate)
        with self.assertRaises(TokenizerMismatchError):
            score_scoring_contracts(reference, candidate)

    def test_false_full_claim_is_rejected_before_scoring(self) -> None:
        reference = _contract(
            positions=(_top_k_position(0, (7, 9), (math.log(0.4), math.log(0.6))),),
            fidelity="full",
            declared_vocabulary_size=151936,
        )
        candidate = _contract(
            model=_MODEL_B,
            positions=(_top_k_position(0, (7, 9), (math.log(0.3), math.log(0.7))),),
            fidelity="full",
            declared_vocabulary_size=151936,
        )
        with self.assertRaises(FidelityProofError):
            verify_scoring_contracts(reference, candidate)
        with self.assertRaises(FidelityProofError):
            score_scoring_contracts(reference, candidate)

    def test_mismatched_fidelity_modes_raise(self) -> None:
        reference = _contract(positions=(_full_position(0, (0.5, 0.5)),), fidelity="full")
        candidate = _contract(
            model=_MODEL_B,
            positions=(_top_k_position(0, (0, 1), (math.log(0.5), math.log(0.5))),),
            fidelity="top_k",
            declared_vocabulary_size=None,
        )
        with self.assertRaises(ScoringError):
            verify_scoring_contracts(reference, candidate)


if __name__ == "__main__":
    unittest.main()
