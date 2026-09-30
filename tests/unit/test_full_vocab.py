from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

from llm_behavior_ci.experiments.validation import compare_plan_kl
from llm_behavior_ci.records import TokenLogprob
from llm_behavior_ci.config import RunConfiguration
from llm_behavior_ci.runtime.full_vocab import (
    DEFAULT_TOP_K,
    DenseFullVocabResult,
    FullVocabError,
    TransformersBundle,
    TransformersFullVocabScorer,
    assert_aligned_with_vllm_topk,
    assert_full_vocab_comparable,
    bind_full_vocab_evidence,
    compare_plan_kl_sample_document,
    kl_position_sample,
    log_probabilities_for_continuation,
    measure_backend_agreement,
    public_backend_agreement,
    public_full_vocab_aggregate,
    render_plan_token_ids,
    score_frozen_plan,
    score_truncation_error,
    top_k_slice,
    write_local_kl_sample,
    write_local_plan_inputs,
)
from llm_behavior_ci.runtime.scoring import (
    FidelityProofError,
    PositionAlignmentError,
    SupportAlignmentError,
    TokenizerMismatchError,
)
from llm_behavior_ci.stats.kl import next_token_kl, truncated_next_token_kl


def _messages() -> list[dict[str, str]]:
    return [
        {"role": "system", "content": "system text"},
        {"role": "user", "content": "solve the task"},
    ]


def _normalized_row(masses: tuple[float, ...]) -> tuple[float, ...]:
    total = sum(masses)
    return tuple(math.log(value / total) for value in masses)


def _fake_scorer(
    *,
    vocabulary_size: int,
    forced_token_ids: tuple[int, ...],
    rows: tuple[tuple[float, ...], ...],
):
    def scorer(*, messages: list[dict[str, str]], plan_text: str) -> DenseFullVocabResult:
        del messages, plan_text
        return DenseFullVocabResult(
            vocabulary_size=vocabulary_size,
            forced_token_ids=forced_token_ids,
            position_log_probabilities=rows,
        )

    return scorer


class FullVocabCollectorTests(unittest.TestCase):
    def test_top_k_slices_are_prefixes_of_the_same_full_distribution(self) -> None:
        self.assertEqual(DEFAULT_TOP_K, 20)
        vocab = 32
        masses = tuple(
            0.04 - (index * 0.001) if index < 20 else 0.001 for index in range(vocab)
        )
        row = _normalized_row(masses)
        score = score_frozen_plan(
            messages=_messages(),
            plan_text="plan step one",
            scorer=_fake_scorer(
                vocabulary_size=vocab,
                forced_token_ids=(3,),
                rows=(row,),
            ),
            larger_k=25,
        )
        self.assertEqual(score.vocabulary_size, vocab)
        self.assertEqual(len(score.top_k_default[0]), DEFAULT_TOP_K)
        self.assertEqual(len(score.top_k_larger[0]), 25)
        default_ids = tuple(item.token_id for item in score.top_k_default[0])
        larger_ids = tuple(item.token_id for item in score.top_k_larger[0])
        self.assertEqual(default_ids, larger_ids[:DEFAULT_TOP_K])
        for item in score.top_k_larger[0]:
            self.assertAlmostEqual(item.logprob, row[item.token_id], places=12)
        ranked = top_k_slice(row, 25)
        self.assertEqual(
            tuple(item.token_id for item in ranked),
            larger_ids,
        )

    def test_top_k_20_and_larger_share_ranking_on_tiny_vocab(self) -> None:
        vocab = 6
        row = _normalized_row((0.40, 0.25, 0.15, 0.10, 0.07, 0.03))
        score = score_frozen_plan(
            messages=_messages(),
            plan_text="frozen",
            scorer=_fake_scorer(
                vocabulary_size=vocab,
                forced_token_ids=(0, 2),
                rows=(row, row),
            ),
            larger_k=4,
            default_k=3,
        )
        for default_pos, larger_pos in zip(
            score.top_k_default, score.top_k_larger, strict=True
        ):
            self.assertEqual(
                tuple(item.token_id for item in default_pos),
                tuple(item.token_id for item in larger_pos)[:3],
            )

    def test_mismatched_position_length_fails_closed(self) -> None:
        row = _normalized_row((0.5, 0.3, 0.2, 1e-9))
        score = score_frozen_plan(
            messages=_messages(),
            plan_text="plan",
            scorer=_fake_scorer(
                vocabulary_size=4,
                forced_token_ids=(1, 2),
                rows=(row, row),
            ),
            larger_k=3,
            default_k=2,
        )
        short_vllm = (
            (TokenLogprob(token_id=1, logprob=row[1], rank=0),),
        )
        with self.assertRaises(PositionAlignmentError):
            assert_aligned_with_vllm_topk(score, short_vllm)

    def test_mismatched_forced_token_id_fails_closed(self) -> None:
        row = _normalized_row((0.5, 0.3, 0.2))
        score = score_frozen_plan(
            messages=_messages(),
            plan_text="plan",
            scorer=_fake_scorer(
                vocabulary_size=3,
                forced_token_ids=(1,),
                rows=(row,),
            ),
            larger_k=2,
            default_k=1,
        )
        wrong = (
            (
                TokenLogprob(token_id=0, logprob=row[0], rank=0),
                TokenLogprob(token_id=1, logprob=row[1], rank=1),
            ),
        )
        with self.assertRaises(SupportAlignmentError):
            assert_aligned_with_vllm_topk(score, wrong)

    def test_aligned_vllm_topk_passes(self) -> None:
        row = _normalized_row((0.5, 0.3, 0.2))
        score = score_frozen_plan(
            messages=_messages(),
            plan_text="plan",
            scorer=_fake_scorer(
                vocabulary_size=3,
                forced_token_ids=(1,),
                rows=(row,),
            ),
            larger_k=2,
            default_k=1,
        )
        matched = (
            (
                TokenLogprob(token_id=1, logprob=row[1], rank=0),
                TokenLogprob(token_id=0, logprob=row[0], rank=1),
            ),
        )
        assert_aligned_with_vllm_topk(score, matched)

    def test_truncation_error_matches_hand_computed_compare_plan_kl(self) -> None:
        production = _normalized_row((0.7, 0.2, 0.1))
        candidate = _normalized_row((0.2, 0.1, 0.7))
        messages = _messages()
        production_score = score_frozen_plan(
            messages=messages,
            plan_text="shared plan",
            scorer=_fake_scorer(
                vocabulary_size=3,
                forced_token_ids=(0,),
                rows=(production,),
            ),
            larger_k=2,
            default_k=1,
        )
        candidate_score = score_frozen_plan(
            messages=messages,
            plan_text="shared plan",
            scorer=_fake_scorer(
                vocabulary_size=3,
                forced_token_ids=(0,),
                rows=(candidate,),
            ),
            larger_k=2,
            default_k=1,
        )
        report = score_truncation_error(
            production_score,
            candidate_score,
            top_k=2,
            provenance="supplied_sample",
        )
        direct = compare_plan_kl(
            (kl_position_sample(production_score, candidate_score),),
            top_k=2,
            provenance="supplied_sample",
            vocabulary_size=3,
        )
        full = next_token_kl((production,), (candidate,)).mean_kl_nats
        truncated = truncated_next_token_kl(
            ((production[0], production[1]),),
            ((candidate[0], candidate[1]),),
            vocabulary_size=3,
        ).mean_kl_nats
        expected = full - truncated
        self.assertEqual(report.status, "passed")
        self.assertEqual(report.vocabulary_size, 3)
        self.assertAlmostEqual(report.mean_signed_error or 0.0, expected, places=8)
        self.assertAlmostEqual(
            report.mean_absolute_error or 0.0, abs(expected), places=8
        )
        self.assertAlmostEqual(
            report.max_absolute_error or 0.0, abs(expected), places=8
        )
        self.assertAlmostEqual(
            report.mean_signed_error or 0.0,
            direct.mean_signed_error or 0.0,
            places=12,
        )
        self.assertFalse(report.gpu_floor_measured)

    def test_vocabulary_size_is_reported(self) -> None:
        row = _normalized_row((0.25, 0.25, 0.25, 0.25))
        score = score_frozen_plan(
            messages=_messages(),
            plan_text="plan",
            scorer=_fake_scorer(
                vocabulary_size=4,
                forced_token_ids=(2,),
                rows=(row,),
            ),
            larger_k=3,
            default_k=2,
        )
        self.assertEqual(score.vocabulary_size, 4)
        report = score_truncation_error(
            score,
            score,
            top_k=2,
            provenance="supplied_sample",
        )
        aggregate = public_full_vocab_aggregate(score, report=report)
        self.assertEqual(aggregate["vocabulary_size"], 4)
        self.assertEqual(report.vocabulary_size, 4)

    def test_public_aggregate_omits_plan_text_and_token_strings(self) -> None:
        row = _normalized_row((0.6, 0.4))
        score = score_frozen_plan(
            messages=_messages(),
            plan_text="secret plan text",
            scorer=_fake_scorer(
                vocabulary_size=2,
                forced_token_ids=(0,),
                rows=(row,),
            ),
            larger_k=2,
            default_k=1,
        )
        report = score_truncation_error(
            score, score, top_k=1, provenance="supplied_sample"
        )
        payload = json.dumps(public_full_vocab_aggregate(score, report=report))
        self.assertNotIn("secret plan text", payload)
        self.assertNotIn("solve the task", payload)
        self.assertNotIn("token_id:", payload)
        self.assertNotIn("forced_token_ids", payload)

    def test_sample_document_is_compare_plan_kl_shape(self) -> None:
        production = _normalized_row((0.7, 0.2, 0.1))
        candidate = _normalized_row((0.2, 0.1, 0.7))
        messages = _messages()
        left = score_frozen_plan(
            messages=messages,
            plan_text="plan",
            scorer=_fake_scorer(
                vocabulary_size=3,
                forced_token_ids=(0,),
                rows=(production,),
            ),
            larger_k=2,
            default_k=1,
        )
        right = score_frozen_plan(
            messages=messages,
            plan_text="plan",
            scorer=_fake_scorer(
                vocabulary_size=3,
                forced_token_ids=(0,),
                rows=(candidate,),
            ),
            larger_k=2,
            default_k=1,
        )
        document = compare_plan_kl_sample_document(((left, right),))
        self.assertIn("samples", document)
        self.assertEqual(len(document["samples"]), 1)
        item = document["samples"][0]
        assert isinstance(item, dict)
        self.assertEqual(item["production"], [list(production)])
        self.assertEqual(item["candidate"], [list(candidate)])

    def test_write_local_sample_refuses_results(self) -> None:
        row = _normalized_row((0.5, 0.5))
        score = score_frozen_plan(
            messages=_messages(),
            plan_text="plan",
            scorer=_fake_scorer(
                vocabulary_size=2,
                forced_token_ids=(0,),
                rows=(row,),
            ),
            larger_k=2,
            default_k=1,
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "results"
            root.mkdir()
            with self.assertRaises(FullVocabError):
                write_local_kl_sample(
                    root / "sample.json",
                    ((score, score),),
                    results_root=root,
                )

    def test_empty_plan_text_fails_closed(self) -> None:
        with self.assertRaises(FullVocabError):
            score_frozen_plan(
                messages=_messages(),
                plan_text="",
                scorer=_fake_scorer(
                    vocabulary_size=2,
                    forced_token_ids=(0,),
                    rows=(_normalized_row((0.5, 0.5)),),
                ),
                larger_k=2,
                default_k=1,
            )

    def test_module_does_not_import_transformers(self) -> None:
        import llm_behavior_ci.runtime.full_vocab as module
        import sys

        self.assertNotIn("transformers", sys.modules)
        self.assertFalse(hasattr(module, "transformers"))
        source = Path(module.__file__).read_text(encoding="utf-8")
        head, loader = source.split("def load_transformers_bundle", maxsplit=1)
        self.assertNotIn("import transformers", head)
        self.assertNotIn("from transformers", head)
        self.assertIn("from transformers import AutoModelForCausalLM, AutoTokenizer", loader)
        self.assertIn("revision=model_revision", loader)
        self.assertIn("revision=tokenizer_revision", loader)
        self.assertIn("inference_mode", loader)


def _configuration(tokenizer_revision: str = "b" * 40) -> RunConfiguration:
    root = Path(__file__).resolve().parents[2]
    document = json.loads(
        (root / "configs/models/qwen3_4b_production.json").read_text(encoding="utf-8")
    )
    document["model"]["tokenizer"]["revision"] = tokenizer_revision
    document["task"] = {
        "appworld_version": "0.1.3.post1",
        "split": "train",
        "selection_rule": "deterministic_sample",
        "selection_seed": 17,
        "task_count": 1,
        "task_set_hash": "c" * 64,
    }
    document["run_seed"] = 17
    document["git_commit"] = "a" * 40
    document["protocol_hash"] = None
    return RunConfiguration.from_dict(document)


def _normalized_pair() -> tuple[float, float]:
    return (math.log(0.5), math.log(0.5))


class _Tokenizer:
    def __init__(self, *, mismatch: bool = False) -> None:
        self.mismatch = mismatch

    def apply_chat_template(
        self,
        messages,
        *,
        add_generation_prompt: bool,
        tokenize: bool,
        enable_thinking: bool,
    ):
        del tokenize, enable_thinking
        if add_generation_prompt:
            return [10, 11]
        if self.mismatch or messages[-1]["content"] != "PLAN":
            return [99]
        return [10, 11, 0, 1]


class IdentityAndBackendTests(unittest.TestCase):
    def test_continuation_uses_rows_after_the_prefix(self) -> None:
        kept = (math.log(0.8), math.log(0.2))
        ignored = (math.log(0.1), math.log(0.9))
        result = log_probabilities_for_continuation(
            [10, 11, 0, 1],
            2,
            (ignored, kept, kept),
            vocabulary_size=2,
        )
        self.assertEqual(result.forced_token_ids, (0, 1))
        self.assertEqual(result.position_log_probabilities, (kept, kept))

    def test_prefix_mismatch_and_unnormalized_rows_fail_closed(self) -> None:
        with self.assertRaises(FullVocabError):
            render_plan_token_ids(
                _Tokenizer(mismatch=True),
                [{"role": "user", "content": "task"}],
                "PLAN",
                enable_thinking=False,
            )
        with self.assertRaises(FullVocabError):
            log_probabilities_for_continuation(
                [1, 2],
                1,
                ((0.0, 0.0),),
                vocabulary_size=2,
            )

    def test_transformers_scorer_uses_injected_bundle(self) -> None:
        kept = _normalized_pair()
        calls: list[tuple[int, ...]] = []

        def forward(token_ids):
            calls.append(tuple(token_ids))
            return (kept, kept, kept)

        scorer = TransformersFullVocabScorer(
            _configuration(),
            device="cpu",
            loader=lambda configuration, device: TransformersBundle(
                tokenizer=_Tokenizer(),
                forward=forward,
                vocabulary_size=2,
            ),
        )
        raw = scorer(
            messages=[{"role": "user", "content": "task"}],
            plan_text="PLAN",
        )
        self.assertEqual(calls, [(10, 11, 0, 1)])
        self.assertEqual(raw.forced_token_ids, (0, 1))
        score = score_frozen_plan(
            messages=[{"role": "user", "content": "task"}],
            plan_text="PLAN",
            scorer=scorer,
            larger_k=2,
            default_k=1,
        )
        evidence = bind_full_vocab_evidence(score, _configuration())
        self.assertEqual(evidence.contract.fidelity, "full")
        self.assertEqual(evidence.contract.declared_vocabulary_size, 2)
        self.assertEqual(
            evidence.contract.model_revision,
            _configuration().model.model.revision,
        )
        assert_full_vocab_comparable(evidence, evidence)

    def test_full_vocabulary_comparison_fails_closed(self) -> None:
        def score_for(revision: str, token_id: int, fingerprint_plan: str):
            row = _normalized_pair()

            def scorer(*, messages, plan_text):
                del messages, plan_text
                return DenseFullVocabResult(
                    vocabulary_size=2,
                    forced_token_ids=(token_id,),
                    position_log_probabilities=(row,),
                )

            scored = score_frozen_plan(
                messages=[{"role": "user", "content": fingerprint_plan}],
                plan_text="PLAN",
                scorer=scorer,
                larger_k=2,
                default_k=1,
            )
            return bind_full_vocab_evidence(scored, _configuration(revision))

        left = score_for("b" * 40, 0, "same")
        with self.assertRaises(TokenizerMismatchError):
            assert_full_vocab_comparable(left, score_for("c" * 40, 0, "same"))
        with self.assertRaises(PositionAlignmentError):
            assert_full_vocab_comparable(left, score_for("b" * 40, 0, "other"))
        with self.assertRaises(SupportAlignmentError):
            assert_full_vocab_comparable(left, score_for("b" * 40, 1, "same"))
        with self.assertRaises(FidelityProofError):
            bind_full_vocab_evidence(
                score_frozen_plan(
                    messages=[{"role": "user", "content": "same"}],
                    plan_text="PLAN",
                    scorer=lambda messages, plan_text: DenseFullVocabResult(
                        vocabulary_size=2,
                        forced_token_ids=(0,),
                        position_log_probabilities=((0.0, 0.0),),
                    ),
                    larger_k=2,
                    default_k=1,
                ),
                _configuration(),
            )

    def test_backend_agreement_is_separate_from_truncation(self) -> None:
        row = (math.log(0.7), math.log(0.3))
        score = score_frozen_plan(
            messages=[{"role": "user", "content": "task"}],
            plan_text="PLAN",
            scorer=lambda messages, plan_text: DenseFullVocabResult(
                vocabulary_size=2,
                forced_token_ids=(0,),
                position_log_probabilities=(row,),
            ),
            larger_k=2,
            default_k=1,
        )
        agreed = measure_backend_agreement(
            score,
            ((TokenLogprob(token_id=0, logprob=row[0], rank=0),),),
            logprob_tolerance=1e-6,
        )
        self.assertEqual(agreed.status, "agree")
        self.assertEqual(agreed.support_mismatches, 0)
        drifted = measure_backend_agreement(
            score,
            (
                (
                    TokenLogprob(token_id=0, logprob=row[0] - 0.2, rank=0),
                    TokenLogprob(token_id=1, logprob=row[1], rank=1),
                ),
            ),
            logprob_tolerance=1e-6,
        )
        self.assertEqual(drifted.status, "disagree")
        self.assertEqual(drifted.forced_token_mismatches, 0)
        self.assertGreater(drifted.logprob_mismatches, 0)
        public = public_backend_agreement(drifted)
        self.assertNotIn("PLAN", json.dumps(public))
        with self.assertRaises(PositionAlignmentError):
            measure_backend_agreement(score, (), logprob_tolerance=1e-6)

    def test_plan_inputs_stay_out_of_results(self) -> None:
        configuration = _configuration()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "results"
            with self.assertRaises(FullVocabError):
                write_local_plan_inputs(
                    root / "plan.json",
                    messages=[{"role": "user", "content": "secret"}],
                    plan_text="PLAN",
                    configuration=configuration,
                    results_root=root,
                )


if __name__ == "__main__":
    unittest.main()
