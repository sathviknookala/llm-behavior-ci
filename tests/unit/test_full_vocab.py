from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

from llm_behavior_ci.experiments.validation import compare_plan_kl
from llm_behavior_ci.records import TokenLogprob
from llm_behavior_ci.runtime.full_vocab import (
    DEFAULT_TOP_K,
    DenseFullVocabResult,
    FullVocabError,
    assert_aligned_with_vllm_topk,
    compare_plan_kl_sample_document,
    kl_position_sample,
    public_full_vocab_aggregate,
    score_frozen_plan,
    score_truncation_error,
    top_k_slice,
    write_local_kl_sample,
)
from llm_behavior_ci.runtime.scoring import (
    PositionAlignmentError,
    SupportAlignmentError,
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
        self.assertNotIn("import transformers", source)
        self.assertNotIn("from transformers", source)


if __name__ == "__main__":
    unittest.main()
