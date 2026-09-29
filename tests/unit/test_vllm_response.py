from __future__ import annotations

import json
import unittest
from unittest.mock import patch

from llm_behavior_ci.config import RunConfiguration
from llm_behavior_ci.records import TokenLogprob
from llm_behavior_ci.runtime.agent import (
    SmolagentsVLLMAgent,
    _parse_prompt_logprobs,
    parse_logprobs,
)
from llm_behavior_ci.runtime.appworld import TaskContext
from llm_behavior_ci.runtime.episode import RuntimeUnavailable


def _payload() -> dict[str, object]:
    return {
        "model": {
            "model": {
                "repository": "Qwen/Qwen3-4B",
                "revision": "0123456789abcdef0123456789abcdef01234567",
            },
            "tokenizer": {
                "repository": "Qwen/Qwen3-4B",
                "revision": "fedcba9876543210fedcba9876543210fedcba98",
            },
            "quantization": {"method": "none"},
            "vllm_version": "0.30.0",
            "serving": {
                "dtype": "bfloat16",
                "max_model_len": 8192,
                "gpu_memory_utilization": 0.9,
                "max_num_seqs": 16,
                "max_num_batched_tokens": 8192,
                "kv_cache_dtype": "bfloat16",
                "enable_prefix_caching": False,
                "enable_chunked_prefill": False,
                "enforce_eager": False,
                "tensor_parallel_size": 1,
                "max_logprobs": 20,
                "batch_invariant": False,
                "sampler_backend": "native",
            },
        },
        "agent": {
            "smolagents_version": "1.22.0",
            "action_interface": "code",
            "prompt": {
                "prompt_version": "prompt-v1",
                "plan_format_version": "plan-v1",
                "thinking_enabled": False,
            },
            "step_limit": 40,
            "sampling": {
                "temperature": 0.0,
                "top_p": 1.0,
                "top_k": 20,
                "min_p": 0.0,
                "seed": 17,
                "max_tokens": 16,
            },
        },
        "task": {
            "appworld_version": "0.1.3.post1",
            "split": "train",
            "selection_rule": "deterministic_sample",
            "selection_seed": 20260926,
            "task_count": 50,
            "task_set_hash": "c" * 64,
        },
        "run_seed": 7,
        "git_commit": "a" * 40,
        "protocol_hash": "e" * 64,
    }


def _config() -> RunConfiguration:
    return RunConfiguration.from_dict(_payload())


def _context() -> TaskContext:
    return TaskContext(
        task_id="synthetic",
        instruction="Reply with the single word ping.",
        api_documentation="",
    )


def _live_shaped_choice() -> dict[str, object]:
    return {
        "token_ids": [9989, 151645],
        "logprobs": {
            "content": [
                {
                    "token": "token_id:9989",
                    "bytes": [116, 111, 107, 101, 110, 95, 105, 100, 58, 57, 57, 56, 57],
                    "logprob": -0.029807694256305695,
                    "top_logprobs": [
                        {
                            "token": "token_id:9989",
                            "bytes": [
                                116,
                                111,
                                107,
                                101,
                                110,
                                95,
                                105,
                                100,
                                58,
                                57,
                                57,
                                56,
                                57,
                            ],
                            "logprob": -0.029807694256305695,
                        },
                        {
                            "token": "token_id:69883",
                            "bytes": [1],
                            "logprob": -4.5,
                        },
                        {
                            "token": "token_id:71661",
                            "bytes": [2],
                            "logprob": -5.0,
                        },
                    ],
                },
                {
                    "token": "token_id:151645",
                    "bytes": [3],
                    "logprob": -0.01,
                    "top_logprobs": [
                        {
                            "token": "token_id:151645",
                            "bytes": [3],
                            "logprob": -0.01,
                        }
                    ],
                },
            ]
        },
    }


class FakeResponse:
    def __init__(self, payload: dict[str, object]) -> None:
        self._payload = payload

    def read(self) -> bytes:
        return json.dumps(self._payload).encode("utf-8")

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, *args: object) -> None:
        return None


class ParseLogprobsLiveShapeTests(unittest.TestCase):
    def test_rank0_matches_token_ids(self) -> None:
        parsed = parse_logprobs(_live_shaped_choice())
        self.assertEqual(parsed[0][0].token_id, 9989)
        self.assertEqual(parsed[1][0].token_id, 151645)
        self.assertEqual(
            parsed[0][1],
            TokenLogprob(token_id=69883, logprob=-4.5, rank=1),
        )

    def test_top_logprobs_accept_live_keys(self) -> None:
        choice = _live_shaped_choice()
        content = choice["logprobs"]["content"][0]
        top = content["top_logprobs"][0]
        self.assertEqual(set(top.keys()), {"token", "bytes", "logprob"})
        parsed = parse_logprobs(choice)
        self.assertEqual(parsed[0][0].token_id, 9989)

    def test_length_mismatch_fails_closed(self) -> None:
        choice = _live_shaped_choice()
        choice["token_ids"] = [9989]
        with self.assertRaisesRegex(RuntimeUnavailable, "length differ"):
            parse_logprobs(choice)

    def test_missing_token_ids_fails_closed(self) -> None:
        choice = _live_shaped_choice()
        del choice["token_ids"]
        with self.assertRaisesRegex(RuntimeUnavailable, "token_ids is missing"):
            parse_logprobs(choice)

    def test_null_token_ids_fails_closed(self) -> None:
        choice = _live_shaped_choice()
        choice["token_ids"] = None
        with self.assertRaisesRegex(RuntimeUnavailable, "token_ids is missing"):
            parse_logprobs(choice)

    def test_rank0_mismatch_fails_closed(self) -> None:
        choice = _live_shaped_choice()
        choice["token_ids"] = [1, 151645]
        with self.assertRaisesRegex(RuntimeUnavailable, "does not match token_ids"):
            parse_logprobs(choice)


class PromptLogprobsLiveShapeTests(unittest.TestCase):
    def test_leading_null_and_live_entry_keys(self) -> None:
        prompt_token_ids = [151644, 872, 198]
        prompt_logprobs = [
            None,
            {
                "872": {
                    "decoded_token": "user",
                    "logprob": -11.352999687194824,
                    "rank": 18943,
                },
                "32804": {
                    "decoded_token": "x",
                    "logprob": -12.0,
                    "rank": 20000,
                },
            },
            {
                "198": {
                    "decoded_token": "\n",
                    "logprob": -0.1,
                    "rank": 1,
                }
            },
        ]
        entry = prompt_logprobs[1]["872"]
        self.assertEqual(set(entry.keys()), {"decoded_token", "logprob", "rank"})
        parsed = _parse_prompt_logprobs(prompt_logprobs, prompt_token_ids, start=1)
        self.assertEqual(parsed[0][0].token_id, 872)
        self.assertEqual(parsed[0][0].logprob, -11.352999687194824)
        self.assertEqual(parsed[1][0].token_id, 198)

    def test_missing_prompt_token_ids_fails_closed(self) -> None:
        with self.assertRaisesRegex(RuntimeUnavailable, "prompt_token_ids"):
            _parse_prompt_logprobs([None, {"3": {"logprob": -0.25}}], None)


class CompletionPayloadFlagTests(unittest.TestCase):
    def test_completion_payload_requests_return_token_ids(self) -> None:
        agent = SmolagentsVLLMAgent("http://127.0.0.1:9")
        agent.begin(_context(), _config())
        payload = agent.completion_payload(agent.messages())
        self.assertIs(payload["return_token_ids"], True)
        self.assertIs(payload["return_tokens_as_token_ids"], True)


class TokenizeResponseKeyTests(unittest.TestCase):
    def test_teacher_force_plan_reads_tokens_list_of_ints(self) -> None:
        agent = SmolagentsVLLMAgent("http://127.0.0.1:9")
        agent.begin(_context(), _config())
        posted: list[str] = []

        def fake_urlopen(request, timeout=None):
            del timeout
            posted.append(request.full_url)
            if request.full_url.endswith("/tokenize"):
                return FakeResponse(
                    {
                        "count": 2,
                        "max_model_len": 32768,
                        "tokens": [11, 12],
                        "token_strs": None,
                    }
                )
            return FakeResponse(
                {
                    "choices": [{"message": {"content": ""}, "logprobs": None}],
                    "prompt_token_ids": [11, 12, 7],
                    "prompt_logprobs": [
                        None,
                        {
                            "12": {
                                "decoded_token": "a",
                                "logprob": -1.0,
                                "rank": 2,
                            }
                        },
                        {
                            "7": {
                                "decoded_token": "b",
                                "logprob": -0.3,
                                "rank": 1,
                            }
                        },
                    ],
                }
            )

        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            logprobs = agent.teacher_force_plan(
                messages=[
                    {"role": "user", "content": "Reply with the single word ping."}
                ],
                plan_text="ping",
            )
        self.assertEqual(posted[0], "http://127.0.0.1:9/tokenize")
        self.assertEqual(
            logprobs,
            ((TokenLogprob(token_id=7, logprob=-0.3, rank=0),),),
        )

    def test_tokenize_without_tokens_key_fails_closed(self) -> None:
        agent = SmolagentsVLLMAgent("http://127.0.0.1:9")
        agent.begin(_context(), _config())

        def fake_urlopen(request, timeout=None):
            del timeout
            if request.full_url.endswith("/tokenize"):
                return FakeResponse({"count": 2, "token_ids": [11, 12]})
            return FakeResponse(
                {
                    "choices": [{"message": {"content": ""}}],
                    "prompt_token_ids": [11, 12, 7],
                    "prompt_logprobs": [
                        None,
                        {"12": {"logprob": -1.0}},
                        {"7": {"logprob": -0.3}},
                    ],
                }
            )

        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            with self.assertRaisesRegex(RuntimeUnavailable, "plan prefix"):
                agent.teacher_force_plan(
                    messages=[
                        {"role": "user", "content": "Reply with the single word ping."}
                    ],
                    plan_text="ping",
                )


if __name__ == "__main__":
    unittest.main()
