from __future__ import annotations

import unittest
from dataclasses import replace

from llm_behavior_ci.config import (
    LoRASettings,
    ModelConfiguration,
    ModelRevision,
    QuantizationSettings,
    VLLMBehaviorSettings,
)
from llm_behavior_ci.runtime.launch_spec import (
    BATCH_INVARIANT_ENV,
    FLASHINFER_SAMPLER_ENV,
    build_vllm_launch_spec,
)

_REVISION = "0" * 40
_TOKENIZER_REVISION = "f" * 40
_LORA_REVISION = "d" * 40


def _model(**overrides: object) -> ModelConfiguration:
    base = ModelConfiguration(
        model=ModelRevision(repository="Qwen/Qwen3-4B", revision=_REVISION),
        tokenizer=ModelRevision(
            repository="Qwen/Qwen3-4B",
            revision=_TOKENIZER_REVISION,
        ),
        quantization=QuantizationSettings(method="none"),
        vllm_version="0.30.0",
        serving=VLLMBehaviorSettings(
            dtype="bfloat16",
            max_model_len=8192,
            gpu_memory_utilization=0.9,
            max_num_seqs=16,
            max_num_batched_tokens=8192,
            kv_cache_dtype="bfloat16",
            enable_prefix_caching=False,
            enable_chunked_prefill=False,
            enforce_eager=False,
            tensor_parallel_size=1,
            max_logprobs=20,
            batch_invariant=False,
            sampler_backend="native",
        ),
    )
    return replace(base, **overrides) if overrides else base


class LaunchSpecTests(unittest.TestCase):
    def test_healthy_configuration_has_no_optional_flags(self) -> None:
        spec = build_vllm_launch_spec(_model())
        self.assertIn("Qwen/Qwen3-4B", spec.argv)
        self.assertNotIn("--quantization", spec.argv)
        self.assertNotIn("--enable-lora", spec.argv)
        self.assertNotIn("--enforce-eager", spec.argv)
        self.assertEqual(spec.env, ((FLASHINFER_SAMPLER_ENV, "0"),))

    def test_quantization_method_reaches_the_launch_spec(self) -> None:
        quantized = _model(quantization=QuantizationSettings(method="fp8"))
        spec = build_vllm_launch_spec(quantized)
        index = spec.argv.index("--quantization")
        self.assertEqual(spec.argv[index + 1], "fp8")

        healthy = build_vllm_launch_spec(_model())
        self.assertNotEqual(spec.argv, healthy.argv)

    def test_batch_invariant_sets_the_env_var_only(self) -> None:
        serving = replace(_model().serving, batch_invariant=True)
        invariant = _model(serving=serving)
        spec = build_vllm_launch_spec(invariant)
        self.assertEqual(
            spec.env,
            ((BATCH_INVARIANT_ENV, "1"), (FLASHINFER_SAMPLER_ENV, "0")),
        )
        self.assertNotIn("--batch-invariant", spec.argv)

        healthy = build_vllm_launch_spec(_model())
        self.assertEqual(healthy.env, ((FLASHINFER_SAMPLER_ENV, "0"),))
        self.assertEqual(spec.argv, healthy.argv)

    def test_sampler_backend_sets_the_flashinfer_env_value_only(self) -> None:
        serving = replace(_model().serving, sampler_backend="flashinfer")
        flashinfer = build_vllm_launch_spec(_model(serving=serving))
        native = build_vllm_launch_spec(_model())
        self.assertEqual(flashinfer.env, ((FLASHINFER_SAMPLER_ENV, "1"),))
        self.assertEqual(native.env, ((FLASHINFER_SAMPLER_ENV, "0"),))
        self.assertEqual(flashinfer.argv, native.argv)

    def test_lora_adds_enable_lora_and_names_the_adapter(self) -> None:
        lora = LoRASettings(repository="org/adapter", revision=_LORA_REVISION)
        with_lora = _model(lora=lora)
        spec = build_vllm_launch_spec(with_lora)
        self.assertIn("--enable-lora", spec.argv)
        index = spec.argv.index("--lora-modules")
        self.assertEqual(spec.argv[index + 1], f"org/adapter=org/adapter@{_LORA_REVISION}")

        healthy = build_vllm_launch_spec(_model())
        self.assertNotIn("--enable-lora", healthy.argv)
        self.assertNotEqual(spec.argv, healthy.argv)

    def test_reset_to_healthy_configuration_reproduces_the_healthy_spec(self) -> None:
        lora = LoRASettings(repository="org/adapter", revision=_LORA_REVISION)
        faulted = _model(
            lora=lora,
            quantization=QuantizationSettings(method="fp8"),
        )
        healthy_again = replace(faulted, lora=None, quantization=QuantizationSettings(method="none"))
        self.assertEqual(
            build_vllm_launch_spec(healthy_again),
            build_vllm_launch_spec(_model()),
        )

    def test_rejects_non_model_configuration_input(self) -> None:
        with self.assertRaises(TypeError):
            build_vllm_launch_spec(object())  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
