"""Pure translation of a ``ModelConfiguration`` into a vLLM server launch spec.

Nothing here imports or starts vllm. ``build_vllm_launch_spec`` is the one
place ``model.quantization.method``, ``model.lora``, and
``model.serving.batch_invariant`` become the argv and environment a real
``vllm serve`` invocation would need, so the quantization, LoRA, and
batch-invariant fault kinds can be proven, with fakes, to change the real
launch path rather than only a config field nothing downstream reads.
``live_fault_available`` in ``experiments.faults`` still reports these three
as needing a different running server process: reconstructing the spec a
process would have started with is not the same as starting it.
"""

from __future__ import annotations

from dataclasses import dataclass

from llm_behavior_ci.config import ModelConfiguration

BATCH_INVARIANT_ENV = "VLLM_BATCH_INVARIANT"


@dataclass(frozen=True)
class VLLMLaunchSpec:
    """The argv and environment one ``vllm serve`` process would use.

    ``env`` is a tuple of ``(name, value)`` pairs, not a dict, so the spec
    stays hashable and its equality is exact: two configurations that would
    launch identically produce an identical spec, and one changed field
    changes it.
    """

    argv: tuple[str, ...]
    env: tuple[tuple[str, str], ...]


def build_vllm_launch_spec(model: ModelConfiguration) -> VLLMLaunchSpec:
    """Build the launch spec one ``ModelConfiguration`` implies.

    ``--quantization`` is present only when ``quantization.method`` is not
    ``"none"``. ``--enable-lora``/``--lora-modules`` are present only when
    ``model.lora`` is set, naming the adapter's revision-qualified
    repository as the served LoRA module. ``VLLM_BATCH_INVARIANT=1`` is
    present in ``env`` only when ``serving.batch_invariant`` is true. Every
    other serving flag this function emits is unconditional, so the healthy
    reference configuration (no quantization, no LoRA, batch invariance
    off) always produces the same spec modulo those three optional pieces.
    """

    if not isinstance(model, ModelConfiguration):
        raise TypeError("build_vllm_launch_spec requires a ModelConfiguration")
    serving = model.serving
    argv: list[str] = [
        "vllm",
        "serve",
        model.model.repository,
        "--revision",
        model.model.revision,
        "--tokenizer",
        model.tokenizer.repository,
        "--tokenizer-revision",
        model.tokenizer.revision,
        "--dtype",
        serving.dtype,
        "--max-model-len",
        str(serving.max_model_len),
        "--gpu-memory-utilization",
        str(serving.gpu_memory_utilization),
        "--max-num-seqs",
        str(serving.max_num_seqs),
        "--max-num-batched-tokens",
        str(serving.max_num_batched_tokens),
        "--kv-cache-dtype",
        serving.kv_cache_dtype,
        "--tensor-parallel-size",
        str(serving.tensor_parallel_size),
        "--max-logprobs",
        str(serving.max_logprobs),
    ]
    argv.append(
        "--enable-prefix-caching"
        if serving.enable_prefix_caching
        else "--no-enable-prefix-caching"
    )
    argv.append(
        "--enable-chunked-prefill"
        if serving.enable_chunked_prefill
        else "--no-enable-chunked-prefill"
    )
    if serving.enforce_eager:
        argv.append("--enforce-eager")
    if model.quantization.method != "none":
        argv.extend(["--quantization", model.quantization.method])
    if model.lora is not None:
        argv.append("--enable-lora")
        argv.extend(
            [
                "--lora-modules",
                f"{model.lora.repository}="
                f"{model.lora.repository}@{model.lora.revision}",
            ]
        )
    env: list[tuple[str, str]] = []
    if serving.batch_invariant:
        env.append((BATCH_INVARIANT_ENV, "1"))
    return VLLMLaunchSpec(argv=tuple(argv), env=tuple(env))
