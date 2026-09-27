# Prior work and tools

Read before finalizing `EVAL_PROTOCOL.md` or writing a comparison claim. Facts below are as checked 2026-09-26 in the handoff. They are not measurements from this repo. Do not claim "first".

Read Gao et al., Podkopaev & Ramdas, Lindon et al., and the substitution-audit paper before the protocol leaves draft status.

## Positioning

Detector benchmarks (Failing Loudly) and model-substitution tests (Gao et al. and successors) compare against a reference once, or batch by batch. The anytime-valid literature is evaluated mostly in simulation. Lindon et al. cover general software canaries. The survey that produced the handoff found no head-to-head of fixed-window tests, CUSUM, ADWIN, and anytime-valid methods on detection delay, misses, and false alarms over long healthy traffic in an LLM service's release lifecycle.

That comparison, with harm defined by labeled data and nondeterminism measured as a source of false alarms, is the contribution this repo is built to make. It is an engineering benchmark, not a new statistical method.

## Papers

| Item | Fact as recorded | Source |
|---|---|---|
| Model equality testing (Gao, Liang, Guestrin; ICLR 2025) | Detecting a silently changed API model (quantization, watermarking, fine-tuning, system-prompt changes) as two-sample testing; MMD with a string kernel; median power 77.4% across real-world modifications; discrepancies on production endpoints; code released | arxiv.org/abs/2410.20247 |
| Auditing model substitution in LLM APIs (2025) | Text-only statistical tests need many queries and miss subtle substitutions such as quantized models; log-probability methods are defeated by ordinary inference nondeterminism; proposes trusted execution environments. Authors not confirmed in the handoff | arxiv.org/abs/2504.04715 |
| Rank-based uniformity test (2025) | A query-efficient test for detecting quantization, fine-tuning, or full substitution behind black-box APIs | arxiv.org/abs/2506.06975 |
| Failing Loudly (Rabanser, Günnemann, Lipton; NeurIPS 2019) | Benchmark of dataset-shift detectors: dimensionality reduction plus a two-sample test (MMD, KS, classifier). A black-box classifier's softmax outputs plus per-dimension KS was consistently among the strongest | arxiv.org/abs/1810.11953 |
| ChatGPT behavior over time (Chen, Zaharia, Zou; 2023) | GPT-4 prime-number accuracy fell from 84% to 51% between the March and June 2023 snapshots; argues for continuous behavior monitoring | arxiv.org/abs/2307.09009 |
| Harmful-shift tracking (Podkopaev & Ramdas; ICLR 2022) | A sequential, anytime-valid test that alarms on rising risk of a chosen metric, not on any distribution shift | arxiv.org/abs/2110.06177 |
| Confidence sequences (Howard, Ramdas, McAuliffe, Sekhon; Annals of Statistics 2021) | Confidence bounds valid uniformly over time and stopping rules, under nonparametric tail conditions | arxiv.org/abs/1810.08240 |
| Always-valid A/B testing (Johari, Koomen, Pekelis, Walsh; KDD 2017; Operations Research 2022) | Always-valid p-values via a mixture sequential probability ratio test (mSPRT), deployed in Optimizely | doi.org/10.1287/opre.2021.2135 |
| E-detectors (Shin, Ramdas, Rinaldo; 2023) | Nonparametric sequential change detection from e-processes, with bounds on false-alarm run length and detection delay | arxiv.org/abs/2203.03532 |
| Safe anytime-valid inference (Ramdas, Grünwald, Vovk, Shafer; Statistical Science 2023) | The theory behind e-processes and confidence sequences | arxiv.org/abs/2210.01948 |
| Sequential canary testing (Lindon, Sanden, Shirikian; KDD 2022) | Sequential tests for regressions in software deployments with type-I control under continuous monitoring. General software, not LLMs | arxiv.org/abs/2205.14762 |
| Prediction-powered risk monitoring (ICML 2026) | Extends Podkopaev–Ramdas with synthetic plus scarce true labels; partly validated on language-model evaluation | arxiv.org/abs/2602.02229 |
| "Who Drifted" (June 2026 preprint, single author, no venue found) | Anytime-valid attribution of drift between an LLM system and its judge. A rolling z-test false-alarmed on 75% of drift-free streams. Closest single item, and a narrower setting. Treat provenance cautiously | arxiv.org/abs/2606.15474 |
| Foundations | ADWIN: Bifet & Gavaldà, SDM 2007. CUSUM: Page, Biometrika 1954. MMD: Gretton et al., JMLR 2012. Classifier two-sample tests: Lopez-Paz & Oquab, ICLR 2017 | cs.upc.edu/~gavalda/papers/adwin06.pdf ; jmlr.org/papers/v13/gretton12a.html |
| Nondeterminism (Thinking Machines, September 2025) | LLM inference is not batch-invariant, so outputs change with concurrent load even at temperature 0. Batch-invariant kernels give bit-identical outputs | thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference |

The 84% and 51% figures, the 77.4% median power, and the 75% false-alarm figure are claims in those papers. They are not results of this repo.

## Tools

| Item | Fact as recorded | Source |
|---|---|---|
| vLLM log-probabilities | `logprobs` per token is capped by `max_logprobs`, default 20. `prompt_logprobs` gives teacher-forced scores. No full-vocabulary logits through the API, so exact KL needs Transformers | docs.vllm.ai/en/stable/configuration/engine_args |
| vLLM determinism | Outputs vary with batch composition even at temperature 0. `VLLM_BATCH_INVARIANT=1` enables batch-invariant kernels at a performance cost, on compute capability 8.0+. This card qualifies | docs.vllm.ai/en/latest/features/batch_invariance |
| vLLM metrics | Prometheus `/metrics`: time to first token, time per output token, end-to-end latency, KV-cache use, queue depth | docs.vllm.ai/en/stable/design/metrics |
| Two versions on one GPU | Multi-LoRA serving on one base model, or two processes with split `gpu_memory_utilization`, which shrinks KV cache for both | docs.vllm.ai, for the installed version, when one exists |
| GitHub runners | Self-hosted runners should almost never serve public repositories. GitHub-hosted GPU runners are paid, about $0.052/min for Linux as checked 2026-09-26, and need a Team or Enterprise plan | docs.github.com/en/actions/reference/security/secure-use ; docs.github.com runner pricing |
| Models | Qwen3 0.6B–8B dense are Apache 2.0. Llama 3.1 uses Meta's community license | qwenlm.github.io/blog/qwen3 ; llama.com/llama3_1/license |

## Licenses of nearby tools

Checked 2026-09-26 against the projects' LICENSE files, as recorded in the handoff. Reference checks may use River, confseq, and SciPy. Do not vendor a copy of any of these as this repo's implementation.

| Tool | License as recorded | Role relative to this repo |
|---|---|---|
| Evidently | Apache-2.0. Batch tests; no sequential tests found | Positioning only. A sequential test upstream would be an optional PR after asking |
| NannyML | Apache-2.0. CBPE/DLE label-free estimation | The label-free monitor follows the idea, implemented here, with error reported against true accuracy |
| River | BSD-3. ADWIN, KSWIN, Page-Hinkley | Reference check only |
| confseq | MIT. Early-stage, v0.0.11, January 2023 | Reference check only |
| promptfoo | MIT. A GitHub Action for prompt regressions | Positioning only |
| DeepEval | Apache-2.0 | Positioning only |
| Deepchecks | AGPL-3.0 | Positioning only. Do not import |
| Alibi Detect | BSL 1.1 from January 22, 2024 | Do not use |
| Arize Phoenix | Elastic License 2.0 | Positioning only. Do not import |
| Prometheus | Apache-2.0 | Metrics scrape |
| Grafana | AGPLv3 since April 2021 | Optional dashboard (`DECISIONS.md` D11) |
