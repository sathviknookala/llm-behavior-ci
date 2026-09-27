# Risks

Read before interpreting a false alarm, a miss, or a threshold, and before changing a method. Mitigations are part of the plan. None of them has been executed.

| Risk | Mitigation |
|---|---|
| Inference nondeterminism makes every monitor false-alarm | Measure the A/A floor with and without batch-invariant mode (stage 0), and include the A/A streams in every false-alarm number |
| Thresholds tuned on the benchmark make methods look better than they are | A separate calibration stream. Thresholds frozen in `EVAL_PROTOCOL.md` |
| The fault catalog is too easy, or too hard | Measure real harm first. Include subtle faults (FP8, a single-slice mapping bug, ramped onsets). Report per fault |
| Natural drift is labeled a false alarm when it is harm, or the reverse | Harm is defined by measured accuracy against the pre-registered margin, per period |
| Top-k KL is misleading | Measure truncation error against full-vocabulary KL (stage 2) |
| Two vLLM processes do not fit 24 GB | Smaller KV cache. The LoRA-adapter fallback, only after the two-process path is tried. A 4B model at FP8 as production is the handoff's further relief |
| A self-hosted runner exposes the machine | CI rules in `CONSTRAINTS.md`. Ask first |
| The project drifts into an observability platform | Non-negotiable 7 in `PROJECT_SPEC.md`, and question 5 of the change test |
| confseq is unmaintained | Implement confidence sequences in this repo. confseq is only a reference check |
| The LLM is weak at the task | The monitoring question still stands. Report the LLM beside the logistic-regression baseline |

A monitor whose false-alarm rate has not been checked on a simulated null and on A/A traffic cannot appear in the benchmark.
