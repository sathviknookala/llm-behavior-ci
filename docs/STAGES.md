# Stages

Read before starting or closing a stage. Status of every stage: **not started**. A hardcoded CPU demo of the three offline-gate statistics exists; it has no replay data, null validation, reference checks, protocol thresholds, or measurements and does not complete a stage. A stage is complete when its gate's evidence is committed, not when a process starts.

Suggested numeric defaults belong in `EVAL_PROTOCOL.md` and stay open until that file is pre-registered. Do not invent a threshold in a script.

Keep the order. Stages 0–2 produce the noise floor and the null checks. The benchmark is stage 6 and does not start before the protocol is committed.

| Stage | Status |
|---|---|
| 0 Service skeleton, replay harness, noise floor | not started |
| 1 Tasks, baseline model, natural drift | not started |
| 2 Statistics library and validity checks | not started |
| 3 Release gate in CI | not started |
| 4 Canary and automatic rollback | not started |
| 5 Production monitors and alerts | not started |
| 6 Fault-injection benchmark | not started |
| 7 Write-up and resume facts | not started |

## Stage 0 — service skeleton, replay harness, and noise floor

Gateway (FastAPI):

- routes requests to one or two vLLM servers;
- logs the request, the response, the top-k log-probabilities, the model version, configuration hash, latency, and timestamp;
- supports a traffic split and a one-command rollback.

Replay harness: streams a dataset in timestamp order, with the clock compressed (the rate is pre-registered), open-loop at a set request rate, and deterministic given a seed.

**The A/A noise floor.** Replay the same stream twice against the same model under different concurrency. Measure the label disagreement rate, the token-level divergence, and the task-metric difference. Measure both with and without `VLLM_BATCH_INVARIANT=1`, and record the throughput cost of batch-invariant mode.

Gate: no dropped logs under the target load; the replay is reproducible; the noise floor is measured and committed.

## Stage 1 — tasks, baseline model, and natural drift

- **T1 (primary).** arXiv abstracts: predict the primary category from a fixed label set (for example, the 15–25 most frequent categories). Output is JSON `{label, confidence}` through vLLM's structured output.
- **T2.** HuffPost headlines and short descriptions: predict the news category, with a pre-registered mapping that merges duplicate categories.
- **Default model.** A Qwen3-4B chat checkpoint (Apache 2.0), thinking mode off. Record the exact revision.
- Prompt and label descriptions are fixed. Confidence comes from label-token log-probabilities, calibrated on an early-years calibration split (temperature scaling; report the expected calibration error).
- Baselines on the same splits: the majority class, and TF-IDF + logistic regression trained on the early years. If logistic regression beats the LLM, report it. The monitoring question does not depend on the LLM winning.
- Natural drift curve: accuracy by year, with bootstrap confidence intervals. It shows whether topic drift over time is harmful or benign for this service. Part 3 of the question depends on it.

Gate: the LLM beats the majority class on both tasks, and the label sets and splits are committed.

## Stage 2 — statistics library and its validity checks

Implement in this repo:

- the paired bootstrap;
- full-vocabulary KL (Transformers, teacher-forced) and top-k KL (from vLLM logs);
- a chi-square test on the label distribution;
- KS on confidences;
- MMD with a permutation null, on sentence embeddings of inputs and on output-text embeddings;
- a classifier two-sample test;
- CUSUM;
- ADWIN;
- Howard et al. confidence sequences for Bernoulli accuracy and for bounded differences;
- e-detectors (Shin et al.);
- the Podkopaev–Ramdas harmful-shift test;
- a sequential canary test in the style of Lindon et al.

Validity checks, run in CI on CPU:

- simulated nulls confirm each method's false-alarm rate or coverage at its nominal level (fixed-window tests are expected to inflate over repeated looks; show that too);
- reference checks against River, confseq, and SciPy on shared inputs;
- on the real A/A streams from stage 0, every detector's false-alarm rate is measured.

Truncation check: top-k KL against full-vocabulary KL on a sample, at k = 20 (vLLM's default `max_logprobs`) and larger.

Gate: every method matches its reference, and its measured null behavior is committed. A method that fails its own null check is fixed or dropped before stage 3.

## Stage 3 — the release gate in CI

**Fault catalog.** Each entry is a configuration diff. Each is run offline on labeled data to measure its true accuracy drop with a confidence interval, and classified as harmful (the drop is at least the pre-registered margin) or benign before any gate or monitor sees it.

1. FP8 weights (vLLM FP8 quantization).
2. NVFP4 weights (made with llm-compressor, or a published checkpoint; log which).
3. Model downgrade to Qwen3-1.7B.
4. A prompt edit that removes the label descriptions.
5. A chat-template bug (thinking mode on, or the system prompt dropped).
6. Sampling temperature 0 → 1.0.
7. `max_tokens` too small, which truncates the JSON.
8. Structured output disabled.
9. A label-mapping bug affecting one category (a fault confined to one slice).
10. A LoRA fine-tune on off-distribution data.
11. Benign controls: a no-op redeploy, toggling batch-invariant mode, a logging refactor, and a candidate identical to production.

**The gate.** On every PR that changes the model, prompt, or serving configuration, replay a pre-registered sample of logged traffic through the candidate and production. Compute the paired bootstrap on accuracy and JSON validity, full KL (same tokenizer only), and MMD on outputs. It fails when a confidence interval crosses the margin or a divergence test rejects.

Report power against replay-set size for each fault, the false-block rate on the benign controls, and GPU-minutes per gate run.

**Anchor.** Run the Gao et al. MMD test on one model modification from their released benchmark and match their reported detection behavior within a tolerance written in `EVAL_PROTOCOL.md`, or document why it does not transfer. Their paper reports a median power of 77.4% across modifications. That figure is theirs (`PRIOR_WORK.md`), not a result of this repo.

Gate: end to end in GitHub Actions, or the local-runner path in `CONSTRAINTS.md`, a planted harmful change is blocked and a no-op passes.

## Stage 4 — canary and automatic rollback

The gateway sends a pre-registered fraction of traffic to the candidate. Two vLLM processes share the GPU with split `gpu_memory_utilization`. The fallback, only after that path has been tried, is LoRA adapters on one base model (`PROJECT_SPEC.md`).

The canary test compares candidate and production as traffic arrives. Label-free signals right away: the output label distribution, confidence, JSON validity, length. Accuracy once delayed labels arrive (label delay is simulated and pre-registered). Methods: the sequential canary test, a confidence sequence on the accuracy difference, and a fixed-window baseline.

Rollback is automatic when a test rejects, and it is logged with the evidence.

Chaos tests: kill the candidate mid-canary; send malformed responses; slow the candidate down. The gateway must fail over to production.

Measure the bad requests served before rollback, and the time to rollback.

Gate: for a planted harmful candidate, the whole sequence works end to end: deploy, canary, reject, roll back, alert.

## Stage 5 — production monitors and alerts

Monitors on the production stream:

- input drift (MMD or a classifier test on input embeddings, windowed and sequential);
- output behavior (the label distribution, confidences, JSON validity, refusal and length rates);
- label-based accuracy once delayed labels arrive (a confidence sequence and the Podkopaev–Ramdas harmful-shift test);
- a label-free accuracy estimate from calibrated confidences (the idea behind NannyML's CBPE), with its error reported against the true accuracy;
- embedding health (effective rank) for the input encoder.

Alerts go to a webhook that Sathvik configures. Each alert carries which monitor fired, the evidence (statistic, threshold, window), the slice that moved most, the deploy and configuration hash, and a link to the runbook section. Alerts are de-duplicated and rate-limited. Do not connect an alert to an account until he names the channel (`DECISIONS.md`).

A dashboard (Grafana or a static page) shows vLLM latency metrics beside the behavior metrics.

Gate: every monitor runs live on replayed traffic, and a planted fault produces exactly one de-duplicated alert that names the right slice.

## Stage 6 — the fault-injection benchmark

This is the headline. It runs only after `EVAL_PROTOCOL.md` is pre-registered and committed.

Pre-register: the fault catalog with its measured harm labels; each fault's onset, abrupt or ramped (the faulty version's share of traffic grows over time); both tasks; the healthy horizon; replicates (seeds and onsets); α and thresholds, set on a separate calibration stream.

Methods compared on the same streams:

- hourly KS and chi-square, with and without correction;
- CUSUM;
- ADWIN;
- confidence sequences;
- e-detectors;
- the Podkopaev–Ramdas test;
- the label-free estimate.

Metrics per method: detection delay in requests and simulated hours; the miss rate within the horizon; false alarms per healthy horizon, including across natural-drift periods and benign controls; the share of harmful faults caught against the share of benign changes flagged; compute cost.

Report with confidence intervals over replicates, and state which method you would ship at which traffic level.

## Stage 7 — write-up and resume facts

- A README with the headline figure (detection delay against false alarms per method) and exact commands.
- A workshop-style draft. Ask before submitting.
- `RESUME_FACTS.md`, filled only from measured results.
- Upstream PRs are optional and require asking first. Candidates named in the handoff: an anytime-valid detector for River, or a sequential drift test for Evidently, if either lacks one. Its documentation, as checked 2026-09-26, shows none.
