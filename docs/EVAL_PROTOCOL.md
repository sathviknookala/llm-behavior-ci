# Evaluation protocol

**Status: DRAFT. Not pre-registered.** Written 2026-09-26 so the slots exist before any data does. No benchmark stream runs against this file until every slot marked OPEN is filled, fault harm labels are measured on labeled data, thresholds are set on a calibration stream that is not a benchmark stream, and this file is committed in that state.

A number in this file is a planned slot or a cited prior. It is not a result of this repo. Results, when they exist, live under `results/` with the script that regenerates them.

## Slots

| Slot | State | Value |
|---|---|---|
| Tasks | planned | T1 arXiv primary category; T2 HuffPost category. Label sets, the T2 merge map, and the splits are OPEN until stage 1 commits them. |
| Default model | planned | Qwen3-4B chat, Apache 2.0, thinking mode off. Exact revision OPEN. |
| Harm margin | OPEN | The accuracy drop that classifies a fault as harmful. No number has been chosen. |
| Fault catalog | planned | The eleven configuration diffs in `STAGES.md` stage 3. Harm labels are not measured. |
| Fault onsets | OPEN | Abrupt or ramped, and the start times. |
| Methods in the benchmark | planned | The stage 6 set in `STAGES.md`. Parameters below are OPEN. |
| Thresholds | OPEN | Tuned only on the calibration stream, then frozen here. Not tuned on benchmark streams. The `DEMO_*` thresholds in `src/offline_gate/offline.py` do not fill this slot. |
| α | OPEN | Nominal level for each test. The demo's value is an execution fixture, not a protocol choice. |
| Bootstrap draws | proposed | 10,000, carried as the rig style from the quantization study (`PRIOR_WORK.md`). Not locked. The CPU demo uses fewer draws only to keep its CI check small. |
| Replication floor | rule, unmeasured | Repeated baseline runs, reported beside every difference. The floor is not subtracted from a difference. The floor's value does not exist yet. |
| Healthy horizon | example only | "30 simulated days" is an example from the handoff, not a lock. |
| Replicates | OPEN | Independent replay seeds and fault start times. |
| Label delay | OPEN | Simulated delay before accuracy labels arrive. |
| Canary fraction | OPEN | Share of traffic sent to the candidate. |
| Replay rate and clock compression | OPEN | Pre-register both. The harness is open-loop at the set rate and deterministic given a seed. |
| Calibration vs benchmark split | rule | Disjoint. Benchmark streams are run once per pre-registered configuration. |
| Top-k for KL truncation check | planned | k = 20, vLLM's default `max_logprobs`, and larger values. The truncation error is a measurement still to run. |
| Batch-invariant mode | OPEN | Stage 0 measures the A/A floor with and without `VLLM_BATCH_INVARIANT=1`, including the throughput cost. The benchmark records which mode was on. |
| Gao et al. match tolerance | OPEN | The anchor is their reported median power of 77.4% across modifications (arxiv.org/abs/2410.20247). The tolerance for "matched" is not chosen. |

## Current offline-gate demo

`src/offline_gate/` currently provides a deterministic, standard-library execution path over hardcoded healthy inputs:

- a paired percentile bootstrap of `candidate - production`;
- per-position `D_KL(production || candidate)` in nats from normalized log-probabilities, with no epsilon flooring when candidate support is missing;
- a biased squared MMD with an RBF kernel and a Monte Carlo p-value from within-request production/candidate swaps, preserving the paired replay design.

`src/offline_gate/offline.py` combines those three checks and `scripts/run_offline_gate.py` prints their evidence. This path proves that the modules compose and that GitHub-hosted CPU CI can execute them. It does not establish calibration, false-alarm control, coverage, reference agreement, an A/A floor, or a release threshold. It therefore satisfies neither the stage 2 nor stage 3 gate.

## What each stage is allowed to use

The library implements the full set in `STAGES.md` stage 2. Each consumer uses a subset:

- **Release gate:** paired bootstrap on accuracy and JSON validity; full-vocabulary KL when the tokenizer matches; MMD on outputs. Fail when a confidence interval crosses the margin or a divergence test rejects.
- **Canary:** the sequential canary test; a confidence sequence on the accuracy difference; a fixed-window baseline. Label-free signals immediately (label distribution, confidence, JSON validity, length); accuracy after the pre-registered label delay.
- **Production monitors:** input MMD or a classifier two-sample test; output label distribution, confidences, JSON validity, refusal and length; label-based accuracy via a confidence sequence and the Podkopaev–Ramdas test; a label-free accuracy estimate from calibrated confidences, with its error against true accuracy; effective rank of the input encoder.
- **Benchmark comparison, same streams:** hourly KS and chi-square, with and without correction; CUSUM; ADWIN; confidence sequences; e-detectors; Podkopaev–Ramdas; the label-free estimate.

## Metrics the benchmark reports

Per method, with confidence intervals over replicates:

- detection delay in requests and in simulated hours;
- miss rate within the horizon;
- false alarms over the healthy horizon, including natural-drift periods and benign controls, and including the A/A streams;
- share of harmful faults caught against share of benign changes flagged;
- compute cost.

Also, for the gate: power against replay-set size per fault, false-block rate on the benign controls, GPU-minutes per gate run. For the canary: bad requests served before rollback, and time to rollback.

## Reporting that travels with every headline

The replay rate, the clock-compression factor, the model revision, whether batch-invariant mode was on, the commit, GPU-hours, and peak memory.

Accuracy numbers carry bootstrap confidence intervals over requests. Benchmark numbers carry confidence intervals over replicates.

Scope line that travels with the claim: replayed traffic, operator-side monitoring, one GPU.

## What is not ground truth

An LLM judge. A divergence with no labeled accuracy drop. A threshold chosen on a benchmark stream. A monitor that has not passed its null check. A synthetic-null false-alarm rate reported without the A/A and natural-drift streams.

## Run log

`results/` contains no result artifacts, only its README and empty stage-aligned directories. When runs start, every benchmark configuration is logged there as raw CSV or JSON, with a script that regenerates every figure. A run that is not in that tree is not a result.
