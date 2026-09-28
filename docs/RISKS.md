# Risks

Read before interpreting a false alarm, a miss, or a threshold, and before changing a method. Mitigations are part of the plan. None of them has been executed.

| Risk | Mitigation |
|---|---|
| Inference nondeterminism makes every monitor false-alarm, and in an agent one different token can send the whole trajectory elsewhere | Measure the A/A floor with and without batch-invariant mode (stage 0), including trajectory divergence and outcome disagreement, and include the A/A streams in every false-alarm number |
| Thresholds tuned on the benchmark make methods look better than they are | Tune only on `dev`. Thresholds frozen in `EVAL_PROTOCOL.md` before `test_normal` is run |
| Repeated exposure to `test_normal` turns it into a development set | Run `test_normal` only in stage 6, once per pre-registered configuration, after everything in the protocol's frozen list is committed. No human inspects individual `test_normal` tasks or task-wise reports, and nothing changes afterward |
| Harm measured on `dev` transfers differently to `test_normal` | Labels stay frozen from `dev`. Report each fault's `test_normal` effect size beside its `dev` label, and treat disagreement as a finding, never as a reason to relabel |
| Qwen3-4B succeeds on too few AppWorld tasks for a harm margin to be detectable (a floor effect) | Stage 1 measures success on `dev` and commits a power calculation before any threshold. Final-state correctness (fraction of requirements passed) is a finer outcome than success. Changing the model size or task selection is Sathvik's call, pre-registered before the benchmark |
| Episodes are not independent: AppWorld groups task variants into scenarios, and long streams repeat tasks | Cluster intervals and resampling at the scenario level, not the raw task. Measure each detector's false-alarm rate on real `train` and `dev` A/A streams with the same scenario structure, not only on i.i.d. simulated nulls |
| A `test_normal` slice uses metadata that split does not release, such as ground-truth required apps | Slice `test_normal` only by its released difficulty indicators and the agent's own observed behavior. Fix the slices on `dev` before the benchmark |
| Plan-only gate evidence does not predict executed harm | Report gate agreement with execution-measured harm labels per fault. A gate that misses executed harm is a finding, not a failure to hide |
| The fault catalog is too easy, or too hard | Measure real harm by execution on `dev` first. Include subtle faults (FP8, a single-app documentation bug, ramped onsets). Report per fault |
| A task-mix shift is labeled a false alarm when it is harm, or the reverse | Harm is defined on configuration faults by measured evaluator success against the pre-registered margin. Monitor the task mix itself, and report the task-mix curve from stage 1 |
| The agent's actions reach the host, or one canary world mutates the other | All execution goes through AppWorld under the boundary in `DECISIONS.md` D20. Stage 0 and stage 4 test world isolation |
| AppWorld's protected content, or any episode-level artifact, leaks into git, CI logs, or a publication | The public and local-only lists in `DATA.md`. Public artifacts are aggregates, counts, rules, seeds, and hashes. Check before every commit and every publish step |
| Top-k KL is misleading | Measure truncation error against full-vocabulary KL on plan traces (stage 2) |
| Two vLLM processes do not fit 24 GB at agent context lengths | Smaller KV cache. The LoRA-adapter fallback, only after the two-process path is tried. A 4B model at FP8 as production is the handoff's further relief |
| A self-hosted runner exposes the machine | CI rules in `CONSTRAINTS.md`. Ask first |
| The project drifts into an observability platform or an agent-building project | Non-negotiable 7 in `PROJECT_SPEC.md`, and question 5 of the change test |
| confseq is unmaintained | Implement confidence sequences in this repo. confseq is only a reference check |

A method whose false-alarm rate has not been checked on a simulated null and on A/A traffic cannot appear in the benchmark.
