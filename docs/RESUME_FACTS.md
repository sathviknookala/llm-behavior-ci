# Resume facts

No headline has been measured. Every slot below stays blank until the number is in a committed artifact under `results/`, with the script that regenerates it. Sathvik merges accepted facts into his own inventory. Do not edit that repo from this project.

For every headline that eventually lands here, record:

- the metric, the baseline, and n (episodes, tasks, streams, replicates);
- the confidence interval;
- the commit, hardware, GPU-hours, and peak memory.

The SWE resume uses the system bullet plus one result. The ML resume uses the benchmark result. Shapes below are placeholders. The letters are not data.

- **Built CI/CD for a tool-using agent on vLLM:** plan-trace CI gates, paired canary rollback, and production regression alerts over N AppWorld episodes.
- **Anytime-valid monitors caught X% of N injected regressions** within M episodes at a Y% false-alarm rate over the healthy horizon, against Z% false alarms for hourly KS tests.
- **Plan-only CI gate blocked X of Y harmful changes** (name the faults) with no false blocks across Z no-op redeploys.
- **Canary rollback cut failed candidate episodes served** from X to Y against a fixed-window test, at the same false-rollback rate.

Rendering rules, from the handoff's account of his facts file:

- 2–3 bullets per project, about 25–35 words each.
- At most two numbers per bullet, and a result travels with its baseline.
- One bold span of 5–10 words.
- Lead with the result or the artifact.
- No bullet built from a finding this repo did not produce.
