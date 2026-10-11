# GitHub-driven release pipeline

How a change on `main` becomes a release: CPU verification, an immutable release directory, a real Tier 1 gate, the Tier 2 canary, promotion or rollback, and Tier 3 monitoring in a persistent service. This is the operator's reference. `HOSTED_LIFECYCLE.md` step 8a covers the lifecycle command itself.

**Status (2026-10-10):** wired and CPU-tested, never run live. No runner is registered, no environment exists, and no paid call has been made. Everything under "CPU-verified" below runs in GitHub's hosted CPU jobs. Everything under "Untested live" depends on infrastructure and spend that are not authorized yet.

## Sequence

```
push to main (or workflow_dispatch mode=release)
 -> cpu         ubuntu-latest: unit, ci, integration, validity suites
 -> synthetic   ubuntu-latest: blocked, healthy, regression, production_regression
 -> provision   self-hosted, environment release-provision
                  release_ctl.py provision  (builds $ROOT/<release_id>, no model call)
 -> live        self-hosted, environment release-live (required reviewers = spend approval)
                  release_ctl.py check-deployed   running service == lineage, no canary open
                  activate_release.sh             verify the release, point $ROOT/active, restart the unit, wait for registration
                  run_live_release.sh             run_three_tier_dev.py release --release-dir ...
```

- **Push to `main`.** `cpu` and `synthetic` always run. `provision` and `live` run only when the repository variable `LIFECYCLE_RELEASE_ENABLED` is `true`, so a push never queues on an unregistered runner. The release id is `main-<first 12 hex of the commit>`.
- **Manual dispatch.** `mode=synthetic` runs only the CPU jobs. `mode=release` needs `release_id` and `confirm_live` equal to it, and runs from `main` only.
- **Paid stage.** `live` waits for a required reviewer on `release-live`. Nothing paid runs until a person approves that deployment. A push without approval leaves a waiting deployment that can be rejected.
- **Pinned commit, full history.** Every self-hosted job checks out `github.sha` with `fetch-depth: 0`. The cross-commit checks (`git merge-base --is-ancestor`, `git show` of the compatibility file, and the execution fingerprint at release N's commit) need release N's commit in the checkout. A shallow clone is refused with a message naming `fetch-depth: 0`. Provisioning refuses a dirty tree, and the release command refuses a manifest provisioned at any other commit.
- **Outcomes.** Only exit `0` (promoted) passes `live`. `1` blocked, rolled back, or admission refused; `2` invalid input or broken lineage, nothing ran; `3` canary incomplete; `4` execution failure; `5` promoted, then rolled back during Tier 3. All non-zero codes fail the job.
- **Logs.** Jobs print exit codes, hashes, and outcome lines. The lifecycle summary and stderr stay in the release directory as `lifecycle_summary.<attempt>.json` and `lifecycle_stderr.<attempt>.txt`. Nothing is uploaded, and no workflow reads `secrets.`.

## Cross-commit release contract

Configuration hashes include `git_commit`, so a deployed configuration is never rebuilt at a new commit. Release N+1 runs at commit B. It keeps release N's serving configuration as production, still built at its own commit A.

1. **Candidate side at HEAD.** The gate candidate and the dev candidate are built at the clean release commit (`require_release_provenance`).
2. **Carried production.** Production and the gate reference come byte-for-byte from release N's files, whichever pair hashes to the configuration release N left serving. Their commit is A.
3. **Trusted record.** `verify_release_record` re-derives release N from release N's own SQLite store. It checks the single gate artifact (id, outcome, evidence source, hashes), the lifecycle decision sequence, the status implied by the last decision, the serving configuration, and that every decision and episode belongs to N's two configurations. A summary that disagrees is refused. The one accepted divergence is a promoted release whose store gained a later `production_rollback`, for example through monitoring after the job ended. The known-good configuration then continues.
4. **Actual deployment.** `require_deployed` checks the running service at provisioning and again just before activation. It must serve that configuration and have no canary in `GATE_PASSED` or `CANARY_ACTIVE`. The service started for release N+1 must register the manifest's hashes (`wait-registered`). The lifecycle's own `require_registered` checks them once more.
5. **Runtime transition.** `require_runtime_transition(A, B)`: the checkout must not be shallow, and A must be an ancestor of B. Then one of two things must hold:
   - The execution fingerprint at A equals the one at B (`identical_execution`). The fingerprint is a SHA-256 over the git object ids of `requirements.txt`, `environments/`, the whole `src/llm_behavior_ci/` package, `scripts/service/`, `scripts/ci/`, `scripts/demo/run_three_tier_dev.py`, `scripts/synthetic/run_connected_lifecycle.py` (which that command loads), and `configs/tasks/`. The package is covered whole because the service, lifecycle, monitors, statistics, storage, and runtime import one another, so a change to any of them can change how production is served, monitored, or rolled back. Documentation, tests, workflows, model templates, and faults are outside it.
   - `configs/release/runtime_compatibility.json`, as committed at B, declares A's fingerprint compatible, with a reason (`declared_compatible`).

   Anything else is refused. An incompatible runtime change ships as a candidate built at B, through the gate and the canary, not by re-executing old production under new code. A declaration is a reviewed, committed claim. It is the one place where a person vouches that B executes A's configuration identically.

   **Compatibility consequences of the wider fingerprint (2026-10-10).** Both fingerprints are computed with the path list of the code at B, so A and B are always compared over the same paths. The commit that widened the list also changed `storage.py`, `runtime/budget.py`, `lifecycle/release.py`, and the release scripts. Its fingerprint therefore differs from that of every earlier commit. A production built at `4bae819` or earlier cannot be carried into a release at or after that commit without a declaration. No such declaration is committed, and none should be, because that commit changes how service model calls are counted. No release has run live, so no lineage exists to break. The first release runs fresh from `initial_production`. Also expect future releases to need a declaration, or a fresh candidate built at B, more often than before: any change under `src/llm_behavior_ci/` now counts.
6. **Lineage freshness.** A release continues the release in `$ROOT/LATEST`. `LATEST` advances only after a release summary is written. A release provisioned from an older `LATEST` is refused at deployment.

`release-summary-v3` records `git_commit` (the release commit), `commits.{production,candidate,serving}`, and `runtime` (the transition). `release-summary-v1` and `-v2` records stay readable as release N. Their single `git_commit` is every configuration's commit.

The first release has no lineage. Production is built from `initial_production` at the release commit, and no service may already be deployed.

## Release spec (`release-spec-v1`, local, never committed)

One JSON file on the runner, named by the environment variable `LIFECYCLE_RELEASE_SPEC`. Relative paths resolve against its directory. Template and fault paths are repository paths under `configs/models/` and `configs/faults/`, read at the release commit.

```json
{
  "version": "release-spec-v1",
  "candidate": {"template": "configs/models/<candidate>.json", "fault": null},
  "initial_production": {"template": "configs/models/<production>.json", "fault": null},
  "run_seed": 17,
  "protocol_hash": null,
  "train_manifest": "train_gate.json",
  "dev_manifest": "dev_release.json",
  "gate_settings": { "...": "GateSettings.to_dict()" },
  "plan_evidence": "plan_evidence.json",
  "canary_settings": { "...": "CanarySettings.to_dict()" },
  "monitor_settings": { "...": "MonitorSettings without reference_configuration_hash" },
  "frozen_reference": {"baselines": [["task_success", 0.5]], "slice_baselines": [], "source": null},
  "service": {
    "host": "127.0.0.1", "port": 8000, "max_in_flight": 2,
    "shutdown_timeout_seconds": 60, "dedup_seconds": 0, "canary_assignment_seed": 0,
    "production_base_url": null, "candidate_base_url": null,
    "post_promotion_rollback": "record_only"
  },
  "gate_endpoints": {"reference": null, "candidate": null},
  "budget": {
    "canary_arrivals": 12, "production_arrivals": 12, "max_episodes": 120,
    "max_gate_model_calls": 200, "max_service_model_calls": 2000
  }
}
```

- `post_promotion_rollback` is required. `rollback_on_alert` exists only when a spec says so explicitly. `serve.py`'s default stays `record_only`.
- The service listens on localhost only.
- Gate thresholds, canary settings, monitor settings, and baselines here are DRAFT development settings. They fill no slot in `EVAL_PROTOCOL.md`.
- A carried production keeps its task binding. If the spec's train or dev manifest differs from the one production was built on, provisioning refuses.

## Budgets

- **Episodes (static).** Provisioning refuses a spec whose worst case exceeds `max_episodes`. The worst case is 2 × train tasks (gate pairs) + 2 × canary arrivals (pairs) + production arrivals. The release command refuses the same bound again before any model call.
- **Durable accounting.** Both model-call budgets live in the release's `episodes.sqlite`, in the `model_calls` table (`storage.ModelCallLedger`). Before every generation and every teacher-forced scoring call, the budget opens its own connection and takes SQLite's write lock (`BEGIN IMMEDIATE`). It then counts the scope's reservations, inserts one more if the limit allows, and commits. The model is called only after that commit. The row is then settled `completed` or `failed`. A row left `reserved` means the process died during the call. Every row is charged, whatever its state. Because the lock is SQLite's, the cap holds across threads, across the service and the release command sharing one store, and across processes. A restart or a rerun reads the same rows back. The gate (`gate` scope) and the service (`service` scope) never spend each other's limit, and each release's fresh store starts at zero. A ledger that cannot be written stops the call as a runtime failure.
- **Gate model calls (hard).** The lifecycle wraps the gate's runtime factory in `BudgetedRuntimeFactory` with scope `gate`. The call past `max_gate_model_calls` raises before reaching the provider, and the gate ends as an execution failure (exit 4). A rerun of the release against the same store continues from the calls already reserved.
- **Service model calls (hard).** `serve.py --max-model-calls` wraps the service's runtimes in one shared budget with scope `service`. It charges the larger of its reservations and the model steps already persisted in the store. Every new step has a reservation, so nothing is counted twice, and a store written before reservations existed does not refund its calls. `POST /episodes` returns 503 when the remaining budget cannot cover the request at its turn limit (both arms while a canary is active). The reservation still stops a call inside an episode. The budget is a lifetime cap on attempted provider calls for that release's service, Tier 3 included, across restarts.

## Release directory

`$LIFECYCLE_RELEASE_ROOT/<release_id>/` is written to a hidden staging directory and renamed into place last. It is never overwritten. `provision --reuse-existing` accepts an existing directory only if its manifest verifies at HEAD and it still continues `LATEST`.

| File | Contents |
| --- | --- |
| `release.json` | `release-manifest-v2`: commit, previous release id, summary SHA-256, lineage, configuration hashes and commits, runtime transition, task-set hashes and counts (no ids), budgets, service settings, gate endpoints, SHA-256 of every file below, and the `code` digest |
| `gate_reference.json`, `gate_candidate.json`, `production.json`, `candidate.json` | configurations (read-only) |
| `train_task_set.json`, `dev_task_set.json` | local manifests (private; never leave the machine) |
| `gate_settings.json`, `plan_evidence.json`, `canary_settings.json`, `monitor_settings.json`, `frozen_reference.json`, `task_selection_allowance.json`, `service.json` | settings (read-only) |
| `episodes.sqlite` | the fresh store, writable: episodes, decisions, alerts, and model-call reservations |
| `code/` | `git archive` of the release commit plus `RELEASE_COMMIT`, read-only and pinned by `code` in the manifest; the persistent service runs from here |
| `release_summary.json` | written once by the release run |
| `lifecycle_summary.<attempt>.json`, `lifecycle_stderr.<attempt>.txt` | written by each release attempt |

**Code integrity.** Provisioning records `code_tree_digest(code/)` (`release-code-v1`) in the manifest. That is a SHA-256 over every directory, regular file (content and executable bit), and symlink (its target text, never followed) under `code/`, sorted by path, plus the entry count. It needs no `.git`. `verify_release_manifest` recomputes it and checks `RELEASE_COMMIT`. It is called by `provision --reuse-existing`, `check-deployed`, `verify`, `wait-registered`, the release command, and `serve` at every service start. An added, removed, renamed, or modified file is refused even when `RELEASE_COMMIT` is unchanged. So is a retargeted or new symlink, a stray `__pycache__`, a `code/` that is itself a symlink, or any special file. Only the store and the files outside `code/` in the table above are written after provisioning. A `release-manifest-v1` directory has no code digest and is refused.

`$LIFECYCLE_RELEASE_ROOT/active` is a symlink to the release the service runs. `$LIFECYCLE_RELEASE_ROOT/LATEST` names the last release with a summary.

## Persistent service

`scripts/ci/llm-behavior-ci-release.service` is a systemd user unit. It runs `$ROOT/active/code/scripts/ci/release_ctl.py serve --release-dir $ROOT/active` with `PYTHONDONTWRITEBYTECODE=1`, and `release_ctl.py` also turns off bytecode writing, so serving never changes the tree it verifies. `serve` refuses to start unless every pinned file and the code digest are unchanged, `RELEASE_COMMIT` matches the manifest, it runs from that release's own `code/`, and `llm_behavior_ci` is imported from there. That check runs from the tree it checks, so it catches accidental or partial changes, not a deliberate rewrite of the checker. The trusted check is `activate_release.sh`. Before it repoints `active`, it runs `release_ctl.py verify` from the job's checkout at the release commit, and a failure exits 2 without touching the service. It then starts `serve.py` with arguments read only from the manifest. The arguments include the task-selection allowance, the dev manifest as task metadata, the rollback policy, and the model-call budget.

The service keeps running after the job ends. It serves whatever the lifecycle left serving, feeds the production monitor, and applies the release's rollback policy. On restart, `_reconcile_deployment` restores the stored state from the same store. `activate_release.sh` switches releases. It verifies the release from the checkout, repoints `active`, restarts the unit, and waits for registration. If the new release does not register, it restores the previous release and fails with exit 4. If there was no previous release, it stops the unit.

## CPU-verified vs untested live

**CPU-verified** (GitHub-hosted, no credentials, no inference, no AppWorld data):
- All four synthetic scenarios: status, deployment state, serving configuration, stored decisions, persisted gate artifact and episodes, and exit code (`scripts/ci/check_synthetic.py`).
- Provisioning from a throwaway git repository. Consecutive releases carry production across commits A→B (`identical_execution`). An execution change at C is refused. A committed declaration at D is accepted (`declared_compatible`).
- Tampered, stale, or mismatched summaries and stores. Deployed-service mismatches. Non-ancestor and unknown commits. Overwrite refusal and leftover-free failure. Changed release files. `v1` and `v2` migration.
- Services built by `serve.py`'s own builder from each manifest. Routing by configuration hash. Gate BLOCK. Canary promotion and rollback. Post-promotion rollback (exit 5). Restart restoring PROMOTED and PRODUCTION_ROLLED_BACK. The service budget surviving a restart, including a failed provider call that left no model step.
- Durable model-call accounting: success, a failed call with no model step, restart after that failure, 16 threads and 4 separate processes racing for the last units (never above the limit), teacher-forced scoring, a `KeyboardInterrupt`, and a child process killed with `os._exit` after its reservation and before any response. An unwritable ledger calls no model. Scopes and stores are isolated, and a pre-reservation store is not refunded.
- Full history: the workflow's two self-hosted checkouts set `fetch-depth: 0`. A real `git clone --depth 1` refuses a cross-commit transition and cross-commit provisioning, and the same transition passes after a full clone or `git fetch --unshallow`.
- Fingerprint scope: changes to the service, canary, monitor, a detector, storage, the protocol, `serve.py`, `release_ctl.py`, the activation script, the lifecycle command and its fixtures, committed task metadata, or dependency pins each change the fingerprint and are refused without a declaration. README, `docs/`, tests, a model template, a fault, or a workflow change keeps it.
- Code integrity: modified, added, deleted, renamed, executable-bit, bytecode, symlink-added, file-to-symlink, and retargeted-symlink tampering, and a symlinked `code/`. Each is refused by `verify_release_manifest`, and a tampered tree also fails reuse, `check-deployed`, `verify`, `serve`, and the release command. A manifest without its code digest, or a v1 manifest, is refused. Writing the store, summaries, and attempt logs still verifies.
- Branch protection: every check named in "Required status checks" below is a job in a workflow with a `pull_request` trigger and no self-hosted runner, and no `Release lifecycle` check is listed.
- Shell steps: release-id resolution, activation (verify before restart, a failed verify touching nothing, success, restore on failure, first-release failure), and outcome mapping.

**Untested live:**
- systemd and the actual unit.
- GitHub environments, approvals, and variables.
- A self-hosted runner.
- Real provider or vLLM calls.
- A real gate PASS (release admission refuses `synthetic_fixture`, so a live candidate is admitted only on `gate_run` evidence).
- AppWorld worlds.
- Wall time against `timeout-minutes: 360`.
- `fetch-depth: 0` on the real runner's checkout (CPU tests read the YAML and use local `git clone --depth 1`).
- Ledger lock contention under real provider latency. A call the provider accepted but whose response was lost is charged once here. The provider may still bill it, and nothing reconciles against provider-side usage.
- Code verification on the real release root and filesystem, including its permissions and ownership.
- Behavior when the 360-minute timeout kills a run mid-canary. The unit keeps the service alive. The next `check-deployed` refuses while a canary is open, and a restart closes it as `restart_with_unfinished_canary`.

## GitHub and machine setup (account actions, none done)

`DECISIONS.md` D12 is still open. Registering a runner needs Sathvik's decision first.

1. **Fork pull requests.** Settings → Actions → General → "Fork pull request workflows from outside collaborators": **Require approval for all external contributors**. On a public personal repository this approval is the only barrier between fork code and a repository runner, because a fork can add its own workflow targeting the runner's labels. Never approve a fork run while the runner is online. Making the repository private, or moving it to an organization whose runner group is restricted to this workflow on `main`, removes the hazard.
2. **Branch protection for `main`.** Require a pull request before merging. Required status checks: `offline-tests`. That is the only job of the `Offline statistical gate` workflow, which runs on every pull request and on pushes to `main`; GitHub shows it on a pull request as `Offline statistical gate / offline-tests`, and the branch-protection search finds it as `offline-tests`. It runs the same four CPU suites as `Release lifecycle / CPU verification`, but not the four synthetic scenarios. Do not require any `Release lifecycle` check: that workflow runs only on pushes to `main` and manual dispatch, so its checks never report on a pull request and would block every merge. Adding a `pull_request` trigger to it is not the fix either, because it is the workflow that reaches the self-hosted runner. Block force pushes.
3. **Environment `release-provision`.** Deployment branches: selected, `main` only. No reviewers. Variables:
   - `LIFECYCLE_RELEASE_ROOT`: an absolute directory outside the runner's work tree, for example `/home/<user>/llm-behavior-ci-releases`.
   - `LIFECYCLE_RELEASE_SPEC`: the absolute path of the release spec.
   - `LIFECYCLE_PYTHON`: the `.venv-service` interpreter.
4. **Environment `release-live`.** Deployment branches: `main` only. Required reviewers: Sathvik. Leave "prevent self-review" off on a single-maintainer repository. Variables: the same `LIFECYCLE_RELEASE_ROOT` and `LIFECYCLE_PYTHON`. No secrets.
5. **Repository variable** `LIFECYCLE_RELEASE_ENABLED` = `true`, only once 1–4 and 6–8 are done. Leave it unset to keep pushes CPU-only.
6. **Runner.** Settings → Actions → Runners → New self-hosted runner (Linux x64). Run `./config.sh --url https://github.com/sathviknookala/llm-behavior-ci --token <token> --labels llm-behavior-ci-release --name llm-behavior-ci-release`. Then `sudo ./svc.sh install <user>` and `sudo ./svc.sh start`. Put `APPWORLD_ROOT` and the provider key the gate needs (`ZAI_API_KEY` or `ANTHROPIC_API_KEY`) in the runner's `.env`, mode 600. The runner then passes them to its jobs. The workflow never echoes them.
7. **Service unit.**
   ```bash
   mkdir -p ~/.config/llm-behavior-ci ~/.config/systemd/user "$LIFECYCLE_RELEASE_ROOT"
   install -m 600 /dev/null ~/.config/llm-behavior-ci/release.env
   # release.env: LIFECYCLE_RELEASE_ROOT=..., LIFECYCLE_PYTHON=..., APPWORLD_ROOT=..., and the provider key
   cp scripts/ci/llm-behavior-ci-release.service ~/.config/systemd/user/
   systemctl --user daemon-reload
   systemctl --user enable llm-behavior-ci-release.service
   sudo loginctl enable-linger <user>
   ```
   Do not start the unit by hand. The first release's `activate_release.sh` starts it.
8. **Spec and manifests.** Build the train and dev manifests and the plan evidence (`HOSTED_LIFECYCLE.md` steps 1–2), then write the release spec.

## First approved release

1. Merge `tier2-canary-test` into `main` through a reviewed pull request.
2. Complete setup steps 1–8, and check `nvidia-smi` if a vLLM endpoint is involved.
3. Approve the spend for one release. Its ceiling is the spec's `max_gate_model_calls` plus `max_service_model_calls`.
4. Actions → Release lifecycle → Run workflow on `main` with `mode=release`, `release_id=r1`, and `confirm_live=r1`. Alternatively, set `LIFECYCLE_RELEASE_ENABLED=true` and push.
5. Approve the `release-live` deployment when it waits.
6. Read the step summary line. The full summary is `$ROOT/r1/lifecycle_summary.<attempt>.json` and the public record is `$ROOT/r1/release_summary.json`.
7. The service keeps running from `$ROOT/active`. The next release on `main` continues from `LATEST`.
