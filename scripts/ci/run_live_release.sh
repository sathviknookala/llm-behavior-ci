#!/usr/bin/env bash
set -uo pipefail

fail() {
  echo "$1" >&2
  exit 2
}

: "${LIFECYCLE_RELEASE_ROOT:?LIFECYCLE_RELEASE_ROOT is required}"
: "${LIFECYCLE_SERVICE_URL:?LIFECYCLE_SERVICE_URL is required}"
: "${RELEASE_ID:?RELEASE_ID is required}"
: "${CANARY_ARRIVALS:?CANARY_ARRIVALS is required}"
: "${PRODUCTION_ARRIVALS:?PRODUCTION_ARRIVALS is required}"
PREVIOUS_RELEASE_ID="${PREVIOUS_RELEASE_ID:-}"
LIFECYCLE_PYTHON="${LIFECYCLE_PYTHON:-python3}"

name='^[A-Za-z0-9][A-Za-z0-9._-]*$'
[[ "$RELEASE_ID" =~ $name ]] || fail "release id must match $name"
[[ -z "$PREVIOUS_RELEASE_ID" || "$PREVIOUS_RELEASE_ID" =~ $name ]] || fail "previous release id must match $name"
[[ "$RELEASE_ID" != "$PREVIOUS_RELEASE_ID" ]] || fail "a release cannot continue itself"
[[ "$CANARY_ARRIVALS" =~ ^[0-9]+$ && "$PRODUCTION_ARRIVALS" =~ ^[0-9]+$ ]] || fail "arrival counts must be integers"

release="$LIFECYCLE_RELEASE_ROOT/$RELEASE_ID"
[[ -d "$release" ]] || fail "release directory $RELEASE_ID is not provisioned"

args=(
  live
  --gate-reference "$release/gate_reference.json"
  --gate-candidate "$release/gate_candidate.json"
  --task-set "$release/train_task_set.json"
  --gate-settings "$release/gate_settings.json"
  --plan-evidence "$release/plan_evidence.json"
  --production-config "$release/production.json"
  --candidate-config "$release/candidate.json"
  --dev-task-set "$release/dev_task_set.json"
  --canary-arrivals "$CANARY_ARRIVALS"
  --production-arrivals "$PRODUCTION_ARRIVALS"
  --service-url "$LIFECYCLE_SERVICE_URL"
  --store "$release/episodes.sqlite"
  --release-summary "$release/release_summary.json"
)
if [[ -n "$PREVIOUS_RELEASE_ID" ]]; then
  args+=(--previous-release "$LIFECYCLE_RELEASE_ROOT/$PREVIOUS_RELEASE_ID/release_summary.json")
fi
if [[ -n "${LIFECYCLE_REFERENCE_ENDPOINT:-}" ]]; then
  args+=(--reference-endpoint "$LIFECYCLE_REFERENCE_ENDPOINT")
fi
if [[ -n "${LIFECYCLE_CANDIDATE_ENDPOINT:-}" ]]; then
  args+=(--candidate-endpoint "$LIFECYCLE_CANDIDATE_ENDPOINT")
fi

"$LIFECYCLE_PYTHON" scripts/demo/run_three_tier_dev.py "${args[@]}" \
  > "$release/lifecycle_summary.json" 2> "$release/lifecycle_stderr.txt"
code=$?

case "$code" in
  0) outcome="promoted" ;;
  1) outcome="not promoted: blocked, admission refused, or rolled back" ;;
  2) outcome="invalid input or broken release lineage; nothing ran" ;;
  3) outcome="not promoted: canary incomplete, rolled back by cleanup" ;;
  4) outcome="execution failure" ;;
  *) outcome="unexpected exit code" ;;
esac
echo "release $RELEASE_ID: exit $code, $outcome" | tee -a "${GITHUB_STEP_SUMMARY:-/dev/null}"

exit "$code"
