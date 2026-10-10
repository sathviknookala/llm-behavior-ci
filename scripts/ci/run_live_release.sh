#!/usr/bin/env bash
set -uo pipefail

fail() {
  echo "$1" >&2
  exit 2
}

: "${LIFECYCLE_RELEASE_ROOT:?LIFECYCLE_RELEASE_ROOT is required}"
: "${RELEASE_ID:?RELEASE_ID is required}"
LIFECYCLE_PYTHON="${LIFECYCLE_PYTHON:-python3}"

name='^[A-Za-z0-9][A-Za-z0-9._-]*$'
[[ "$RELEASE_ID" =~ $name ]] || fail "release id must match $name"

release="$LIFECYCLE_RELEASE_ROOT/$RELEASE_ID"
[[ -f "$release/release.json" ]] || fail "release directory $RELEASE_ID is not provisioned"
attempt="$(date -u +%Y%m%dT%H%M%SZ)-$$"

"$LIFECYCLE_PYTHON" scripts/demo/run_three_tier_dev.py release --release-dir "$release" \
  > "$release/lifecycle_summary.$attempt.json" 2> "$release/lifecycle_stderr.$attempt.txt"
code=$?

case "$code" in
  0) outcome="promoted" ;;
  1) outcome="not promoted: blocked, admission refused, or rolled back" ;;
  2) outcome="invalid input or broken release lineage; nothing ran" ;;
  3) outcome="not promoted: canary incomplete, rolled back by cleanup" ;;
  4) outcome="execution failure" ;;
  5) outcome="not promoted: promoted, then rolled back to the previous production during Tier 3" ;;
  *) outcome="unexpected exit code" ;;
esac
echo "release $RELEASE_ID: exit $code, $outcome" | tee -a "${GITHUB_STEP_SUMMARY:-/dev/null}"

exit "$code"
