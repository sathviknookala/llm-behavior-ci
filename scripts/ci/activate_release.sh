#!/usr/bin/env bash
set -uo pipefail

fail() {
  echo "$1" >&2
  exit "${2:-2}"
}

: "${LIFECYCLE_RELEASE_ROOT:?LIFECYCLE_RELEASE_ROOT is required}"
: "${RELEASE_ID:?RELEASE_ID is required}"
LIFECYCLE_PYTHON="${LIFECYCLE_PYTHON:-python3}"
LIFECYCLE_SERVICE_UNIT="${LIFECYCLE_SERVICE_UNIT:-llm-behavior-ci-release.service}"
SYSTEMCTL="${SYSTEMCTL:-systemctl}"

name='^[A-Za-z0-9][A-Za-z0-9._-]*$'
[[ "$RELEASE_ID" =~ $name ]] || fail "release id must match $name"
release="$LIFECYCLE_RELEASE_ROOT/$RELEASE_ID"
[[ -f "$release/release.json" ]] || fail "release $RELEASE_ID is not provisioned"
active="$LIFECYCLE_RELEASE_ROOT/active"
previous=""
if [[ -L "$active" ]]; then
  previous="$(readlink "$active")"
fi

point() {
  ln -sfn "$1" "$LIFECYCLE_RELEASE_ROOT/.active.next" && mv -Tf "$LIFECYCLE_RELEASE_ROOT/.active.next" "$active"
}

point "$RELEASE_ID" || fail "could not point the service at $RELEASE_ID" 4
if "$SYSTEMCTL" --user restart "$LIFECYCLE_SERVICE_UNIT" \
  && "$LIFECYCLE_PYTHON" scripts/ci/release_ctl.py wait-registered --release-dir "$release"; then
  echo "release $RELEASE_ID: service active" | tee -a "${GITHUB_STEP_SUMMARY:-/dev/null}"
  exit 0
fi

if [[ -n "$previous" ]]; then
  point "$previous" && "$SYSTEMCTL" --user restart "$LIFECYCLE_SERVICE_UNIT"
  echo "release $RELEASE_ID: service did not start; restored $previous" | tee -a "${GITHUB_STEP_SUMMARY:-/dev/null}"
else
  rm -f "$active"
  "$SYSTEMCTL" --user stop "$LIFECYCLE_SERVICE_UNIT"
  echo "release $RELEASE_ID: service did not start" | tee -a "${GITHUB_STEP_SUMMARY:-/dev/null}"
fi
exit 4
