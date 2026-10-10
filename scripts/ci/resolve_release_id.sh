#!/usr/bin/env bash
set -uo pipefail

fail() {
  echo "$1" >&2
  exit 2
}

: "${EVENT_NAME:?EVENT_NAME is required}"
: "${COMMIT:?COMMIT is required}"
[[ "$COMMIT" =~ ^[0-9a-f]{40}$ ]] || fail "commit must be a 40-character git id"

case "$EVENT_NAME" in
  push)
    id="main-${COMMIT:0:12}"
    ;;
  workflow_dispatch)
    id="${INPUT_RELEASE_ID:-}"
    [[ -n "$id" ]] || fail "release_id is required"
    [[ "${CONFIRM_LIVE:-}" == "$id" ]] || fail "confirm_live must repeat release_id"
    ;;
  *)
    fail "releases start only from a push to main or a manual dispatch"
    ;;
esac

name='^[A-Za-z0-9][A-Za-z0-9._-]{0,99}$'
[[ "$id" =~ $name ]] || fail "release id must match $name"
[[ "$id" != "active" && "$id" != "LATEST" ]] || fail "release id $id is reserved"
echo "release_id=$id" >> "${GITHUB_OUTPUT:-/dev/null}"
echo "release $id at $COMMIT"
