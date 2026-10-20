#!/usr/bin/env bash
# sea-learn-loop driver: build the abi CLI and exercise the SEA self-learning
# completion path (`abi complete --learn`). Asserts the loop runs, persists, and
# reports evidence/adaptation counters. Resolves the repo root from its location.
#
# SEA is always available in the Rust workspace (no feature-off CLI flag).
# `--sea` is accepted for compatibility and is a no-op build-path alias.
#
# Usage:
#   .claude/skills/sea-learn-loop/learn.sh ["input text"]
#   .claude/skills/sea-learn-loop/learn.sh --sea ["input text"]
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../../.." && pwd)
cd "$REPO_ROOT"

SEA=0
INPUT="sea-learn-loop probe: summarize the scheduler status"
for a in "$@"; do
    case "$a" in
        --sea) SEA=1 ;;
        *) INPUT="$a" ;;
    esac
done

ABI="$REPO_ROOT/target/debug/abi"
# Two separate processes must persist and recall from the same fresh scratch
# store. Never reuse/delete another run's directory or inherit live settings.
SCRATCH_DIR="$REPO_ROOT/target/skill-scratch"
say() { printf '\n=== %s ===\n' "$*"; }
die() { echo "RESULT: FAIL — $*"; exit 1; }

say "build cli (SEA always linked in Rust workspace)"
if ./tools/cargo.sh build --locked -p abi-cli --target-dir "$REPO_ROOT/target"; then
    echo "[ok] build"
else
    die "build failed; no existing binary was run"
fi
[ -x "$ABI" ] || { echo "[FAIL] $ABI not produced"; exit 1; }

mkdir -p "$SCRATCH_DIR"
STORE=$(mktemp -d "$SCRATCH_DIR/sea-learn.XXXXXX")
trap 'rm -rf -- "$STORE"' EXIT
export ABI_WDBX_PATH="$STORE"
export ABI_WDBX_PERSIST=1

say "abi complete --learn (ABI_WDBX_PATH=$STORE)"
for pass in 1 2; do
    say "learning pass $pass"
    if out=$("$ABI" complete --learn -- "$INPUT" 2>&1); then
        printf '%s\n' "$out"
    else
        rc=$?
        printf '%s\n' "$out"
        die "complete --learn exit $rc"
    fi
    # Only the metadata line is evidence: generated response text can repeat
    # arbitrary markers and must not turn a failed write into a passing smoke.
    status=${out%%$'\n'*}
    for marker in "learn=true" "provider=local" \
        "generation_engine=persona-template" "persisted=true" "adapted=true"; do
        [[ " $status " == *" $marker "* ]] || die "missing metadata: $marker"
    done
    [[ "$status" =~ (^|[[:space:]])requested_model=[^[:space:]]+ ]] \
        || die "missing requested model"
    [[ "$status" =~ (^|[[:space:]])evidence_count=([0-9]+)($|[[:space:]]) ]] \
        || die "invalid evidence counter"
    evidence=${BASH_REMATCH[2]}
    if [[ "$pass" -eq 2 && ! "$evidence" =~ [1-9] ]]; then
        die "second process did not recall persisted evidence"
    fi
done

say "summary"
echo "feat-sea: $([ "$SEA" -eq 1 ] && echo flag-accepted-noop || echo default-on)"
echo "RESULT: PASS — SEA learning persisted, adapted and recalled across processes."
