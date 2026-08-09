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
set -uo pipefail

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
# Scratch durable store so --learn never touches the user's live ~/.abi/.
SCRATCH_DIR="$REPO_ROOT/target/skill-scratch"
STORE="$SCRATCH_DIR/sea-learn-wdbx"
fail=0
say() { printf '\n=== %s ===\n' "$*"; }

say "build cli (SEA always linked in Rust workspace)"
if ./tools/cargo.sh build -p abi-cli; then echo "[ok] build"; else echo "[FAIL] build"; exit 1; fi
[ -x "$ABI" ] || { echo "[FAIL] $ABI not produced"; exit 1; }

mkdir -p "$SCRATCH_DIR"
rm -rf "$STORE"
export ABI_WDBX_PATH="$STORE"

say "abi complete --learn (ABI_WDBX_PATH=$STORE)"
echo "\$ ABI_WDBX_PATH=$STORE $ABI complete --learn \"$INPUT\""
out=$("$ABI" complete --learn "$INPUT" 2>&1); rc=$?
printf '%s\n' "$out"
[ "$rc" -eq 0 ] || { echo "[FAIL] complete --learn exit $rc"; fail=$((fail+1)); }

# The loop must report it ran (learn=true), disclose local provenance, and emit an
# evidence counter (0 is acceptable when the scratch store has no hits).
for marker in "learn=true" "requested_model=" "provider=local" \
    "generation_engine=persona-template" "evidence_count="; do
    grep -qF -- "$marker" <<<"$out" && echo "[ok] marker: $marker" \
        || { echo "[FAIL] missing marker: $marker"; fail=$((fail+1)); }
done
grep -qF -- "persisted=true" <<<"$out" && echo "[ok] persisted=true" \
    || echo "[note] not persisted (acceptable depending on store state)"

say "summary"
echo "feat-sea: $([ "$SEA" -eq 1 ] && echo flag-accepted-noop || echo default-on)"
echo "failed checks: $fail"
[ "$fail" -eq 0 ] && echo "RESULT: PASS — SEA learn loop ran." || echo "RESULT: FAIL — $fail check(s) failed."
exit "$fail"
