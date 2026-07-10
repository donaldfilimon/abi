#!/usr/bin/env bash
# wdbx-roundtrip driver: build the abi CLI and drive a full WDBX persistence
# round-trip on a scratch segment — db init -> block insert -> query -> db verify.
# Proves the on-disk checkpoint + WAL chain stays valid across the cycle.
# Asserts exit codes + output markers. Resolves repo root from own path.
#
# Usage: .claude/skills/wdbx-roundtrip/roundtrip.sh [profile] [metadata-json]
set -uo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../../.." && pwd)
cd "$REPO_ROOT"
ABI="$REPO_ROOT/zig-out/bin/abi"
PROFILE="${1:-abi}"
META="${2:-{\"note\":\"wdbx-roundtrip\"}}"
STORE="$REPO_ROOT/zig-out/skill-wdbx-roundtrip.jsonl"
fail=0
say() { printf '\n=== %s ===\n' "$*"; }
step() { local label="$1"; shift; local -a markers=(); local a
    while [ "$1" != "--" ]; do markers+=("$1"); shift; done; shift
    local out; out=$("$@" 2>&1); local rc=$?
    printf '%s\n' "$out"
    [ "$rc" -eq 0 ] || { echo "[FAIL] $label exit $rc"; fail=$((fail+1)); }
    for a in "${markers[@]}"; do
        grep -qF -- "$a" <<<"$out" && echo "[ok] $label :: $a" \
            || { echo "[FAIL] $label missing: $a"; fail=$((fail+1)); }; done; }

say "build cli"
./build.sh cli >/dev/null 2>&1 && echo "[ok] build" || { echo "[FAIL] build"; exit 1; }
[ -x "$ABI" ] || { echo "[FAIL] $ABI not produced"; exit 1; }
rm -f "$STORE"

say "wdbx db init";      step "init"   "initialized empty WDBX" -- "$ABI" wdbx db init "$STORE"
say "wdbx block insert"; step "insert" "appended block:" "blocks=1" -- "$ABI" wdbx block insert "$STORE" "$PROFILE" "$META"
say "wdbx query";        step "query"  '"blocks":1' -- "$ABI" wdbx query "$STORE"
say "wdbx db verify";    step "verify" "checkpoint OK:" "chain_valid=true" -- "$ABI" wdbx db verify "$STORE"

rm -f "$STORE"
say "summary"; echo "failed checks: $fail"
[ "$fail" -eq 0 ] && echo "RESULT: PASS — WDBX persistence round-trip verified." || echo "RESULT: FAIL — $fail check(s)."
exit "$fail"
