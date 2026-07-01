#!/usr/bin/env bash
# complete-base driver: build the abi CLI and drive the BASE completion path —
# `abi complete "<input>"` with no flags. This routes to the local model
# (claude-fable-5 by default), runs the constitution audit, and records the
# completion in the default WDBX store. It is fully local: no `--live` (no remote
# provider) and no `--learn` (that SEA path is covered by sea-learn-loop).
# Asserts exit codes + output markers. Resolves repo root from own path.
#
# Usage: .claude/skills/complete-base/complete.sh ["prompt"] [model-id]
set -uo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../../.." && pwd)
cd "$REPO_ROOT"
ABI="$REPO_ROOT/zig-out/bin/abi"
PROMPT="${1:-complete-base probe: summarize the scheduler status}"
MODEL="${2:-}"
fail=0
say() { printf '\n=== %s ===\n' "$*"; }
markers() { local out="$1"; shift; for m in "$@"; do
    printf '%s' "$out" | grep -qF -- "$m" && echo "[ok] marker: $m" \
        || { echo "[FAIL] missing marker: $m"; fail=$((fail+1)); }; done; }

say "build cli"
./build.sh cli >/dev/null 2>&1 && echo "[ok] build" || { echo "[FAIL] build"; exit 1; }
[ -x "$ABI" ] || { echo "[FAIL] $ABI not produced"; exit 1; }

if [ -n "$MODEL" ]; then
    say "abi complete --model $MODEL"
    out=$("$ABI" complete --model "$MODEL" "$PROMPT" 2>&1); rc=$?
else
    say "abi complete (default model)"
    out=$("$ABI" complete "$PROMPT" 2>&1); rc=$?
fi
printf '%s\n' "$out"
[ "$rc" -eq 0 ] || { echo "[FAIL] complete exit $rc"; fail=$((fail+1)); }
markers "$out" "model=" "audit_passed=true" "persisted=true" "wdbx kv_entries="

say "summary"; echo "failed checks: $fail"
[ "$fail" -eq 0 ] && echo "RESULT: PASS — base completion ran local + persisted." || echo "RESULT: FAIL — $fail check(s)."
exit "$fail"
