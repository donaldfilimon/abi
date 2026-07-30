#!/usr/bin/env bash
# complete-base driver: build the abi CLI and drive the BASE completion path —
# `abi complete "<input>"` with no flags. This routes to the local model
# (claude-fable-5 by default) and runs the constitution audit. Persistence is
# optional: when no durable WDBX path is configured the CLI reports
# `persisted=false` honestly. Fully local: no `--live`, no `--learn`.
# Asserts exit codes + output markers. Resolves repo root from own path.
#
# Usage: .agents/skills/complete-base/complete.sh ["prompt"] [model-id]
set -uo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../../.." && pwd)
cd "$REPO_ROOT"
ABI="$REPO_ROOT/target/debug/abi"
PROMPT="${1:-complete-base probe: summarize the scheduler status}"
MODEL="${2:-}"
fail=0
say() { printf '\n=== %s ===\n' "$*"; }
markers() { local out="$1"; shift; for m in "$@"; do
    grep -qF -- "$m" <<<"$out" && echo "[ok] marker: $m" \
        || { echo "[FAIL] missing marker: $m"; fail=$((fail+1)); }; done; }

say "build cli"
./tools/cargo.sh build -p abi-cli >/dev/null 2>&1 && echo "[ok] build" || { echo "[FAIL] build"; exit 1; }
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
markers "$out" "model=" "audit_passed=true" "wdbx kv_entries="
if grep -qE 'persisted=(true|false)' <<<"$out"; then
    echo "[ok] marker: persisted=(true|false)"
else
    echo "[FAIL] missing marker: persisted=(true|false)"; fail=$((fail+1))
fi

say "summary"; echo "failed checks: $fail"
[ "$fail" -eq 0 ] && echo "RESULT: PASS — base completion ran local (audit + model line)." || echo "RESULT: FAIL — $fail check(s)."
exit "$fail"
