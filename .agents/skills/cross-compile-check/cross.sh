#!/usr/bin/env bash
# cross-compile-check driver: run the opt-in cross-target compile smoke
# (`zig build cross-smoke` -> tools/cross_smoke.sh), which compiles + links the
# CLI for Linux/Windows/macOS cross targets. Asserts the success marker.
#
# Slow: each target is a fresh cross compile. Pass extra args through to select
# targets, e.g. `cross.sh aarch64-linux-gnu`.
#
# Usage: .agents/skills/cross-compile-check/cross.sh [targets...]
set -uo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../../.." && pwd)
cd "$REPO_ROOT"
fail=0
say() { printf '\n=== %s ===\n' "$*"; }

say "cross-smoke (compile+link CLI for cross targets — slow)"
if [ "$#" -gt 0 ]; then
    out=$(bash tools/cross_smoke.sh "$@" 2>&1); rc=$?
else
    out=$(zig build cross-smoke 2>&1); rc=$?
fi
printf '%s\n' "$out"
[ "$rc" -eq 0 ] || { echo "[FAIL] cross-smoke exit $rc"; fail=$((fail+1)); }
printf '%s' "$out" | grep -qF "all targets compiled + linked" && echo "[ok] all cross targets linked" \
    || { echo "[FAIL] missing success marker"; fail=$((fail+1)); }

say "summary"; echo "failed: $fail"
[ "$fail" -eq 0 ] && echo "RESULT: PASS — CLI cross-compiles for all targets." || echo "RESULT: FAIL — $fail check(s) (see output for the failing target)."
exit "$fail"
