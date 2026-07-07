#!/usr/bin/env bash
# backend-diagnostics driver: build the abi CLI and capture the GPU / accelerator
# / shader / MLIR backend report plus the compute-backend matrix. Asserts the
# stable section markers. Resolves the repo root from its own location.
#
# Usage: .agents/skills/backend-diagnostics/diag.sh
set -uo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../../.." && pwd)
cd "$REPO_ROOT"
ABI="$REPO_ROOT/zig-out/bin/abi"
fail=0
say() { printf '\n=== %s ===\n' "$*"; }
check() { local label="$1" expect="$2"; shift 2; local out; out=$("$@" 2>&1)
    printf '%s\n' "$out"
    printf '%s' "$out" | grep -qF -- "$expect" && echo "[ok] $label" || { echo "[FAIL] $label (missing: $expect)"; fail=$((fail+1)); }; }

say "build cli"
./build.sh cli >/dev/null 2>&1 && echo "[ok] build" || { echo "[FAIL] build"; exit 1; }
[ -x "$ABI" ] || { echo "[FAIL] no binary"; exit 1; }

check "backends report"  "GPU backend report"  "$ABI" backends
check "compute matrix"   "compute backends"    "$ABI" wdbx compute info
check "gpu info"         "GPU backend"          "$ABI" wdbx gpu info

say "summary"; echo "failed: $fail"
[ "$fail" -eq 0 ] && echo "RESULT: PASS — backend diagnostics captured." || echo "RESULT: FAIL — $fail check(s)."
exit "$fail"
