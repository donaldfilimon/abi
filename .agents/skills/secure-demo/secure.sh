#!/usr/bin/env bash
# secure-demo driver: build the abi CLI and run the WDBX "secure" demo —
# int8 embedding compression, additive homomorphic sum, and DGHV somewhat-
# homomorphic eval. Asserts each section's match/ratio marker.
#
# Usage: .agents/skills/secure-demo/secure.sh
set -uo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../../.." && pwd)
cd "$REPO_ROOT"
ABI="$REPO_ROOT/zig-out/bin/abi"
fail=0
say() { printf '\n=== %s ===\n' "$*"; }

say "build cli"
./build.sh cli >/dev/null 2>&1 && echo "[ok] build" || { echo "[FAIL] build"; exit 1; }
[ -x "$ABI" ] || { echo "[FAIL] no binary"; exit 1; }

say "abi wdbx secure demo"
out=$("$ABI" wdbx secure demo 2>&1); rc=$?
printf '%s\n' "$out"
[ "$rc" -eq 0 ] || { echo "[FAIL] secure demo exit $rc"; fail=$((fail+1)); }
for marker in "compression:" "additive HE:" "homomorphic eval:" "match=true"; do
    printf '%s' "$out" | grep -qF -- "$marker" && echo "[ok] marker: $marker" \
        || { echo "[FAIL] missing marker: $marker"; fail=$((fail+1)); }
done

say "summary"; echo "failed: $fail"
[ "$fail" -eq 0 ] && echo "RESULT: PASS — secure (compression + HE) demo ran." || echo "RESULT: FAIL — $fail check(s)."
exit "$fail"
