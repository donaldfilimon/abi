#!/usr/bin/env bash
# nn-demo driver: build the abi CLI and exercise the miniature character-level
# neural-net demo — `nn train` (real manual-backprop char-LM) and `nn sample`
# (train-then-generate). This is a DEMO trainer, not a production/LLM/distributed
# trainer. Asserts exit codes + output markers. Resolves repo root from own path.
#
# Usage: .agents/skills/nn-demo/nn.sh ["training corpus"] [seed-char] [n]
set -uo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../../.." && pwd)
cd "$REPO_ROOT"
ABI="$REPO_ROOT/zig-out/bin/abi"
CORPUS="${1:-the quick brown fox jumps over the lazy dog the quick brown fox}"
SEED="${2:-t}"
N="${3:-20}"
fail=0
say() { printf '\n=== %s ===\n' "$*"; }
markers() { local out="$1"; shift; for m in "$@"; do
    printf '%s' "$out" | grep -qF -- "$m" && echo "[ok] marker: $m" \
        || { echo "[FAIL] missing marker: $m"; fail=$((fail+1)); }; done; }

say "build cli"
./build.sh cli >/dev/null 2>&1 && echo "[ok] build" || { echo "[FAIL] build"; exit 1; }
[ -x "$ABI" ] || { echo "[FAIL] $ABI not produced"; exit 1; }

say "abi nn train"
out=$("$ABI" nn train "$CORPUS" 2>&1); rc=$?
printf '%s\n' "$out"
[ "$rc" -eq 0 ] || { echo "[FAIL] nn train exit $rc"; fail=$((fail+1)); }
markers "$out" "nn train:" "final_loss=" "steps="

say "abi nn sample"
out=$("$ABI" nn sample --text "$CORPUS" --seed "$SEED" --n "$N" 2>&1); rc=$?
printf '%s\n' "$out"
[ "$rc" -eq 0 ] || { echo "[FAIL] nn sample exit $rc"; fail=$((fail+1)); }
markers "$out" "nn sample:"

say "summary"; echo "failed checks: $fail"
[ "$fail" -eq 0 ] && echo "RESULT: PASS — nn demo trains + samples." || echo "RESULT: FAIL — $fail check(s)."
exit "$fail"
