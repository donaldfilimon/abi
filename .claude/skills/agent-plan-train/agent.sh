#!/usr/bin/env bash
# agent-plan-train driver: non-interactive agent surfaces — plan, train, multi,
# spawn, browser (claim-honest). Skips tui (interactive) and os (os-control-dryrun).
#
# Usage: .agents/skills/agent-plan-train/agent.sh ["plan text"] [profile]
#   profile: abbey | aviva | abi | all   (default: abbey)
set -uo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../../.." && pwd)
cd "$REPO_ROOT"
ABI="$REPO_ROOT/zig-out/bin/abi"
PLAN="${1:-summarize the scheduler status}"
PROFILE="${2:-abbey}"
fail=0
say() { printf '\n=== %s ===\n' "$*"; }
markers() { local out="$1"; shift; for m in "$@"; do
    grep -qF -- "$m" <<<"$out" && echo "[ok] marker: $m" \
        || { echo "[FAIL] missing marker: $m"; fail=$((fail+1)); }; done; }

say "build cli"
./build.sh cli >/dev/null 2>&1 && echo "[ok] build" || { echo "[FAIL] build"; exit 1; }
[ -x "$ABI" ] || { echo "[FAIL] $ABI not produced"; exit 1; }

say "abi agent plan (dry-run)"
out=$("$ABI" agent plan "$PLAN" 2>&1); rc=$?
printf '%s\n' "$out"
[ "$rc" -eq 0 ] || { echo "[FAIL] agent plan exit $rc"; fail=$((fail+1)); }
markers "$out" "agent=cli-agent" "mode=dry-run" "selected_profile=" "response="

say "abi agent train $PROFILE"
out=$("$ABI" agent train "$PROFILE" 2>&1); rc=$?
printf '%s\n' "$out"
[ "$rc" -eq 0 ] || { echo "[FAIL] agent train exit $rc"; fail=$((fail+1)); }
markers "$out" "training executed via scheduler" "recorded in wdbx"

say "abi agent multi"
out=$("$ABI" agent multi "skill multi smoke" 2>&1); rc=$?
printf '%s\n' "$out"
[ "$rc" -eq 0 ] || { echo "[FAIL] agent multi exit $rc"; fail=$((fail+1)); }
markers "$out" "MULTI-AGENT RESULTS"

say "abi agent spawn"
out=$("$ABI" agent spawn "skill spawn smoke" 2>&1); rc=$?
printf '%s\n' "$out"
[ "$rc" -eq 0 ] || { echo "[FAIL] agent spawn exit $rc"; fail=$((fail+1)); }
markers "$out" "CUSTOM MULTI-AGENT RESULTS"

say "abi agent browser (dry-run)"
out=$("$ABI" agent browser "skill browser smoke" 2>&1); rc=$?
printf '%s\n' "$out"
[ "$rc" -eq 0 ] || { echo "[FAIL] agent browser exit $rc"; fail=$((fail+1)); }
markers "$out" "embedded_browser=false" "delegation_hint=external-mcp-playwright"

say "abi agent browser --execute without --confirm (expect exit 2)"
out=$("$ABI" agent browser --execute "skill" 2>&1); rc=$?
printf '%s\n' "$out"
[ "$rc" -eq 2 ] || { echo "[FAIL] browser execute gate expected exit 2 got $rc"; fail=$((fail+1)); }

say "summary"; echo "failed checks: $fail"
[ "$fail" -eq 0 ] && echo "RESULT: PASS — agent plan/train/multi/spawn/browser verified." || echo "RESULT: FAIL — $fail check(s)."
exit "$fail"
