#!/usr/bin/env bash
# agent-plan-train driver: build the abi CLI and exercise the non-interactive
# agent surfaces — `agent plan` (dry-run planning through the profile router)
# and `agent train <profile>` (real scheduler-backed training that records
# metadata in WDBX). Skips `agent tui` (interactive REPL; non-TTY line mode is
# smoke-testable separately) and `agent os` (covered by os-control-dryrun).
# Asserts exit codes + markers.
#
# Usage: .claude/skills/agent-plan-train/agent.sh ["plan text"] [profile]
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
markers "$out" "training executed via scheduler" "metadata recorded in wdbx"

say "summary"; echo "failed checks: $fail"
[ "$fail" -eq 0 ] && echo "RESULT: PASS — agent plan + train verified." || echo "RESULT: FAIL — $fail check(s)."
exit "$fail"
