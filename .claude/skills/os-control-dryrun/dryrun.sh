#!/usr/bin/env bash
# os-control-dryrun driver: build the abi CLI and exercise the OS-control policy
# PLANNING path only — `abi agent os dry-run`. This NEVER executes a command;
# it only prints the plan. The execute path (`agent os execute --confirm`) is
# deliberately out of scope for this skill (destructive / requires confirmation).
#
# Usage: .claude/skills/os-control-dryrun/dryrun.sh ["command description"]
set -uo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../../.." && pwd)
cd "$REPO_ROOT"
ABI="$REPO_ROOT/zig-out/bin/abi"
PLAN="${1:-list current directory}"
fail=0
say() { printf '\n=== %s ===\n' "$*"; }

say "build cli"
./build.sh cli >/dev/null 2>&1 && echo "[ok] build" || { echo "[FAIL] build"; exit 1; }
[ -x "$ABI" ] || { echo "[FAIL] no binary"; exit 1; }

say "abi agent os dry-run (planning only — no execution)"
out=$("$ABI" agent os dry-run "$PLAN" 2>&1); rc=$?
printf '%s\n' "$out"
[ "$rc" -eq 0 ] || { echo "[FAIL] dry-run exit $rc"; fail=$((fail+1)); }
grep -qF "dry-run:" <<<"$out" && echo "[ok] plan emitted (no execution)" \
    || { echo "[FAIL] missing 'dry-run:' marker"; fail=$((fail+1)); }

# Safety assertion: execute WITHOUT --confirm must be refused with usage (exit 2),
# never run. This guards the confirm gate without ever executing anything.
say "safety: execute without --confirm is refused"
"$ABI" agent os execute "list current directory" >/dev/null 2>&1; rc=$?
if [ "$rc" -eq 2 ]; then echo "[ok] execute without --confirm -> usage (exit 2)"; \
    else echo "[FAIL] execute without --confirm returned $rc (expected 2)"; fail=$((fail+1)); fi

say "summary"; echo "failed: $fail"
[ "$fail" -eq 0 ] && echo "RESULT: PASS — OS-control dry-run + confirm-gate verified." || echo "RESULT: FAIL — $fail check(s)."
exit "$fail"
