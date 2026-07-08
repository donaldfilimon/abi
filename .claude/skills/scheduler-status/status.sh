#!/usr/bin/env bash
# scheduler-status — build the abi CLI and smoke-test the one-shot scheduler surface.
#
# Runs `abi scheduler status` (the ONLY valid subcommand — not stats/info, which
# are the MCP tool names, not the CLI word), asserts the probe task ran to
# completion, and asserts the always-on Prometheus telemetry block is present.
# Output is on stderr, so we capture 2>&1. Self-terminating one-shot: no server,
# no state, no network, no `timeout` needed (macOS lacks it anyway).
set -uo pipefail

cd "$(git rev-parse --show-toplevel 2>/dev/null)" 2>/dev/null || {
  echo "RESULT: FAIL (run from inside the abi repo)"; exit 1; }

LOG=$(mktemp /tmp/abi-scheduler-status.XXXXXX 2>/dev/null || mktemp -t abi-scheduler-status)
trap 'rm -f "$LOG"' EXIT
echo "[1/3] building abi CLI ..."
if ! ./build.sh cli >"$LOG" 2>&1; then
  echo "RESULT: FAIL (build) — see $LOG"; exit 1
fi
BIN=zig-out/bin/abi
[ -x "$BIN" ] || { echo "RESULT: FAIL (no binary at $BIN — build is near-silent, verify with ls)"; exit 1; }

echo "[2/3] running 'abi scheduler status' (output is on stderr) ..."
OUT="$("$BIN" scheduler status 2>&1)"

echo "[3/3] asserting probe task + telemetry ..."
fail=0
check() { # <needle> <label>
  case "$OUT" in
    *"$1"*) : ;;
    *) echo "  x missing: $2 ('$1')"; fail=1 ;;
  esac
}
check "completed=1"                 "probe task completed"
check "total_tasks=1"               "one task submitted"
check "failed=0"                    "no failures"
check "scheduler_tasks_completed 1" "prometheus telemetry block"

# Negative: the only valid subcommand is `status`. stats/info (the MCP names) must be rejected.
if "$BIN" scheduler stats >/dev/null 2>&1; then
  echo "  x 'scheduler stats' should be rejected (only 'status' is valid) but it succeeded"; fail=1
fi

if [ "$fail" -eq 0 ]; then
  echo "RESULT: PASS — scheduler ran the probe task to completion and emitted telemetry"
  printf '%s\n' "$OUT" | sed 's/^/  /'
  exit 0
fi
echo "RESULT: FAIL"
printf '%s\n' "$OUT" | sed 's/^/  /'
exit 1
