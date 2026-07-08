#!/usr/bin/env bash
# dashboard-smoke — build the abi CLI and non-interactively smoke the diagnostics dashboard.
#
# `abi dashboard` (== `abi tui` == `abi --tui`) is INTERACTIVE by default (1s
# auto-refresh loop, quits on q/Esc/EOF) and HANGS on a TTY. Feeding a non-TTY
# stdin (`< /dev/null`) forces the one-shot render + exit 0 — the CI-friendly
# smoke that the interactive `run-tui` skill does NOT cover. All output is on
# stderr (capture 2>&1). The non-TTY path should fall back cleanly without a
# `tcgetattr`/errno stack trace; assert the panels and absence of terminal errors.
set -uo pipefail

cd "$(git rev-parse --show-toplevel 2>/dev/null)" 2>/dev/null || {
  echo "RESULT: FAIL (run from inside the abi repo)"; exit 1; }

LOG=$(mktemp /tmp/abi-dashboard-smoke.XXXXXX 2>/dev/null || mktemp -t abi-dashboard-smoke)
trap 'rm -f "$LOG"' EXIT
echo "[1/3] building abi CLI ..."
if ! ./build.sh cli >"$LOG" 2>&1; then
  echo "RESULT: FAIL (build) — see $LOG"; exit 1
fi
BIN=zig-out/bin/abi
[ -x "$BIN" ] || { echo "RESULT: FAIL (no binary at $BIN — build is near-silent, verify with ls)"; exit 1; }

echo "[2/3] rendering dashboard one-shot (stdin=/dev/null forces non-interactive) ..."
OUT="$("$BIN" dashboard </dev/null 2>&1)"; rc=$?

echo "[3/3] asserting exit 0 + all 5 panels ..."
# Strip ANSI so panel headers match cleanly.
PLAIN="$(printf '%s' "$OUT" | sed $'s/\033\[[0-9;]*m//g')"
fail=0
[ "$rc" -eq 0 ] || { echo "  x expected exit 0, got $rc"; fail=1; }
case "$PLAIN" in *"ABI Diagnostics Dashboard"*) : ;; *) echo "  x missing title 'ABI Diagnostics Dashboard'"; fail=1 ;; esac
panels=0
for p in System Plugins "WDBX Storage" Scheduler Memory; do
  case "$PLAIN" in *"$p "*) panels=$((panels+1)) ;; *) echo "  x missing panel: $p" ;; esac
done
[ "$panels" -eq 5 ] || { echo "  x expected 5 panels, rendered $panels"; fail=1; }
case "$PLAIN" in *"unexpected errno"*|*"tcgetattr"*|*"panic"*) echo "  x terminal error leaked into one-shot render"; fail=1 ;; esac

if [ "$fail" -eq 0 ]; then
  echo "RESULT: PASS — dashboard rendered all 5 panels one-shot, exit 0"
  exit 0
fi
echo "RESULT: FAIL"
exit 1
