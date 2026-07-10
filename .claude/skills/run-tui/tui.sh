#!/usr/bin/env bash
# run-tui driver: build the abi CLI and drive the interactive diagnostics
# dashboard through a tmux pty. Piped stdin uses the one-shot fallback; tmux gives
# the command a real terminal so we can exercise the interactive path, capture the
# rendered pane, send the quit key, and tear the session down.
#
# Usage: .claude/skills/run-tui/tui.sh [command]   # command: dashboard (default) | tui
set -uo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../../.." && pwd)
cd "$REPO_ROOT"
export PATH="$PATH:/opt/homebrew/bin"   # append (not prepend!) so brew tmux is found without shadowing the zvm zig with brew's 0.16
ABI="$REPO_ROOT/zig-out/bin/abi"
CMD="${1:-dashboard}"
SESSION="abi-tui-skill-$$"
MARKER="ABI Diagnostics Dashboard"
fail=0
say() { printf '\n=== %s ===\n' "$*"; }
cleanup() { tmux kill-session -t "$SESSION" 2>/dev/null || true; }
trap cleanup EXIT

command -v tmux >/dev/null || { echo "[FAIL] tmux not installed (brew install tmux)"; exit 1; }

say "build cli"
./build.sh cli >/dev/null 2>&1 && echo "[ok] build" || { echo "[FAIL] build"; exit 1; }
[ -x "$ABI" ] || { echo "[FAIL] $ABI not produced"; exit 1; }

say "launch 'abi $CMD' under tmux pty"
tmux kill-session -t "$SESSION" 2>/dev/null || true
tmux new-session -d -s "$SESSION" -x 200 -y 50 "$ABI $CMD"
sleep 2.5
pane=$(tmux capture-pane -pt "$SESSION" 2>/dev/null || true)
printf '%s\n' "$pane" | sed -n '1,12p'

grep -qF -- "$MARKER" <<<"$pane" && echo "[ok] dashboard painted" \
    || { echo "[FAIL] dashboard did not paint (missing: $MARKER)"; fail=$((fail+1)); }
grep -qiE "errno 19|tcgetattr|panic|unreachable" <<<"$pane" \
    && { echo "[FAIL] tty error / panic in pane"; fail=$((fail+1)); } || echo "[ok] no tty error"

say "send quit key (q) + teardown"
tmux send-keys -t "$SESSION" q 2>/dev/null || true
sleep 1
tmux kill-session -t "$SESSION" 2>/dev/null || true
tmux has-session -t "$SESSION" 2>/dev/null && { echo "[FAIL] session survived teardown"; fail=$((fail+1)); } || echo "[ok] session torn down"

say "summary"; echo "failed checks: $fail"
[ "$fail" -eq 0 ] && echo "RESULT: PASS — diagnostics dashboard painted under pty." || echo "RESULT: FAIL — $fail check(s)."
exit "$fail"
