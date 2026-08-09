#!/usr/bin/env bash
# run-tui driver: build the abi CLI and drive the interactive diagnostics
# dashboard through a tmux pty. Piped stdin uses the one-shot fallback; tmux gives
# the command a real terminal so we can exercise the interactive path, capture the
# rendered pane, send the quit key, and tear the session down.
#
# Usage: .agents/skills/run-tui/tui.sh [command]   # command: dashboard (default) | tui
set -uo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../../.." && pwd)
cd "$REPO_ROOT"
export PATH="$PATH:/opt/homebrew/bin"   # append so brew tmux is found; keep rustup/cargo.sh toolchain first
ABI="$REPO_ROOT/target/debug/abi"
CMD="${1:-dashboard}"
SESSION="abi-tui-skill-$$"
MARKER="ABI Diagnostics Dashboard"
CAPTURE_LOG=$(mktemp "${TMPDIR:-/tmp}/abi-tui-capture.XXXXXX")
fail=0
say() { printf '\n=== %s ===\n' "$*"; }
cleanup() {
    tmux kill-session -t "$SESSION" 2>/dev/null || true
    rm -f -- "$CAPTURE_LOG"
}
trap cleanup EXIT

command -v tmux >/dev/null || { echo "[FAIL] tmux not installed (brew install tmux)"; exit 1; }

say "build cli"
./tools/cargo.sh build -p abi-cli >/dev/null 2>&1 && echo "[ok] build" || { echo "[FAIL] build"; exit 1; }
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

say "send SGR mouse press and verify pane focus"
tmux set-option -t "$SESSION" remain-on-exit on >/dev/null
tmux pipe-pane -o -t "$SESSION" "cat > '$CAPTURE_LOG'"
tmux send-keys -t "$SESSION" -l $'\033[<0;2;10M'
sleep 1
mouse_pane=$(tmux capture-pane -ept "$SESSION" 2>/dev/null || true)
plugin_line=$(grep -F -- "┌ Plugins" <<<"$mouse_pane" | head -1 || true)
[[ "$plugin_line" == *$'\033'* ]] && echo "[ok] mouse selected Plugins pane" \
    || { echo "[FAIL] mouse did not focus Plugins pane"; fail=$((fail+1)); }

say "send quit key (q), verify capture cleanup, and teardown"
tmux send-keys -t "$SESSION" q 2>/dev/null || true
sleep 1
pane_dead=$(tmux display-message -p -t "$SESSION" '#{pane_dead}' 2>/dev/null || true)
[ "$pane_dead" = "1" ] && echo "[ok] dashboard exited" \
    || { echo "[FAIL] dashboard did not exit after q"; fail=$((fail+1)); }
LC_ALL=C grep -qF -- $'\033[?1006l\033[?1000l' "$CAPTURE_LOG" \
    && echo "[ok] mouse capture disabled before exit" \
    || { echo "[FAIL] mouse capture disable sequence missing"; fail=$((fail+1)); }
tmux kill-session -t "$SESSION" 2>/dev/null || true
tmux has-session -t "$SESSION" 2>/dev/null && { echo "[FAIL] session survived teardown"; fail=$((fail+1)); } || echo "[ok] session torn down"

say "launch agent REPL and verify single-stream exit"
tmux new-session -d -s "$SESSION" -x 200 -y 50 "$ABI agent tui"
tmux set-option -t "$SESSION" remain-on-exit on >/dev/null
sleep 1
tmux send-keys -t "$SESSION" /status Enter
sleep 1
tmux send-keys -t "$SESSION" /quit Enter
sleep 1
repl_pane=$(tmux capture-pane -pS - -t "$SESSION" 2>/dev/null || true)
status_count=$(grep -cF -- "status: session_id=" <<<"$repl_pane" || true)
[ "$status_count" -eq 1 ] && echo "[ok] status streamed exactly once" \
    || { echo "[FAIL] status replay count=$status_count"; fail=$((fail+1)); }
grep -qF -- "ABI agent line-mode ready" <<<"$repl_pane" \
    && { echo "[FAIL] interactive exit printed line-mode fallback"; fail=$((fail+1)); } \
    || echo "[ok] no false line-mode fallback"
tmux kill-session -t "$SESSION" 2>/dev/null || true

say "summary"; echo "failed checks: $fail"
[ "$fail" -eq 0 ] && echo "RESULT: PASS — dashboard keyboard/mouse + cleanup and agent REPL verified under pty." || echo "RESULT: FAIL — $fail check(s)."
exit "$fail"
