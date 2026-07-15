#!/usr/bin/env bash
set -euo pipefail

bin="zig-out/bin/abi"
if [ ! -x "$bin" ]; then
  echo "tui smoke: missing executable at $bin" >&2
  exit 1
fi

out=$("$bin" dashboard </dev/null 2>&1)
plain=$(printf '%s' "$out" | sed $'s/\033\[[0-9;]*m//g')

case "$out" in
  *$'\033[?1049h'*|*$'\033[?1049l'*)
    echo "tui smoke: one-shot dashboard emitted alternate-screen control" >&2
    exit 1
    ;;
esac

case "$out" in
  *$'\033[H'*|*$'\033[0J'*)
    echo "tui smoke: one-shot dashboard emitted interactive redraw control" >&2
    exit 1
    ;;
esac

case "$plain" in
  *"ABI Diagnostics Dashboard"*) ;;
  *) echo "tui smoke: missing dashboard title" >&2; exit 1 ;;
esac

for panel in System Plugins "WDBX Storage" Scheduler Memory; do
  case "$plain" in
    *"$panel"*) ;;
    *) echo "tui smoke: missing panel: $panel" >&2; exit 1 ;;
  esac
done

case "$plain" in
  *"[1-5] Panes"*) ;;
  *) echo "tui smoke: missing pane navigation footer" >&2; exit 1 ;;
esac

memory_out=$("$bin" dashboard --pane memory </dev/null 2>&1)
case "$memory_out" in
  *$'\033[7m\033[1;31m'*"Memory"*) ;;
  *) echo "tui smoke: --pane memory did not highlight memory pane" >&2; exit 1 ;;
esac

plain_out=$("$bin" dashboard --plain </dev/null 2>&1)
case "$plain_out" in
  *$'\033['*)
    echo "tui smoke: --plain emitted ANSI styling" >&2
    exit 1
    ;;
esac
case "$plain_out" in
  *"ABI Diagnostics Dashboard"*"[1-5] Panes"*) ;;
  *) echo "tui smoke: --plain missing dashboard content" >&2; exit 1 ;;
esac

compact_out=$("$bin" dashboard --compact --pane scheduler --plain </dev/null 2>&1)
case "$compact_out" in
  *"ABI Diagnostics Dashboard"*"Scheduler"*) ;;
  *) echo "tui smoke: --compact missing selected scheduler pane" >&2; exit 1 ;;
esac
case "$compact_out" in
  *"WDBX Storage"*|*"Memory"*)
    echo "tui smoke: --compact rendered non-selected panes" >&2
    echo "$compact_out" >&2
    exit 1
    ;;
esac

shortcut_out=$("$bin" --tui --compact --pane scheduler --plain </dev/null 2>&1)
case "$shortcut_out" in
  *"ABI Diagnostics Dashboard"*"Scheduler"*) ;;
  *) echo "tui smoke: --tui shortcut did not route dashboard flags" >&2; exit 1 ;;
esac
case "$shortcut_out" in
  *"WDBX Storage"*|*"Memory"*)
    echo "tui smoke: --tui shortcut compact render included non-selected panes" >&2
    echo "$shortcut_out" >&2
    exit 1
    ;;
esac

json_out=$("$bin" dashboard --json --pane scheduler --interval 250 </dev/null 2>&1)
printf '%s' "$json_out" | python3 -c 'import json,sys; data=json.load(sys.stdin); layout=data["layout"]; assert data["type"] == "abi.dashboard"; assert data["selected_pane"] == "scheduler"; assert data["refresh_interval_ms"] == 250; assert layout["format"] == "json"; assert layout["compact"] is False; assert "scheduler" in layout["visible_panes"]; assert any(p["name"]=="scheduler" and p["hotkey"]=="4" and p["selected"] for p in layout["panes"]); assert "scheduler" in data; assert "plugins" in data'
case "$json_out" in
  *"ABI Diagnostics Dashboard"*|*$'\033['*)
    echo "tui smoke: --json emitted text dashboard or ANSI controls" >&2
    echo "$json_out" >&2
    exit 1
    ;;
esac

json_compact_out=$("$bin" dashboard --json --compact --pane scheduler --plain --interval 250 </dev/null 2>&1)
printf '%s' "$json_compact_out" | python3 -c 'import json,sys; data=json.load(sys.stdin); layout=data["layout"]; assert data["selected_pane"] == "scheduler"; assert layout["format"] == "json"; assert layout["compact"] is True; assert layout["color"] is False; assert layout["visible_panes"] == ["scheduler"]; assert any(p["name"]=="memory" and not p["visible"] for p in layout["panes"])'

panes_out=$("$bin" dashboard --list-panes --pane scheduler </dev/null 2>&1)
case "$panes_out" in
  *"Dashboard panes:"*"* scheduler (Scheduler) hotkey=4"*) ;;
  *) echo "tui smoke: --list-panes missing pane metadata" >&2; echo "$panes_out" >&2; exit 1 ;;
esac
case "$json_compact_out" in
  *"ABI Diagnostics Dashboard"*|*$'\033['*)
    echo "tui smoke: compact --json emitted text dashboard or ANSI controls" >&2
    echo "$json_compact_out" >&2
    exit 1
    ;;
esac

once_out=$(printf '' | script -q /dev/null "$bin" dashboard --once --interval 250 2>&1 || true)
case "$once_out" in
  *"ABI Diagnostics Dashboard"*"[1-5] Panes"*|"ABI Diagnostics Dashboard"*) ;;
  *) echo "tui smoke: --once under pty missing dashboard content" >&2; echo "$once_out" >&2; exit 1 ;;
esac
case "$once_out" in
  *"live snapshot every 250ms"*) ;;
  *) echo "tui smoke: --interval did not update footer" >&2; echo "$once_out" >&2; exit 1 ;;
esac
case "$once_out" in
  *$'\033[?1049h'*|*$'\033[?1049l'*|*$'\033[H'*|*$'\033[0J'*)
    echo "tui smoke: --once under pty emitted interactive screen controls" >&2
    echo "$once_out" >&2
    exit 1
    ;;
esac

footer_count=$(printf '%s' "$plain" | grep -F -c "[q/Esc]")
if [ "$footer_count" -ne 1 ]; then
  echo "tui smoke: expected one dashboard footer, got $footer_count" >&2
  exit 1
fi

case "$plain" in
  *"unexpected errno"*|*"tcgetattr"*|*"panic"*)
    echo "tui smoke: terminal error leaked into one-shot render" >&2
    exit 1
    ;;
esac

agent_out=$(ABI_WDBX_PATH=:memory: "$bin" agent tui 2>&1 <<'EOF'
/help
/model abi-local
/status
/model two words
/status
/profile
/history
/reset
/bogus
/quit
EOF
)

case "$agent_out" in
  *"Commands:"*"/help"*"/quit"*) ;;
  *) echo "tui smoke: missing agent REPL help output" >&2; exit 1 ;;
esac

for needle in "model set to abi-local" "status: session_id=" "model=abi-local provider=local" "model id must be printable non-whitespace ASCII" "profile:" "history:" "session reset" "unknown command: /bogus"; do
  case "$agent_out" in
    *"$needle"*) ;;
    *) echo "tui smoke: missing agent REPL output: $needle" >&2; exit 1 ;;
  esac
done

case "$agent_out" in
  *"model set to two"*)
    echo "tui smoke: invalid model changed session state" >&2
    exit 1
    ;;
esac

case "$agent_out" in
  *"interactive REPL failed"*|*"FeatureDisabled"*|*"panic"*)
    echo "tui smoke: agent REPL reported an unexpected failure" >&2
    exit 1
    ;;
esac

# The heredoc above deliberately covers the stable non-TTY fallback. When tmux
# is available, also drive the raw-mode editor through a real PTY: cursor edit,
# tab completion, ambiguous command discovery, and history recall all need a
# terminal to exercise the byte-oriented input path.
if command -v tmux >/dev/null 2>&1; then
  agent_session="abi-agent-tui-smoke-$$"
  agent_pty_bin="$PWD/$bin"
  tmux kill-session -t "$agent_session" 2>/dev/null || true
  tmux new-session -d -s "$agent_session" -x 120 -y 40 "env ABI_WDBX_PATH=:memory: '$agent_pty_bin' agent tui"
  cleanup_agent_session() {
    tmux kill-session -t "$agent_session" 2>/dev/null || true
  }
  trap cleanup_agent_session EXIT

  sleep 1
  tmux send-keys -t "$agent_session" -l '/helpx'
  tmux send-keys -t "$agent_session" Left Delete Enter
  tmux send-keys -t "$agent_session" -l '/mod'
  tmux send-keys -t "$agent_session" Tab Enter
  tmux send-keys -t "$agent_session" -l '/s'
  tmux send-keys -t "$agent_session" Tab Enter
  tmux send-keys -t "$agent_session" -l '/model abi-local'
  tmux send-keys -t "$agent_session" Enter Up Enter
  sleep 1
  agent_pty_out=$(tmux capture-pane -p -S -200 -t "$agent_session" 2>&1 || true)

  case "$agent_pty_out" in
    *"Commands:"*"usage: /model <id>"*"matches: /status /sync-clis"*) ;;
    *) echo "tui smoke: raw agent editor missed edit or tab-completion output" >&2; echo "$agent_pty_out" >&2; exit 1 ;;
  esac
  model_set_count=$(printf '%s' "$agent_pty_out" | grep -F -c "model set to abi-local")
  if [ "$model_set_count" -lt 2 ]; then
    echo "tui smoke: raw agent editor did not recall history" >&2
    echo "$agent_pty_out" >&2
    exit 1
  fi
  case "$agent_pty_out" in
    *"interactive REPL failed"*|*"FeatureDisabled"*|*"panic"*)
      echo "tui smoke: raw agent editor reported an unexpected failure" >&2
      echo "$agent_pty_out" >&2
      exit 1
      ;;
  esac

  tmux send-keys -t "$agent_session" -l '/quit'
  tmux send-keys -t "$agent_session" Enter
  sleep 1
  tmux has-session -t "$agent_session" 2>/dev/null && {
    echo "tui smoke: raw agent REPL did not exit after /quit" >&2
    exit 1
  }
  trap - EXIT
else
  echo "tui smoke: tmux unavailable; skipped raw agent PTY editor check" >&2
fi

echo "tui smoke: ok"
