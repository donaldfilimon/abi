#!/usr/bin/env bash
set -euo pipefail

bin="zig-out/bin/abi"
if [ ! -x "$bin" ]; then
  echo "tui smoke: missing executable at $bin" >&2
  exit 1
fi

out=$("$bin" dashboard </dev/null 2>&1)
plain=$(printf '%s' "$out" | sed $'s/\033\[[0-9;]*m//g')

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
  *"unexpected errno"*|*"tcgetattr"*|*"panic"*)
    echo "tui smoke: terminal error leaked into one-shot render" >&2
    exit 1
    ;;
esac

agent_out=$(ABI_WDBX_PATH=:memory: "$bin" agent tui 2>&1 <<'EOF'
/help
/quit
EOF
)

case "$agent_out" in
  *"Commands:"*"/help"*"/quit"*) ;;
  *) echo "tui smoke: missing agent REPL help output" >&2; exit 1 ;;
esac

case "$agent_out" in
  *"interactive REPL failed"*|*"FeatureDisabled"*|*"panic"*)
    echo "tui smoke: agent REPL reported an unexpected failure" >&2
    exit 1
    ;;
esac

echo "tui smoke: ok"
