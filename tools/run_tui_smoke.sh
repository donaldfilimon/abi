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
