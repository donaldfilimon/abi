#!/usr/bin/env bash
# mcp-smoke — build abi-mcp and assert the frozen 12-tool `tools/list` contract.
#
# Sends one JSON-RPC `tools/list` over the stdio transport (newline-delimited
# JSON; closing stdin via pipe EOF exits the server — no `timeout` needed, which
# matters because macOS has no `timeout`). Evidence is the `RESULT:` line.
# Fully local; no network dispatch (the loopback HTTP listener that abi-mcp
# always starts on :8080 only prints an info line to stderr, which we drop).
set -uo pipefail

cd "$(git rev-parse --show-toplevel 2>/dev/null)" 2>/dev/null || {
  echo "RESULT: FAIL (run from inside the abi repo)"; exit 1; }

# Frozen MCP tool contract (see CLAUDE.md / tests/contracts/surface.zig).
EXPECTED='ai_run
ai_complete
ai_learn
ai_train
wdbx_query
scheduler_stats
scheduler_info
connector_test
gpu_status
plugin_list
wdbx_stats
plugin_run'

LOG=$(mktemp /tmp/abi-mcp-smoke.XXXXXX 2>/dev/null || mktemp -t abi-mcp-smoke)
trap 'rm -f "$LOG"' EXIT
echo "[1/3] building abi-mcp ..."
if ! ./build.sh mcp >"$LOG" 2>&1; then
  echo "RESULT: FAIL (build) — see $LOG"; exit 1
fi
BIN=zig-out/bin/abi-mcp
[ -x "$BIN" ] || { echo "RESULT: FAIL (no binary at $BIN — build succeeds silently, verify with ls)"; exit 1; }

echo "[2/3] sending tools/list over stdio ..."
REQ='{"jsonrpc":"2.0","id":1,"method":"tools/list"}'
# `initialize` is NOT a prerequisite — a bare tools/list works. Pipe EOF -> clean exit.
RESP="$(printf '%s\n' "$REQ" | "$BIN" 2>/dev/null)"
[ -n "$RESP" ] || { echo "RESULT: FAIL (empty response on stdout)"; exit 1; }

echo "[3/3] extracting tool names ..."
if command -v jq >/dev/null 2>&1; then
  NAMES="$(printf '%s' "$RESP" | jq -r '.result.tools[].name' 2>/dev/null)"
else
  # Fallback: match name *values* ("name":"x"), NOT bare "name": — the latter also
  # hits plugin_run's inputSchema `name` property and miscounts 13 (false contract break).
  NAMES="$(printf '%s' "$RESP" | grep -oE '"name":"[^"]+"' | sed -E 's/"name":"([^"]+)"/\1/')"
fi

count="$(printf '%s\n' "$NAMES" | grep -c .)"
fail=0
if [ "$count" -ne 12 ]; then
  echo "  x expected 12 tools, got $count"; fail=1
fi
if [ "$(printf '%s\n' "$EXPECTED" | sort)" != "$(printf '%s\n' "$NAMES" | sort)" ]; then
  echo "  x tool set mismatch (expected < vs got >):"
  diff <(printf '%s\n' "$EXPECTED" | sort) <(printf '%s\n' "$NAMES" | sort) | sed 's/^/    /'
  fail=1
fi

if [ "$fail" -eq 0 ]; then
  echo "RESULT: PASS — 12/12 frozen MCP tools present"
  printf '  %s\n' "$NAMES"
  exit 0
fi
echo "RESULT: FAIL"
exit 1
