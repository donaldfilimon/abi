#!/usr/bin/env bash
# mcp-smoke — build abi-mcp and assert the frozen 12-tool `tools/list` contract.
#
# Sends one JSON-RPC `tools/list` over the stdio transport (newline-delimited
# JSON; closing stdin via pipe EOF exits the server — no `timeout` needed, which
# matters because macOS has no `timeout`). Evidence is the `RESULT:` line.
# Fully local; no network dispatch (the loopback HTTP listener that abi-mcp
# always starts on :8080 only prints an info line to stderr, which we drop).
set -uo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd) || exit 1
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../../.." && pwd) || exit 1
cd -- "$REPO_ROOT" || exit 1
export ABI_WDBX_PATH=:memory:
export ABI_WDBX_PERSIST=0

# Frozen MCP tool contract (see AGENTS.md / tests/golden/mcp-tools-list.json).
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
if ! ./tools/cargo.sh build --locked --release -p abi-mcp --target-dir "$REPO_ROOT/target" >"$LOG" 2>&1; then
  cat "$LOG" >&2
  echo "RESULT: FAIL (build)"; exit 1
fi
BIN=target/release/abi-mcp
[ -x "$BIN" ] || { echo "RESULT: FAIL (no binary at $BIN — build succeeds silently, verify with ls)"; exit 1; }

echo "[2/3] sending tools/list over stdio ..."
REQ='{"jsonrpc":"2.0","id":1,"method":"tools/list"}'
# `initialize` is NOT a prerequisite — a bare tools/list works. Pipe EOF -> clean exit.
if ! RESP="$(printf '%s\n' "$REQ" | ./mcp/launcher.sh stdio 2>"$LOG")"; then
  cat "$LOG" >&2
  echo "RESULT: FAIL (MCP process)"; exit 1
fi
[ -n "$RESP" ] || { echo "RESULT: FAIL (empty response on stdout)"; exit 1; }

echo "[3/3] extracting tool names ..."
if ! NAMES="$(printf '%s' "$RESP" | python3 -c '
import json
import sys
try:
    response = json.load(sys.stdin)
    if (not isinstance(response, dict) or response.get("jsonrpc") != "2.0"
            or type(response.get("id")) is not int or response["id"] != 1
            or "error" in response):
        raise ValueError("unexpected JSON-RPC response envelope")
    tools = response["result"]["tools"]
    if not isinstance(tools, list):
        raise ValueError("tools must be an array")
    names = [tool["name"] for tool in tools]
    if any(not isinstance(name, str) or not name or "\n" in name for name in names):
        raise ValueError("invalid tool name")
    print("\n".join(names))
except (ValueError, KeyError, TypeError) as error:
    print("invalid tools/list response:", error, file=sys.stderr)
    sys.exit(1)
')"; then
  echo "RESULT: FAIL (invalid tools/list response)"; exit 1
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
