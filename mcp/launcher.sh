#!/usr/bin/env bash
# Host launcher for abi-mcp (stdio JSON-RPC). Used by .mcp.json / opencode.json.
# By default builds nothing — binaries come from ./build.sh mcp (or zig build).
# Optional: ABI_MCP_AUTO_BUILD=1 runs ./build.sh mcp when the binary is missing.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd -- "$SCRIPT_DIR/.." && pwd)
BIN="$ROOT_DIR/zig-out/bin/abi-mcp"

if [[ ! -x "$BIN" ]]; then
  if [[ "${ABI_MCP_AUTO_BUILD:-0}" == "1" ]]; then
    echo "abi-mcp not found at: $BIN; ABI_MCP_AUTO_BUILD=1 — building via ./build.sh mcp" >&2
    (cd "$ROOT_DIR" && ./build.sh mcp)
  fi
fi

if [[ ! -x "$BIN" ]]; then
  echo "abi-mcp not found at: $BIN" >&2
  echo "Build it first from the repo root:" >&2
  echo "  ./build.sh mcp" >&2
  echo "  # or: zig build" >&2
  echo "  # or: ABI_MCP_AUTO_BUILD=1 $0 $*" >&2
  exit 127
fi

# Default transport is stdio when no args (MCP host convention).
exec "$BIN" "${@:-stdio}"
