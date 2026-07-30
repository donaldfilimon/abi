#!/usr/bin/env bash
# Host launcher for abi-mcp (stdio JSON-RPC). Used by .mcp.json / opencode.json.
# Optional: ABI_MCP_AUTO_BUILD=1 builds via ./tools/cargo.sh when missing.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd -- "$SCRIPT_DIR/.." && pwd)

pick_bin() {
  if [[ -x "$ROOT_DIR/target/release/abi-mcp" ]]; then
    echo "$ROOT_DIR/target/release/abi-mcp"
  elif [[ -x "$ROOT_DIR/target/debug/abi-mcp" ]]; then
    echo "$ROOT_DIR/target/debug/abi-mcp"
  else
    return 1
  fi
}

BIN="$(pick_bin || true)"
if [[ -z "${BIN:-}" ]]; then
  if [[ "${ABI_MCP_AUTO_BUILD:-0}" == "1" ]]; then
    echo "abi-mcp not found; ABI_MCP_AUTO_BUILD=1 — building via ./tools/cargo.sh" >&2
    (cd "$ROOT_DIR" && ./tools/cargo.sh build -p abi-mcp)
    BIN="$(pick_bin || true)"
  fi
fi

if [[ -z "${BIN:-}" || ! -x "$BIN" ]]; then
  echo "abi-mcp not found under target/{release,debug}/" >&2
  echo "Build it first from the repo root:" >&2
  echo "  ./tools/cargo.sh build -p abi-mcp" >&2
  echo "  # or: ABI_MCP_AUTO_BUILD=1 $0 $*" >&2
  exit 127
fi

cd -- "$ROOT_DIR"
exec "$BIN" "${@:-stdio}"
