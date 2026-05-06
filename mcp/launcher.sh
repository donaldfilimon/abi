#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
OS="$(uname -s)"
case "$OS" in
  Linux|Darwin) BIN="$ROOT_DIR/zig-out/bin/abi-mcp" ;;
  MINGW*|MSYS*|CYGWIN*) BIN="$ROOT_DIR/zig-out/bin/abi-mcp.exe" ;;
  *) echo "Unsupported OS: $OS"; exit 1 ;;
esac
exec "$BIN" "$@"
