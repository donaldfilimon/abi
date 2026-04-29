#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd \"$(dirname \"$0\")/..\" && pwd)"
OS="$(uname -s)"
case "$OS" in
  Linux|Darwin) BIN="$ROOT_DIR/zig-out/bin/abi-mcp" ;;
  MINGW*|MSYS*|CYGWIN*) BIN="$ROOT_DIR/zig-out/bin/abi-mcp.exe" ;;
  *) echo "Unsupported OS: $OS"; exit 1 ;;
esac
exec "$BIN" "$@"
