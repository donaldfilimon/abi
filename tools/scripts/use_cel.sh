#!/usr/bin/env bash
# Use the .cel toolchain's Zig for this repo.
# Usage:
#   eval "$(./tools/scripts/use_cel.sh)"
#   — or —
#   source tools/scripts/use_cel.sh
#
# This script sets the PATH so that .cel/bin/{zig,zls} take precedence.
# It also validates the version against .zigversion.
set -euo pipefail

# Determine repo root relative to this script (tools/scripts/use_cel.sh → ../../)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

CEL_ZIG="$REPO_ROOT/.cel/bin/zig"
EXPECTED="$(tr -d '[:space:]' < "$REPO_ROOT/.zigversion" 2>/dev/null || true)"
BOOTSTRAP_HOST_ZIG="$REPO_ROOT/zig-bootstrap-emergency/out/host/bin/zig"

stock_zig_path() {
  if command -v zig >/dev/null 2>&1; then
    command -v zig
  else
    return 1
  fi
}

stock_zig_version() {
  local path
  path="$(stock_zig_path)" || return 1
  "$path" version 2>/dev/null | tr -d '\r' | head -n 1
}

if [[ ! -x "$CEL_ZIG" ]]; then
  echo "Error: .cel toolchain not found at $CEL_ZIG" >&2
  if [[ -n "$EXPECTED" ]]; then
    echo "Repo pin: $EXPECTED" >&2
  fi
  if command -v zig >/dev/null 2>&1; then
    echo "Stock zig on PATH: $(stock_zig_path) ($(stock_zig_version))" >&2
    if [[ -n "$EXPECTED" && "$(stock_zig_version)" != "$EXPECTED" ]]; then
      echo "Stock zig does not match the repo pin." >&2
    fi
  else
    echo "Stock zig on PATH: not found" >&2
  fi
  echo "" >&2
  if [[ -x "$BOOTSTRAP_HOST_ZIG" ]]; then
    echo "Next action:" >&2
    echo "  cd $REPO_ROOT && ./.cel/build.sh" >&2
  elif [[ -d "$REPO_ROOT/zig-bootstrap-emergency/zig" ]]; then
    echo "Next action:" >&2
    echo "  cd $REPO_ROOT && abi toolchain bootstrap" >&2
  else
    echo "Next action:" >&2
    echo "  cd $REPO_ROOT && ./tools/scripts/cel_migrate.sh --check" >&2
  fi
  exit 1
fi

export PATH="$REPO_ROOT/.cel/bin:$PATH"

# Verify version matches .zigversion if the file exists
ZIGVERSION_FILE="$REPO_ROOT/.zigversion"
if [[ -f "$ZIGVERSION_FILE" ]]; then
  ACTUAL="$(zig version 2>/dev/null | tr -d '[:space:]')"
  if [[ -n "$EXPECTED" && -n "$ACTUAL" && "$ACTUAL" != "$EXPECTED" ]]; then
    echo "Error: .cel zig version ($ACTUAL) does not match .zigversion ($EXPECTED)" >&2
    echo "Rebuild with: ./.cel/build.sh --clean" >&2
    exit 1
  fi
fi

if [[ "${USE_CEL_QUIET:-0}" != "1" ]]; then
  echo "Using Zig (.cel): $(zig version)"
  echo "  Binary: $CEL_ZIG"
  if [[ -x "$REPO_ROOT/.cel/bin/zls" ]]; then
    echo "  ZLS: $REPO_ROOT/.cel/bin/zls"
  fi
  echo "  PATH updated: $REPO_ROOT/.cel/bin is first"
fi
