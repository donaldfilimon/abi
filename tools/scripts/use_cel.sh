#!/usr/bin/env bash
# Use the .cel toolchain's Zig for this repo.
# Usage:
#   eval "$(./tools/scripts/use_cel.sh)"
#   — or —
#   source tools/scripts/use_cel.sh
#
# This script sets the PATH so that .cel/bin/{zig,zls} take precedence.
# It also validates the version against .zigversion.

# Only enable strict mode when executed (not sourced), to avoid breaking
# the caller's shell environment.
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  set -euo pipefail
fi

# Determine repo root relative to this script (tools/scripts/use_cel.sh → ../../)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Source shared helpers
source "$REPO_ROOT/.cel/config.sh"
source "$REPO_ROOT/.cel/lib.sh"

CEL_ZIG="$REPO_ROOT/.cel/bin/$(cel_binary_name zig)"
EXPECTED="$(cel_expected_zig_version || true)"
BOOTSTRAP_HOST_ZIG="$REPO_ROOT/zig-bootstrap-emergency/out/host/bin/$(cel_binary_name zig)"

if [[ ! -x "$CEL_ZIG" ]]; then
  echo "Error: .cel toolchain not found at $CEL_ZIG" >&2
  if [[ -n "$EXPECTED" ]]; then
    echo "Repo pin: $EXPECTED" >&2
  fi
  if command -v zig >/dev/null 2>&1; then
    echo "Stock zig on PATH: $(cel_stock_zig_path) ($(cel_stock_zig_version))" >&2
    if [[ -n "$EXPECTED" && "$(cel_stock_zig_version)" != "$EXPECTED" ]]; then
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
  # Return 1 when sourced, exit 1 when executed
  if [[ "${BASH_SOURCE[0]}" != "${0}" ]]; then
    return 1 2>/dev/null || true
  fi
  exit 1
fi

export PATH="$REPO_ROOT/.cel/bin:$PATH"

# Verify version matches .zigversion if the file exists
if [[ -f "$REPO_ROOT/.zigversion" ]]; then
  ACTUAL="$(zig version 2>/dev/null | tr -d '[:space:]')"
  if [[ -n "$EXPECTED" && -n "$ACTUAL" && "$ACTUAL" != "$EXPECTED" ]]; then
    echo "Error: .cel zig version ($ACTUAL) does not match .zigversion ($EXPECTED)" >&2
    echo "Rebuild with: ./.cel/build.sh --clean" >&2
    if [[ "${BASH_SOURCE[0]}" != "${0}" ]]; then
      return 1 2>/dev/null || true
    fi
    exit 1
  fi
fi

if [[ "${USE_CEL_QUIET:-0}" != "1" ]]; then
  echo "Using Zig (.cel): $(zig version)"
  echo "  Binary: $CEL_ZIG"
  if [[ -x "$REPO_ROOT/.cel/bin/$(cel_binary_name zls)" ]]; then
    echo "  ZLS: $REPO_ROOT/.cel/bin/$(cel_binary_name zls)"
  fi
  echo "  PATH updated: $REPO_ROOT/.cel/bin is first"
fi
