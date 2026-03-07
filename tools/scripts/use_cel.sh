#!/usr/bin/env bash
# Use the .cel toolchain's Zig for this repo.
# Usage:
#   eval "$(./tools/scripts/use_cel.sh)"
#   — or —
#   source tools/scripts/use_cel.sh
set -e

# Determine repo root relative to this script (tools/scripts/use_cel.sh → ../../)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

CEL_ZIG="$REPO_ROOT/.cel/bin/zig"

if [[ ! -x "$CEL_ZIG" ]]; then
  echo "Error: .cel toolchain not found at $CEL_ZIG" >&2
  echo "Build it first:  cd $REPO_ROOT && ./.cel/build.sh" >&2
  exit 1
fi

export PATH="$REPO_ROOT/.cel/bin:$PATH"

# Verify version matches .zigversion if the file exists
ZIGVERSION_FILE="$REPO_ROOT/.zigversion"
if [[ -f "$ZIGVERSION_FILE" ]]; then
  EXPECTED="$(cat "$ZIGVERSION_FILE" | tr -d '[:space:]')"
  ACTUAL="$(zig version 2>/dev/null | tr -d '[:space:]')"
  if [[ -n "$EXPECTED" && -n "$ACTUAL" && "$ACTUAL" != "$EXPECTED" ]]; then
    echo "Warning: .cel zig version ($ACTUAL) does not match .zigversion ($EXPECTED)" >&2
  fi
fi

if [[ "${USE_CEL_QUIET:-0}" != "1" ]]; then
  echo "Using Zig (.cel): $(zig version)"
fi
