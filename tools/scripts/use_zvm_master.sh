#!/usr/bin/env bash
# Use ZVM-managed Zig 0.16 master for this architecture (darwin/arm64, etc.).
# Source this file: source tools/scripts/use_zvm_master.sh
# Or run directly: ./tools/scripts/use_zvm_master.sh  (then zig is on PATH in this shell)
set -e
if command -v zvm &>/dev/null; then
  zvm use master &>/dev/null
  export PATH="$HOME/.zvm/bin:$PATH"
  if [[ "${USE_ZVM_QUIET:-0}" != "1" ]]; then
    echo "Using Zig: $(zig version)"
  fi
else
  echo "zvm not found. Install from https://github.com/tristanisham/zvm" >&2
  exit 1
fi
