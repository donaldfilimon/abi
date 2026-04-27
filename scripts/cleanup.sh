#!/usr/bin/env bash
set -euo pipefail

# Mass cleanup script for local repository noise.
# Safe, idempotent cleanup of common OS/build artifacts.

ROOT_DIR=$(cd "$(dirname "$0")/.." && pwd)

echo "Cleaning noise artifacts in $ROOT_DIR ..."

# List of known noise paths to remove if present
NOISE_PATHS=(
  "zig-out"
  ".cursor"
  ".DS_Store"
  "install_debug.log"
  "manual_build_err.log"
  "test_results.txt"
  "parity_results.txt"
  ".darwin-sdk-overlay"
  "zig-cache"
  ".zig-cache"
  "zig-out"
)

for p in "${NOISE_PATHS[@]}"; do
  if [ -e "$ROOT_DIR/$p" ]; then
    rm -rf "$ROOT_DIR/$p" && echo "Removed $p"
  fi
done

# Remove empty dirs that might be left behind
for d in $(find "$ROOT_DIR" -type d -empty 2>/dev/null | sort -r); do
  rmdir "$d" 2>/dev/null && echo "Removed empty dir $d"
done

echo "Cleanup complete."
exit 0
