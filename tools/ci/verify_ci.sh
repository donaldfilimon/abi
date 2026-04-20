#!/bin/sh
set -euo

echo "CI verification: parity baseline and shell syntax checks up to date"
echo "Step 1: run verify_changes.sh"
tools/verify_changes.sh || exit 1

echo "Step 2: run parity checks if Zig is available"
if command -v zig >/dev/null 2>&1; then
  if zig --version >/dev/null 2>&1; then
    echo "Running: zig build check-parity"
    zig build check-parity || {
      echo "Parity check failed." >&2
      exit 1
    }
  else
    echo "Zig is not usable; skipping parity checks."
  fi
fi

echo "CI verification completed successfully."
exit 0
