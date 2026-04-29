#!/usr/bin/env bash
set -euo pipefail

echo "[CI] Running Zig check-parity baseline..."
echo "Using Zig from pinned environment: $(zig --version 2>/dev/null || echo unknown)"

if [ -n "${ABI_JWT_SECRET:-}" ]; then
  echo "[CI] ABI_JWT_SECRET detected; parity tests can run auth paths."
else
  echo "[CI] ABI_JWT_SECRET not set; parity will skip auth-heavy tests or gate accordingly." 
fi

zig build check-parity

# Optional: run a focused parity/test subset if parity passes
echo "[CI] Parity check completed. Consider running targeted tests with -test-filter if needed."
