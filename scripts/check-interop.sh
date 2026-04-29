#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

echo "=== MCP-ACP Interop Check ==="

# 1) MCP health (multi-HA)
if command -v bash >/dev/null 2>&1; then
  if [ -f scripts/check-mcp-health.sh ]; then
    bash scripts/check-mcp-health.sh || true
  fi
fi

echo "Checking ACP endpoints connectivity (if ACP_ENDPOINTS provided)"
if [[ -n "${ACP_ENDPOINTS:-}" ]]; then
  bash scripts/list-acp-endpoints.sh
else
  echo "ACP_ENDPOINTS not set; skipping ACP reachability checks."
fi

echo "Interop check completed."
