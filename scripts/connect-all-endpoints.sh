#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

echo "=== Connect All MCP & ACP Endpoints (Summary) ==="

# Connect to MCP endpoints from mcp/servers.json
if ! command -v jq >/dev/null 2>&1; then
  echo "jq not found. Cannot parse mcp/servers.json for endpoints."
  echo "Manual checks: curl http://127.0.0.1:50051/health; curl http://127.0.0.1:50052/health (if configured)"
  exit 0
fi

pairs=$(jq -r '.mcpServers | to_entries[] | "\(.value.env.ABI_MCP_HOST)|\(.value.env.ABI_MCP_PORT)"' mcp/servers.json)
if [[ -z "$pairs" ]]; then
  echo "No MCP entries found in mcp/servers.json."
else
  while IFS='|' read -r host port; do
    url="http://${host}:${port}/health"
    printf "CONNECT MCP %s:%s ... " "$host" "$port"
    if curl -fsS --max-time 2 "$url" >/dev/null; then
      echo "OK"
    else
      echo "FAILED"
    fi
  done <<< "$pairs"
fi

echo "\n# ACP Endpoints (from ACP_ENDPOINTS env)"
if [[ -n "${ACP_ENDPOINTS:-}" ]]; then
  IFS=',' read -ra addrs <<< "$ACP_ENDPOINTS"
  for ep in "${addrs[@]}"; do
    ep_trim=$(echo "$ep" | xargs)
    if [[ -z "$ep_trim" ]]; then continue; fi
    url="$ep_trim/health"
    printf "CONNECT ACP %s ... " "$ep_trim"
    if curl -fsS --max-time 3 "$url" >/dev/null; then
      echo "OK"
    else
      echo "FAILED"
    fi
  done
else
  echo "No ACP endpoints configured (ACP_ENDPOINTS not set)."
fi

echo
echo "Interop summary complete."

## Optional: also run ACP connectivity check if endpoints are provided
if [[ -n "${ACP_ENDPOINTS:-}" ]]; then
  echo "\nACP endpoints configured; running quick checks."
  bash scripts/list-acp-endpoints.sh || true
fi
