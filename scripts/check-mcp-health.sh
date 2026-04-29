#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

echo "=== MCP Health Check (multi-HA) (enhanced) ==="

if ! command -v jq >/dev/null 2>&1; then
  echo "jq not found. Cannot parse mcp/servers.json automatically."
  echo "Please install jq to enable automated health checks."
  echo "Manual check: curl http://127.0.0.1:50051/health; curl http://127.0.0.1:50052/health"
  exit 0
fi

endpoints=()
while IFS= read -r url; do
  endpoints+=("$url")
done < <(jq -r '.mcpServers | to_entries[] | "http://" + (.value.env.ABI_MCP_HOST) + ":" + (.value.env.ABI_MCP_PORT) + "/health"' mcp/servers.json)

if [ ${#endpoints[@]} -eq 0 ]; then
  echo "No MCP servers defined in mcp/servers.json; nothing to health-check."
  exit 0
fi

fail=0
for url in "${endpoints[@]}"; do
  if curl -fsS --max-time 2 "$url" >/dev/null; then
    echo "OK: MCP health at $url"
  else
    echo "ERROR: MCP health check failed or unreachable ($url)"
    fail=1
  fi
done

exit $fail
