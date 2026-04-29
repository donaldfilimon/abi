#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

echo "=== ACP Endpoints Discovery ==="
if [[ -z "${ACP_ENDPOINTS:-}" ]]; then
  echo "No ACP_ENDPOINTS environment variable set. Provide a comma-separated list of ACP endpoints (e.g. https://acp1.contoso.com,https://acp2.contoso.com)."
  exit 0
fi

IFS=',' read -ra ENDS <<< "$ACP_ENDPOINTS"
for ep in "${ENDS[@]}"; do
  ep_trim=$(echo "$ep" | xargs) # trim spaces
  if [[ -z "$ep_trim" ]]; then continue; fi
  url="$ep_trim/health"; # health check if available
  if curl -fsS --max-time 3 "$url" >/dev/null 2>&1; then
    echo "OK: $ep_trim is reachable (health OK at $url)"
  else
    echo "WARN: Could not health-check $ep_trim (endpoint: $url)"
  fi
done
