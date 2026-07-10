#!/usr/bin/env bash
# wdbx-api-serve driver: build the abi CLI and drive the WDBX REST server
# end-to-end — launch it on a loopback port, hit GET /health and GET /stats,
# then verify ABI_WDBX_REST_TOKEN bearer auth (401 without / with a wrong token,
# 200 with the right one). Servers are always killed on exit. Loopback only.
#
# Usage: .agents/skills/wdbx-api-serve/serve.sh [port]   # default port 8091
set -uo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../../.." && pwd)
cd "$REPO_ROOT"
ABI="$REPO_ROOT/zig-out/bin/abi"
PORT="${1:-8091}"
PORT2=$((PORT + 1))
fail=0
PIDS=()
cleanup() { for p in "${PIDS[@]:-}"; do kill "$p" 2>/dev/null || true; done; }
trap cleanup EXIT
say() { printf '\n=== %s ===\n' "$*"; }
mark() { grep -qF -- "$2" <<<"$1" && echo "[ok] $3" || { echo "[FAIL] $3 (missing: $2)"; fail=$((fail+1)); }; }
wait_up() { local port="$1"; for _ in $(seq 1 25); do curl -s -o /dev/null "http://127.0.0.1:$port/health" && return 0; sleep 0.3; done; return 1; }
code() { curl -s -o /dev/null -w '%{http_code}' "$@"; }

say "build cli"
./build.sh cli >/dev/null 2>&1 && echo "[ok] build" || { echo "[FAIL] build"; exit 1; }
[ -x "$ABI" ] || { echo "[FAIL] $ABI not produced"; exit 1; }

say "launch WDBX REST (no auth) on :$PORT"
"$ABI" wdbx api serve "$PORT" >/dev/null 2>&1 & PIDS+=($!)
wait_up "$PORT" || { echo "[FAIL] server did not come up on :$PORT"; fail=$((fail+1)); }

say "GET /health + GET /stats"
health=$(curl -s "http://127.0.0.1:$PORT/health"); printf 'health: %s\n' "$health"
mark "$health" '"status":"ok"' "/health ok"
stats=$(curl -s "http://127.0.0.1:$PORT/stats"); printf 'stats: %s\n' "$stats"
mark "$stats" '"backend"' "/stats returns store stats"

say "bearer auth (ABI_WDBX_REST_TOKEN) on :$PORT2"
ABI_WDBX_REST_TOKEN=probe-tok "$ABI" wdbx api serve "$PORT2" >/dev/null 2>&1 & PIDS+=($!)
wait_up "$PORT2" >/dev/null 2>&1 || true
c_none=$(code "http://127.0.0.1:$PORT2/health")
c_wrong=$(code -H "Authorization: Bearer nope" "http://127.0.0.1:$PORT2/health")
c_right=$(code -H "Authorization: Bearer probe-tok" "http://127.0.0.1:$PORT2/health")
echo "no-token=$c_none wrong-token=$c_wrong right-token=$c_right"
[ "$c_none" = "401" ]  && echo "[ok] no token rejected (401)"    || { echo "[FAIL] no-token expected 401, got $c_none"; fail=$((fail+1)); }
[ "$c_wrong" = "401" ] && echo "[ok] wrong token rejected (401)" || { echo "[FAIL] wrong-token expected 401, got $c_wrong"; fail=$((fail+1)); }
[ "$c_right" = "200" ] && echo "[ok] right token accepted (200)" || { echo "[FAIL] right-token expected 200, got $c_right"; fail=$((fail+1)); }

say "summary"; echo "failed checks: $fail"
[ "$fail" -eq 0 ] && echo "RESULT: PASS — WDBX REST serves + enforces bearer auth." || echo "RESULT: FAIL — $fail check(s)."
exit "$fail"
