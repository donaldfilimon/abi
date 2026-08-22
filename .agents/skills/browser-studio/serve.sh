#!/usr/bin/env bash
# browser-studio driver: build the abi CLI and drive the loopback multimodal
# studio — health, capabilities, POST /analyze, then bearer auth on POST.
# Servers are always killed on exit. Loopback only.
#
# Usage: .agents/skills/browser-studio/serve.sh [port]   # default port 8095
set -uo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../../.." && pwd)
cd "$REPO_ROOT"
ABI="$REPO_ROOT/target/debug/abi"
PORT="${1:-8095}"
PORT2=$((PORT + 1))
fail=0
PIDS=()
cleanup() {
  for p in "${PIDS[@]:-}"; do
    kill "$p" 2>/dev/null || true
    wait "$p" 2>/dev/null || true
  done
}
trap cleanup EXIT
say() { printf '\n=== %s ===\n' "$*"; }
mark() { grep -qF -- "$2" <<<"$1" && echo "[ok] $3" || { echo "[FAIL] $3 (missing: $2)"; fail=$((fail+1)); }; }
wait_up() { local port="$1"; for _ in $(seq 1 25); do curl -s -o /dev/null "http://127.0.0.1:$port/health" && return 0; sleep 0.3; done; return 1; }
code() { curl -s -o /dev/null -w '%{http_code}' "$@"; }

say "build cli"
./tools/cargo.sh build -p abi-cli >/dev/null 2>&1 && echo "[ok] build" || { echo "[FAIL] build"; exit 1; }
[ -x "$ABI" ] || { echo "[FAIL] $ABI not produced"; exit 1; }

say "launch browser studio (no auth) on :$PORT"
"$ABI" agent browser --studio --port "$PORT" >/dev/null 2>&1 & PIDS+=($!)
wait_up "$PORT" || { echo "[FAIL] server did not come up on :$PORT"; fail=$((fail+1)); }

say "GET /health + GET /capabilities + POST /analyze"
health=$(curl -s "http://127.0.0.1:$PORT/health"); printf 'health: %s\n' "$health"
mark "$health" '"surface":"browser-studio"' "/health ok"
caps=$(curl -s "http://127.0.0.1:$PORT/capabilities"); printf 'capabilities: %s\n' "$caps"
mark "$caps" '"neural_vision":false' "capabilities disclose no neural vision"
mark "$caps" '"neural_stt":false' "capabilities disclose no neural STT"
page=$(curl -s "http://127.0.0.1:$PORT/")
mark "$page" "ABI browser studio" "HTML studio served"
analyze=$(curl -s -H 'Content-Type: application/json' \
  -d '{"kind":"image","width":2,"height":2,"pixels":[0,64,128,255]}' \
  "http://127.0.0.1:$PORT/analyze")
mark "$analyze" '"kind":"image"' "POST /analyze image"
mark "$analyze" 'deterministic-local' "analyze names the local engine"

say "bearer auth (ABI_BROWSER_STUDIO_TOKEN) on :$PORT2"
ABI_BROWSER_STUDIO_TOKEN=probe-tok "$ABI" agent browser --studio --port "$PORT2" >/dev/null 2>&1 & PIDS+=($!)
# GET / is open so the page can load; wait via that, then probe /health.
for _ in $(seq 1 25); do curl -s -o /dev/null "http://127.0.0.1:$PORT2/" && break; sleep 0.3; done
c_none=$(code "http://127.0.0.1:$PORT2/health")
c_wrong=$(code -H "Authorization: Bearer nope" "http://127.0.0.1:$PORT2/health")
c_right=$(code -H "Authorization: Bearer probe-tok" "http://127.0.0.1:$PORT2/health")
c_page=$(code "http://127.0.0.1:$PORT2/")
echo "no-token=$c_none wrong-token=$c_wrong right-token=$c_right page=$c_page"
[ "$c_none" = "401" ]  && echo "[ok] /health no token rejected (401)"    || { echo "[FAIL] no-token expected 401, got $c_none"; fail=$((fail+1)); }
[ "$c_wrong" = "401" ] && echo "[ok] /health wrong token rejected (401)" || { echo "[FAIL] wrong-token expected 401, got $c_wrong"; fail=$((fail+1)); }
[ "$c_right" = "200" ] && echo "[ok] /health right token accepted (200)" || { echo "[FAIL] right-token expected 200, got $c_right"; fail=$((fail+1)); }
[ "$c_page" = "200" ]  && echo "[ok] GET / stays open so the page can load" || { echo "[FAIL] GET / expected 200, got $c_page"; fail=$((fail+1)); }

say "summary"; echo "failed checks: $fail"
[ "$fail" -eq 0 ] && echo "RESULT: PASS — browser studio serves + enforces bearer on API routes." || echo "RESULT: FAIL — $fail check(s)."
exit "$fail"
