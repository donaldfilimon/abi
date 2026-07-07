#!/usr/bin/env bash
# wdbx-cluster-serve driver: build the abi CLI and run a networked WDBX consensus
# node (`abi wdbx cluster serve <port>`) — background-launch the loopback
# RequestVote/AppendEntries RPC listener, poll its stderr for the readiness
# marker, probe the port (if `nc` is present), assert no bind/panic error, then
# tear the node down. The node blocks in the accept loop, so readiness is the
# marker in its log, not process exit. Loopback only; always killed on exit.
#
# Usage: .agents/skills/wdbx-cluster-serve/cluster-serve.sh [port]   # default 8092
set -uo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../../.." && pwd)
cd "$REPO_ROOT"
ABI="$REPO_ROOT/zig-out/bin/abi"
PORT="${1:-8092}"
case "$PORT" in
  ''|*[!0-9]*) echo "usage: cluster-serve.sh [port]   (port must be numeric; default 8092)" >&2; exit 2 ;;
esac
fail=0
PIDS=()
LOG=$(mktemp "${TMPDIR:-/tmp}/wdbx-cluster-serve.XXXXXX")
cleanup() {
  for p in "${PIDS[@]:-}"; do kill "$p" 2>/dev/null || true; done
  rm -f "$LOG" 2>/dev/null || true
}
trap cleanup EXIT
say() { printf '\n=== %s ===\n' "$*"; }

say "build cli"
./build.sh cli >/dev/null 2>&1 && echo "[ok] build" || { echo "[FAIL] build"; exit 1; }
[ -x "$ABI" ] || { echo "[FAIL] $ABI not produced"; exit 1; }

MARKER="serving consensus RPC on 127.0.0.1:$PORT"

say "launch WDBX consensus node on 127.0.0.1:$PORT"
"$ABI" wdbx cluster serve "$PORT" >"$LOG" 2>&1 &
NODE_PID=$!
PIDS+=("$NODE_PID")
disown "$NODE_PID" 2>/dev/null || true  # silence the shell's "Terminated" notice on trap-kill

# The node blocks in the accept loop, so watch its stderr for the readiness
# marker rather than waiting for exit. Bail early if the process dies.
ready=0
for _ in $(seq 1 25); do
  if grep -qF -- "$MARKER" "$LOG"; then ready=1; break; fi
  kill -0 "$NODE_PID" 2>/dev/null || break
  sleep 0.3
done

say "assert readiness marker"
if [ "$ready" = 1 ]; then
  echo "[ok] marker: $MARKER"
else
  echo "[FAIL] readiness marker not seen"; sed 's/^/  log| /' "$LOG"; fail=$((fail+1))
fi

say "probe port 127.0.0.1:$PORT"
if command -v nc >/dev/null 2>&1; then
  if nc -z 127.0.0.1 "$PORT" 2>/dev/null; then
    echo "[ok] port $PORT accepting"
  else
    echo "[FAIL] port $PORT not accepting"; fail=$((fail+1))
  fi
else
  echo "[skip] nc absent — the readiness marker is the primary gate"
fi

say "assert no bind/panic error"
if grep -qiE 'bind .* failed|panic' "$LOG"; then
  echo "[FAIL] error in node log"; grep -iE 'bind .* failed|panic' "$LOG" | sed 's/^/  err| /'; fail=$((fail+1))
else
  echo "[ok] no bind/panic error"
fi

say "summary"; echo "failed checks: $fail"
[ "$fail" -eq 0 ] \
  && echo "RESULT: PASS — WDBX cluster node served consensus RPC on loopback." \
  || echo "RESULT: FAIL — $fail check(s)."
exit "$fail"
