#!/usr/bin/env bash
# run-abi smoke harness: build + exercise the real abi CLI and abi-mcp server.
#
# One command to build both binaries, run representative CLI subcommands, and
# drive the MCP server over stdio with real JSON-RPC. Checks exit codes and
# greps output for expected markers. Writes a full transcript to disk.
#
# Usage:   .claude/skills/run-abi/smoke.sh
# From any cwd; it resolves the repo root from its own location.
set -uo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../../.." && pwd)
cd "$REPO_ROOT"

ABI="$REPO_ROOT/zig-out/bin/abi"
ABI_MCP="$REPO_ROOT/zig-out/bin/abi-mcp"
STORE="$REPO_ROOT/zig-out/smoke-memory.jsonl"
TRANSCRIPT="$REPO_ROOT/zig-out/run-abi-smoke.txt"

fail=0
pass=0

# zig-out/ may not exist before the first build; ensure the transcript dir is
# writable up front so every log line lands instead of erroring out per-line.
mkdir -p "$(dirname -- "$TRANSCRIPT")"

log() { printf '%s\n' "$*" | tee -a "$TRANSCRIPT"; }

# run <label> <expected-substring|-> -- <command...>
run() {
    local label="$1" expect="$2"; shift 2
    [ "$1" = "--" ] && shift
    log ""
    log "### $label"
    log "\$ $*"
    local out rc
    out=$("$@" 2>&1); rc=$?
    printf '%s\n' "$out" | tee -a "$TRANSCRIPT"
    log "[exit=$rc]"
    if [ "$rc" -ne 0 ]; then
        log "FAIL: $label exited $rc"; fail=$((fail+1)); return
    fi
    if [ "$expect" != "-" ] && ! grep -Fq -- "$expect" <<<"$out"; then
        log "FAIL: $label missing expected output: $expect"; fail=$((fail+1)); return
    fi
    pass=$((pass+1))
}

: > "$TRANSCRIPT"
log "=== run-abi smoke @ $(date) ==="
log "repo: $REPO_ROOT"
log "zig:  $(zig version 2>/dev/null || echo MISSING)"

# --- Build ---------------------------------------------------------------
run "build cli"  "-" -- ./build.sh cli
run "build mcp"  "-" -- ./build.sh mcp

if [ ! -x "$ABI" ] || [ ! -x "$ABI_MCP" ]; then
    log "FATAL: binaries not produced (abi=$ABI abi-mcp=$ABI_MCP)"
    exit 1
fi

# --- CLI surface ---------------------------------------------------------
run "cli help"          "Usage: abi"        -- "$ABI" help
run "cli backends"      "GPU backend report" -- "$ABI" backends
run "cli scheduler"     "source=cli-scheduler-status" -- "$ABI" scheduler status
run "cli complete"      "model=claude-fable-5"   -- "$ABI" complete "smoke: summarize scheduler status"
run "cli plugin list"   "Installed Plugins (" -- "$ABI" plugin list

# --- WDBX store round-trip ----------------------------------------------
rm -f "$STORE"
run "wdbx db init"      "initialized empty WDBX" -- "$ABI" wdbx db init "$STORE"
run "wdbx block insert" "appended block"          -- "$ABI" wdbx block insert "$STORE" abi '{"note":"smoke checkpoint"}'
run "wdbx query"        "blocks"                  -- "$ABI" wdbx query "$STORE"

# --- MCP server over stdio (real JSON-RPC) -------------------------------
log ""
log "### mcp stdio (initialize + tools/list + tools/call)"
MCP_OUT=$(printf '%s\n' \
  '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"smoke","version":"0"}}}' \
  '{"jsonrpc":"2.0","id":2,"method":"tools/list"}' \
  '{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"scheduler_info","arguments":{}}}' \
  '{"jsonrpc":"2.0","id":4,"method":"tools/call","params":{"name":"gpu_status","arguments":{}}}' \
  '{"jsonrpc":"2.0","id":5,"method":"tools/call","params":{"name":"plugin_list","arguments":{}}}' \
  | "$ABI_MCP" stdio 2>>"$TRANSCRIPT")
mcp_rc=$?
printf '%s\n' "$MCP_OUT" | tee -a "$TRANSCRIPT"
log "[exit=$mcp_rc]"
if [ "$mcp_rc" -eq 0 ]; then
    pass=$((pass+1)); log "ok: mcp exited 0"
else
    fail=$((fail+1)); log "FAIL: mcp exited $mcp_rc"
fi
for marker in '"serverInfo"' '"name":"gpu_status"' 'scheduler running=' 'backend=metal' 'plugins count=16'; do
    if grep -Fq -- "$marker" <<<"$MCP_OUT"; then
        pass=$((pass+1)); log "ok: mcp has $marker"
    else
        fail=$((fail+1)); log "FAIL: mcp missing $marker"
    fi
done

# --- Summary -------------------------------------------------------------
log ""
log "=== summary: pass=$pass fail=$fail ==="
log "transcript: $TRANSCRIPT"
[ "$fail" -eq 0 ] && log "SMOKE OK" || log "SMOKE FAILED"
exit "$fail"
