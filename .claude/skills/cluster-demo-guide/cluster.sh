#!/usr/bin/env bash
# cluster-demo-guide driver: build the abi CLI and exercise the in-process
# Raft-style consensus demo (elect -> replicate -> fail over -> re-elect) plus
# the single-node status report. Asserts the consensus markers.
#
# Usage: .claude/skills/cluster-demo-guide/cluster.sh [nodes]   # default 3
set -uo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../../.." && pwd)
cd "$REPO_ROOT"
ABI="$REPO_ROOT/zig-out/bin/abi"
NODES="${1:-3}"
case "$NODES" in ''|*[!0-9]*) echo "usage: cluster.sh [nodes]" >&2; exit 2;; esac
fail=0
say() { printf '\n=== %s ===\n' "$*"; }
check() { local label="$1" expect="$2"; shift 2; local out; out=$("$@" 2>&1)
    printf '%s\n' "$out"
    grep -qF -- "$expect" <<<"$out" && echo "[ok] $label" || { echo "[FAIL] $label (missing: $expect)"; fail=$((fail+1)); }; }

say "build cli"
./build.sh cli >/dev/null 2>&1 && echo "[ok] build" || { echo "[FAIL] build"; exit 1; }
[ -x "$ABI" ] || { echo "[FAIL] no binary"; exit 1; }

check "cluster status"      "cluster: nodes="     "$ABI" wdbx cluster status
check "demo: election"      "leader_elected=true" "$ABI" wdbx cluster demo "$NODES"
check "demo: replication"   "replicate("          "$ABI" wdbx cluster demo "$NODES"
check "demo: failover+reelect" "re-election"       "$ABI" wdbx cluster demo "$NODES"

say "summary"; echo "failed: $fail"
[ "$fail" -eq 0 ] && echo "RESULT: PASS — cluster consensus demo ran." || echo "RESULT: FAIL — $fail check(s)."
exit "$fail"
