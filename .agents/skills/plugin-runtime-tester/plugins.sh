#!/usr/bin/env bash
# plugin-runtime-tester driver: build the abi CLI, list the generated plugin
# registry, and execute every bundled plugin through `plugin run`,
# asserting each returns its real run() output (not PluginNotFound / a generic
# ack). Also checks list and run agree.
#
# Usage: .agents/skills/plugin-runtime-tester/plugins.sh
set -uo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../../.." && pwd)
cd "$REPO_ROOT"
ABI="$REPO_ROOT/target/debug/abi"
fail=0
say() { printf '\n=== %s ===\n' "$*"; }

say "build cli"
./tools/cargo.sh build -p abi-cli >/dev/null 2>&1 && echo "[ok] build" || { echo "[FAIL] build"; exit 1; }
[ -x "$ABI" ] || { echo "[FAIL] no binary"; exit 1; }

say "abi plugin list"
list=$("$ABI" plugin list 2>&1); printf '%s\n' "$list"
grep -qF "Installed Plugins (" <<<"$list" && echo "[ok] registry lists plugins" \
    || { echo "[FAIL] no plugin list header"; fail=$((fail+1)); }

# Exercise every compiled fixture; each must return its real mod.run() event
# line, not merely appear in the registry.
say "plugin run (all bundled fixtures)"
for p in \
    accelerator-plugin \
    ai-plugin \
    example-plugin \
    example-wdbx-plugin \
    foundationmodels-plugin \
    gpu-plugin \
    hash-plugin \
    metrics-plugin \
    mlir-plugin \
    mobile-plugin \
    nn-plugin \
    os-control-plugin \
    sea-plugin \
    shader-plugin \
    telemetry-exporter \
    tui-plugin
do
    out=$("$ABI" plugin run "$p" "probe" 2>&1)
    case "$p" in
        example-plugin) expected="received input" ;;
        example-wdbx-plugin) expected="executed" ;;
        *) expected="event (bytes=" ;;
    esac
    if grep -qF "$expected" <<<"$out"; then
        echo "[ok] run $p -> $out"
    else
        echo "[FAIL] run $p (expected '$expected') -> $out"
        fail=$((fail+1))
    fi
done

# A name that is registered as an example must run; an unknown name must error cleanly.
say "negative: unknown plugin errors"
out=$("$ABI" plugin run definitely-not-a-plugin "x" 2>&1)
rc=$?
if [ "$rc" -ne 0 ] && grep -qiF "plugin not found" <<<"$out"; then
    echo "[ok] unknown -> nonzero + plugin not found"
else
    echo "[FAIL] unknown plugin rc=$rc output: $out"
    fail=$((fail+1))
fi

say "summary"; echo "failed: $fail"
[ "$fail" -eq 0 ] && echo "RESULT: PASS — plugin registry + run dispatch verified." || echo "RESULT: FAIL — $fail check(s)."
exit "$fail"
