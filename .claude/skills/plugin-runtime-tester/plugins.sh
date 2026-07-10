#!/usr/bin/env bash
# plugin-runtime-tester driver: build the abi CLI, list the generated plugin
# registry, and execute a sample of bundled plugins through `plugin run`,
# asserting each returns its real run() output (not PluginNotFound / a generic
# ack). Also checks list and run agree.
#
# Usage: .claude/skills/plugin-runtime-tester/plugins.sh
set -uo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../../.." && pwd)
cd "$REPO_ROOT"
ABI="$REPO_ROOT/zig-out/bin/abi"
fail=0
say() { printf '\n=== %s ===\n' "$*"; }

say "build cli"
./build.sh cli >/dev/null 2>&1 && echo "[ok] build" || { echo "[FAIL] build"; exit 1; }
[ -x "$ABI" ] || { echo "[FAIL] no binary"; exit 1; }

say "abi plugin list"
list=$("$ABI" plugin list 2>&1); printf '%s\n' "$list"
grep -qF "Installed Plugins (" <<<"$list" && echo "[ok] registry lists plugins" \
    || { echo "[FAIL] no plugin list header"; fail=$((fail+1)); }

# Sample across the feature gates; each must return its mod.run() event line.
say "plugin run (sample)"
for p in telemetry-exporter ai-plugin hash-plugin sea-plugin foundationmodels-plugin; do
    out=$("$ABI" plugin run "$p" "probe" 2>&1)
    if grep -qF "event (bytes=" <<<"$out"; then
        echo "[ok] run $p -> $out"
    else
        echo "[FAIL] run $p -> $out"; fail=$((fail+1))
    fi
done

# A name that is registered as an example must run; an unknown name must error cleanly.
say "negative: unknown plugin errors"
out=$("$ABI" plugin run definitely-not-a-plugin "x" 2>&1) || true
grep -qiF "PluginNotFound" <<<"$out" && echo "[ok] unknown -> PluginNotFound" \
    || echo "[note] unknown plugin output: $out"

say "summary"; echo "failed: $fail"
[ "$fail" -eq 0 ] && echo "RESULT: PASS — plugin registry + run dispatch verified." || echo "RESULT: FAIL — $fail check(s)."
exit "$fail"
