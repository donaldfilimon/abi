#!/usr/bin/env bash
# Cross-compile smoke check: verifies the ABI CLI compiles+links for the
# supported non-native targets. Opt-in (NOT part of `./build.sh check`) because
# the cold cross builds are slow. Run from the repo root:
#
#   ./tools/cross_smoke.sh            # default target set
#   ./tools/cross_smoke.sh aarch64-linux-gnu x86_64-windows-gnu
#
# Build-time tools (gen_plugin_registry, check_parity) are host-targeted in
# build.zig so they remain executable under a cross build.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR/.."

ZIG_BIN=$(command -v zig)

TARGETS=("$@")
if [ "${#TARGETS[@]}" -eq 0 ]; then
    TARGETS=(x86_64-linux-gnu x86_64-windows-gnu aarch64-macos)
fi

fail=0
for t in "${TARGETS[@]}"; do
    printf '== cross cli: %s ==\n' "$t"
    if "$ZIG_BIN" build cli -Dtarget="$t"; then
        printf '   OK: %s\n' "$t"
    else
        printf '   FAIL: %s\n' "$t"
        fail=1
    fi
    printf '== cross example-3d-hybrid: %s ==\n' "$t"
    if "$ZIG_BIN" build example-3d-hybrid -Dtarget="$t"; then
        printf '   OK: %s\n' "$t"
    else
        printf '   FAIL: %s\n' "$t"
        fail=1
    fi
done

if [ "$fail" -ne 0 ]; then
    echo "cross_smoke: one or more targets failed" >&2
    exit 1
fi
echo "cross_smoke: all targets compiled + linked"
