#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

resolve_zig() {
    local candidate
    for candidate in \
        "$HOME/.zvm/bin/zig" \
        "$HOME/.local/bin/zig" \
        "$HOME/.zigly/versions/"*/bin/zig \
        "$(command -v zig 2>/dev/null || true)"; do
        if [ -n "$candidate" ] && [ -x "$candidate" ]; then
            readlink -f "$candidate" 2>/dev/null || echo "$candidate"
            return 0
        fi
    done
    return 1
}

ZIG2="$(resolve_zig || true)"
if [ -z "$ZIG2" ]; then
    echo "Error: zig not found. Run 'tools/zigly --install' first." >&2
    exit 1
fi

echo "ZIG2 is: $ZIG2"

rm -rf zig-out/bin/zigly zigly manual_build_err.log
mkdir -p zig-out/bin

"$ZIG2" build-exe src/main.zig \
    -lc \
    -O ReleaseSafe \
    --name zigly \
    --cache-dir .zig-cache \
    --global-cache-dir "$HOME/.cache/zig" \
    2>manual_build_err.log

if [ ! -x zigly ]; then
    echo "Failed to build zigly native helper" >&2
    cat manual_build_err.log >&2
    exit 1
fi

mv zigly zig-out/bin/zigly
echo "Success! Binary is at zig-out/bin/zigly"
