#!/bin/bash
set -euo pipefail

# ABI Build Wrapper for macOS 26.4+
# Works around LLD linker incompatibility by relinking with Apple ld + compiler_rt
#
# Usage: ./build.sh [--link] [zig build args...]
# Example: ./build.sh test --summary all
#          ./build.sh --link lib
#          ./build.sh -Dfeat-gpu=false

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
AUTO_LINK=true

# Check for --link flag
ARGS=""
for arg in "$@"; do
    if [ "$arg" = "--link" ]; then
        AUTO_LINK=true
    else
        ARGS="$ARGS $arg"
    fi
done
set -- $ARGS

# Resolve zig via zigup.sh (auto-downloads if missing)
ZIG2="$("$SCRIPT_DIR/tools/zigup.sh" --status)"
ZIG_LIB="$(dirname "$(dirname "$ZIG2")")/lib"

SYSROOT="$(xcrun --show-sdk-path 2>/dev/null || echo "/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk")"
MACOS_VER="26.4"

STDERR_FILE="$(mktemp)"
trap 'rm -f "$STDERR_FILE"' EXIT

run_build() {
    # Run zig build, capture stderr
    "$ZIG2" build \
        --zig-lib-dir "$ZIG_LIB" \
        --global-cache-dir "$HOME/.cache/zig" \
        --cache-dir .zig-cache \
        "$@" \
        2>"$STDERR_FILE" && return 0
    return 1
}

# Try direct build first
if run_build "$@"; then
    if [ "$AUTO_LINK" = true ]; then
        "$SCRIPT_DIR/tools/zigup.sh" --link 2>&1 || true
    fi
    exit 0
fi

# Find the build_zcu.o
BUILD_O="$(find .zig-cache/o -name 'build_zcu.o' 2>/dev/null | head -1 || true)"

# Find compiler_rt - use the most recent one from global cache
CRT="$(ls -t "$HOME/.cache/zig/o/"*/libcompiler_rt.a 2>/dev/null | head -1 || true)"

if [[ -n "$BUILD_O" && -f "$BUILD_O" ]]; then
    BUILD_DIR="$(dirname "$BUILD_O")"
    BUILD_BIN="$BUILD_DIR/build"

    echo "[darwin-wrapper] Relinking $BUILD_BIN with Apple ld + compiler_rt..." >&2

    LD_ARGS=(-dynamic -platform_version macos "$MACOS_VER" "$MACOS_VER" -syslibroot "$SYSROOT" -e _main -o "$BUILD_BIN" "$BUILD_O")
    if [[ -n "$CRT" && -f "$CRT" ]]; then
        LD_ARGS+=("$CRT")
    fi
    LD_ARGS+=(-lSystem)

    /usr/bin/ld "${LD_ARGS[@]}" 2>&1

    if [[ -x "$BUILD_BIN" ]]; then
        echo "[darwin-wrapper] Build runner relinked. Running build..." >&2
        "$BUILD_BIN" "$ZIG2" "$ZIG_LIB" "$(pwd)" ".zig-cache" "${HOME}/.cache/zig" "$@"
        if [ "$AUTO_LINK" = true ]; then
            "$SCRIPT_DIR/tools/zigup.sh" --link 2>&1 || true
        fi
        exit 0
    else
        echo "[darwin-wrapper] Build runner binary not created" >&2
        exit 1
    fi
fi

echo "[darwin-wrapper] Could not find build_zcu.o" >&2
cat "$STDERR_FILE" >&2
exit 1
