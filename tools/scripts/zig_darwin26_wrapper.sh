#!/bin/bash
# zig_darwin26_wrapper.sh — Wrapper for zig build on macOS 26+
# Works around Zig's MachO linker not finding libSystem symbols by:
# 1. Running zig build (which compiles .o but fails at link)
# 2. Relinking with Apple's system linker
# 3. Re-running zig build (which finds the cached binary)
#
# Usage: ./tools/scripts/zig_darwin26_wrapper.sh [zig build args...]

set -euo pipefail

ZIG="${ZIG:-$(which zig)}"
SYSROOT="${SDKROOT:-/Library/Developer/CommandLineTools/SDKs/MacOSX15.4.sdk}"
BUILD_CACHE=".zig-cache/o/51b3087bf5e2440cc614a856ab2390bf"
BUILD_O="$BUILD_CACHE/build_zcu.o"
BUILD_BIN="$BUILD_CACHE/build"

# Locate compiler_rt dynamically instead of hardcoding user path
COMPILER_RT="$(find "$HOME/.cache/zig/o" -name 'libcompiler_rt.a' -print -quit 2>/dev/null || true)"
if [[ -z "$COMPILER_RT" ]]; then
    echo "ERROR: Could not find libcompiler_rt.a in $HOME/.cache/zig/o/" >&2
    exit 1
fi

# Step 1: Try zig build. If it succeeds, great. If it fails at link, relink.
if "$ZIG" build "$@" 2>/dev/null; then
    exit 0
fi

# Step 2: Check if the .o file exists (compilation succeeded, link failed)
if [[ ! -f "$BUILD_O" ]]; then
    echo "ERROR: Build failed before producing object file. Not a linker issue." >&2
    "$ZIG" build "$@"  # Re-run to show the actual error
    exit 1
fi

echo "[darwin26-fix] Relinking build runner with Apple ld..." >&2

# Step 3: Link with system ld
/usr/bin/ld -dynamic \
    -platform_version macos 26.4.0 26.4 \
    -syslibroot "$SYSROOT" \
    -e _main \
    -o "$BUILD_BIN" \
    "$BUILD_O" \
    -lSystem \
    "$COMPILER_RT" || {
    echo "ERROR: System linker also failed." >&2
    exit 1
}

echo "[darwin26-fix] Relink successful. Re-running zig build..." >&2

# Step 4: Re-run zig build with the now-linked binary
exec "$ZIG" build "$@"
