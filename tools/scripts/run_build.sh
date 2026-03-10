#!/bin/bash
# run_build.sh — Direct build runner invocation for macOS 26+
#
# Simple two-pass script:
#   1. zig build → fails at link → build_zcu.o exists
#   2. Relink .o with Apple's /usr/bin/ld → build runner binary
#   3. Execute build runner directly with the right args
#
# Usage: ./tools/scripts/run_build.sh [zig build args...]
#   e.g. ./tools/scripts/run_build.sh test --summary all
#        ./tools/scripts/run_build.sh lint
#        ./tools/scripts/run_build.sh full-check

set -euo pipefail

cd "$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"

ZIG="${ZIG_REAL:-${ZIG:-$(which zig)}}"
SYSROOT="${SDKROOT:-$(xcrun --show-sdk-path 2>/dev/null || echo /Library/Developer/CommandLineTools/SDKs/MacOSX.sdk)}"
MACOS_VER="$(sw_vers -productVersion 2>/dev/null || echo 26.0)"

find_compiler_rt() {
    local from_stderr
    from_stderr="$(grep -oE '/[^ )]*libcompiler_rt\.a' "$STDERR_FILE" | head -1 || true)"
    if [[ -n "$from_stderr" && -f "$from_stderr" ]]; then
        echo "$from_stderr"
    fi
}

# ── Step 1: Try zig build normally ───────────────────────────────────────
STDERR_FILE="$(mktemp)"
trap 'rm -f "$STDERR_FILE"' EXIT

if "$ZIG" build "$@" 2>"$STDERR_FILE"; then
    exit 0
fi

# ── Step 2: Find build_zcu.o ─────────────────────────────────────────────
BUILD_O="$(grep -oE '\.zig-cache/o/[a-f0-9]+/build_zcu\.o' "$STDERR_FILE" | head -1 || true)"
if [[ -z "$BUILD_O" ]]; then
    BUILD_O="$(find .zig-cache/o -name 'build_zcu.o' -newer "$STDERR_FILE" -print -quit 2>/dev/null || true)"
fi

if [[ -z "$BUILD_O" || ! -f "$BUILD_O" ]]; then
    echo "ERROR: Not a linker failure (no build_zcu.o). Original error:" >&2
    cat "$STDERR_FILE" >&2
    exit 1
fi

BUILD_DIR="$(dirname "$BUILD_O")"
BUILD_BIN="$BUILD_DIR/build"

# ── Step 3: Find compiler_rt ─────────────────────────────────────────────
COMPILER_RT="$(find_compiler_rt)"
RT_ARGS=()
if [[ -n "$COMPILER_RT" ]]; then
    RT_ARGS=("$COMPILER_RT")
fi

echo "[run_build] Relinking build runner with Apple ld..." >&2
echo "[run_build]   obj: $BUILD_O" >&2
echo "[run_build]   sdk: $SYSROOT" >&2

# ── Step 4: Link with Apple's ld ─────────────────────────────────────────
/usr/bin/ld -dynamic \
    -platform_version macos "$MACOS_VER" "$MACOS_VER" \
    -syslibroot "$SYSROOT" \
    -e _main \
    -o "$BUILD_BIN" \
    "$BUILD_O" \
    -lSystem \
    "${RT_ARGS[@]}" || {
    echo "ERROR: Apple ld also failed." >&2
    exit 1
}

# ── Step 5: Execute the build runner ─────────────────────────────────────
ZIG_LIB_DIR="$("$ZIG" env 2>/dev/null | grep '\.lib_dir' | sed 's/.*= *"\(.*\)".*/\1/' || true)"
if [[ -z "$ZIG_LIB_DIR" ]]; then
    ZIG_LIB_DIR="$(dirname "$(dirname "$ZIG")")/lib"
fi

echo "[run_build] Executing build runner..." >&2
exec "$BUILD_BIN" "$ZIG" "$ZIG_LIB_DIR" "$(pwd)" ".zig-cache" "${HOME}/.cache/zig" "$@"
