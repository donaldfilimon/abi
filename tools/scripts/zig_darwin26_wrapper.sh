#!/bin/bash
# zig_darwin26_wrapper.sh — Full zig shim for macOS 26+ (Tahoe)
#
# Zig 0.16-dev's MachO linker cannot resolve libSystem symbols on macOS 26+.
# This wrapper intercepts zig invocations and, on link failure, relinks using
# Apple's /usr/bin/ld. Works for both the build runner and compile targets.
#
# Usage:
#   export PATH="$(pwd)/tools/scripts:$PATH"  # if symlinked as 'zig'
#   # or
#   ZIG=./tools/scripts/zig_darwin26_wrapper.sh zig build test
#
# The wrapper is a no-op on non-Darwin or Darwin < 26.

set -euo pipefail

# Find the real zig binary (skip ourselves if on PATH)
SELF="$(realpath "${BASH_SOURCE[0]}" 2>/dev/null || readlink -f "${BASH_SOURCE[0]}")"
find_real_zig() {
    # Prefer ZIG_REAL if explicitly set
    if [[ -n "${ZIG_REAL:-}" ]]; then echo "$ZIG_REAL"; return; fi
    # Walk PATH, skip our own directory
    local self_dir; self_dir="$(dirname "$SELF")"
    local IFS=:
    for dir in $PATH; do
        [[ "$dir" == "$self_dir" ]] && continue
        [[ -x "$dir/zig" ]] && { echo "$dir/zig"; return; }
    done
    echo "zig"  # fallback
}

ZIG="$(find_real_zig)"
SYSROOT="${SDKROOT:-$(xcrun --show-sdk-path 2>/dev/null || echo /Library/Developer/CommandLineTools/SDKs/MacOSX.sdk)}"
MACOS_VER="$(sw_vers -productVersion 2>/dev/null || echo 26.0)"
MACOS_MAJOR="${MACOS_VER%%.*}"

# Pass through directly if not macOS 26+
if [[ "$(uname -s)" != "Darwin" ]] || [[ "$MACOS_MAJOR" -lt 26 ]]; then
    exec "$ZIG" "$@"
fi

# ── Intercept and fix link failures ──────────────────────────────────────

STDERR_FILE="$(mktemp)"
trap 'rm -f "$STDERR_FILE"' EXIT

# Run the real zig command, capture stderr
if "$ZIG" "$@" 2>"$STDERR_FILE"; then
    exit 0
else
    ZIG_EXIT=$?
fi

# Check if this is a linker failure we can fix
if ! grep -qE '(undefined.*_arc4random_buf|undefined.*__availability_version|using LLD to link|MachO|lld-link)' "$STDERR_FILE" 2>/dev/null; then
    # Not a linker issue — pass through the original error
    cat "$STDERR_FILE" >&2
    exit $ZIG_EXIT
fi

# Find the .o file from zig-cache
find_object_file() {
    local pattern="$1"
    # Try extracting from stderr first
    local from_stderr
    from_stderr="$(grep -oE '\.zig-cache/o/[a-f0-9]+/'"$pattern" "$STDERR_FILE" | head -1 || true)"
    if [[ -n "$from_stderr" && -f "$from_stderr" ]]; then echo "$from_stderr"; return; fi
    # Fallback: most recent match in cache
    find .zig-cache/o -name "$pattern" -newer "$STDERR_FILE" -print 2>/dev/null | head -1
}

# Find compiler_rt
find_compiler_rt() {
    local from_stderr
    from_stderr="$(grep -oE '/[^ )]*libcompiler_rt\.a' "$STDERR_FILE" | head -1 || true)"
    if [[ -n "$from_stderr" && -f "$from_stderr" ]]; then
        echo "$from_stderr"
    fi
}

relink_with_apple_ld() {
    local obj="$1" output="$2"
    local compiler_rt; compiler_rt="$(find_compiler_rt)"

    local rt_args=()
    if [[ -n "$compiler_rt" ]]; then
        rt_args=("$compiler_rt")
    fi

    echo "[darwin26-fix] Relinking $(basename "$output") with Apple ld..." >&2
    /usr/bin/ld -dynamic \
        -platform_version macos "$MACOS_VER" "$MACOS_VER" \
        -syslibroot "$SYSROOT" \
        -e _main \
        -o "$output" \
        "$obj" \
        -lSystem \
        "${rt_args[@]}" || {
        echo "[darwin26-fix] Apple ld also failed:" >&2
        cat "$STDERR_FILE" >&2
        return 1
    }
}

# ── Handle "zig build" (build runner relink + re-exec) ───────────────────
if [[ "${1:-}" == "build" ]]; then
    BUILD_O="$(find_object_file 'build_zcu.o')"
    if [[ -z "$BUILD_O" || ! -f "$BUILD_O" ]]; then
        echo "[darwin26-fix] No build_zcu.o found — not a build runner link failure." >&2
        cat "$STDERR_FILE" >&2
        exit $ZIG_EXIT
    fi

    BUILD_DIR="$(dirname "$BUILD_O")"
    BUILD_BIN="$BUILD_DIR/build"

    relink_with_apple_ld "$BUILD_O" "$BUILD_BIN" || exit 1

    # Determine zig lib dir
    ZIG_LIB_DIR="$("$ZIG" env 2>/dev/null | grep '\.lib_dir' | sed 's/.*= *"\(.*\)".*/\1/' || true)"
    if [[ -z "$ZIG_LIB_DIR" ]]; then
        ZIG_LIB_DIR="$(dirname "$(dirname "$ZIG")")/lib"
    fi

    BUILD_ROOT="$(pwd)"
    CACHE_ROOT="$BUILD_ROOT/.zig-cache"
    GLOBAL_CACHE="${HOME}/.cache/zig"

    echo "[darwin26-fix] Executing build runner directly..." >&2
    shift  # remove "build"
    exec "$BUILD_BIN" "$ZIG" "$ZIG_LIB_DIR" "$BUILD_ROOT" "$CACHE_ROOT" "$GLOBAL_CACHE" "$@"
fi

# ── Handle other zig commands (build-exe, test, etc.) ────────────────────
# For non-build commands, we can't easily relink since we don't know the
# output path. Print guidance instead.
echo "[darwin26-fix] Zig linker failed on macOS $MACOS_VER." >&2
echo "[darwin26-fix] For 'zig build', this wrapper handles relinking automatically." >&2
echo "[darwin26-fix] For direct 'zig test/build-exe', use -fno-emit-bin for compile-only." >&2
cat "$STDERR_FILE" >&2
exit $ZIG_EXIT
