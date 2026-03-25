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
AUTO_LINK=false

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
TEST_STDERR="$(mktemp)"
trap 'rm -f "$STDERR_FILE" "$TEST_STDERR"' EXIT

build_runner_arch() {
    local build_bin="$1"
    local archs
    local file_out

    archs="$(lipo -archs "$build_bin" 2>/dev/null || true)"
    if [[ -n "$archs" ]]; then
        set -- $archs
        printf '%s\n' "$1"
        return 0
    fi

    file_out="$(file -b "$build_bin" 2>/dev/null || true)"
    case "$file_out" in
        *arm64*|*aarch64*)
            printf 'arm64\n'
            return 0
            ;;
        *x86_64*)
            printf 'x86_64\n'
            return 0
            ;;
    esac

    # Fallback: check the .o file in the same directory (unlinked build runner)
    local build_o="${build_bin}_zcu.o"
    if [[ -f "$build_o" ]]; then
        file_out="$(file -b "$build_o" 2>/dev/null || true)"
        case "$file_out" in
            *arm64*|*aarch64*)
                printf 'arm64\n'
                return 0
                ;;
            *x86_64*)
                printf 'x86_64\n'
                return 0
                ;;
        esac
    fi

    return 1
}

archive_matches_arch() {
    local archive="$1"
    local expected_arch="$2"
    local info
    local arch_list
    local arch

    info="$(lipo -info "$archive" 2>/dev/null || true)"
    [[ -z "$info" ]] && return 1

    case "$info" in
        *" are: "*)
            arch_list="${info##*: }"
            ;;
        *" architecture: "*)
            arch_list="${info##*: }"
            ;;
        *)
            return 1
            ;;
    esac

    for arch in $arch_list; do
        if [[ "$arch" == "$expected_arch" ]]; then
            return 0
        fi
    done

    return 1
}

find_host_compiler_rt() {
    local expected_arch="$1"
    local archive

    while IFS= read -r archive; do
        [[ -z "$archive" ]] && continue
        if archive_matches_arch "$archive" "$expected_arch"; then
            printf '%s\n' "$archive"
            return 0
        fi
    done < <(ls -t "$HOME/.cache/zig/o/"*/libcompiler_rt.a 2>/dev/null || true)

    return 1
}

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

# Find the newest build_zcu.o from the current cache.
BUILD_O="$(ls -t .zig-cache/o/*/build_zcu.o 2>/dev/null | head -1 || true)"

if [[ -n "$BUILD_O" && -f "$BUILD_O" ]]; then
    BUILD_DIR="$(dirname "$BUILD_O")"
    BUILD_BIN="$BUILD_DIR/build"
    BUILD_ARCH="$(build_runner_arch "$BUILD_BIN" || true)"

    if [[ -z "$BUILD_ARCH" ]]; then
        echo "[darwin-wrapper] Could not determine build runner architecture from $BUILD_BIN" >&2
        cat "$STDERR_FILE" >&2
        exit 1
    fi

    CRT="$(find_host_compiler_rt "$BUILD_ARCH" || true)"
    if [[ -z "$CRT" || ! -f "$CRT" ]]; then
        echo "[darwin-wrapper] No host-compatible libcompiler_rt.a found for build runner architecture $BUILD_ARCH" >&2
        cat "$STDERR_FILE" >&2
        exit 1
    fi

    echo "[darwin-wrapper] Relinking $BUILD_BIN with Apple ld + compiler_rt ($BUILD_ARCH)..." >&2

    LD_ARGS=(-dynamic -platform_version macos "$MACOS_VER" "$MACOS_VER" -syslibroot "$SYSROOT" -e _main -o "$BUILD_BIN" "$BUILD_O")
    LD_ARGS+=("$CRT")
    LD_ARGS+=(-lSystem)

    /usr/bin/ld "${LD_ARGS[@]}" 2>&1

    if [[ -x "$BUILD_BIN" ]]; then
        echo "[darwin-wrapper] Build runner relinked. Running build..." >&2
        if "$BUILD_BIN" "$ZIG2" "$ZIG_LIB" "$(pwd)" ".zig-cache" "${HOME}/.cache/zig" "$@" 2>"$TEST_STDERR"; then
            rm -f "$TEST_STDERR"
            if [ "$AUTO_LINK" = true ]; then
                "$SCRIPT_DIR/tools/zigup.sh" --link 2>&1 || true
            fi
            exit 0
        fi

        # Check if failure is due to Accelerate/vDSP link errors
        if grep -q "undefined symbol:.*vDSP\|undefined symbol:.*vvexpf\|undefined symbol:.*vvsqrtf" "$TEST_STDERR" 2>/dev/null; then
            echo "[darwin-wrapper] LLD cannot resolve Accelerate symbols on macOS 26.4+." >&2
            echo "[darwin-wrapper] Retrying with -Dfeat-gpu=false ..." >&2
            rm -f "$TEST_STDERR"
            "$BUILD_BIN" "$ZIG2" "$ZIG_LIB" "$(pwd)" ".zig-cache" "${HOME}/.cache/zig" -Dfeat-gpu=false "$@"
            EXIT_CODE=$?
            if [ "$AUTO_LINK" = true ]; then
                "$SCRIPT_DIR/tools/zigup.sh" --link 2>&1 || true
            fi
            exit $EXIT_CODE
        fi

        # Other failure — print captured stderr and exit
        cat "$TEST_STDERR" >&2
        rm -f "$TEST_STDERR"
        exit 1
    else
        echo "[darwin-wrapper] Build runner binary not created" >&2
        exit 1
    fi
fi

echo "[darwin-wrapper] Could not find build_zcu.o" >&2
cat "$STDERR_FILE" >&2
exit 1
