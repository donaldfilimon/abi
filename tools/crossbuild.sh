#!/bin/sh
set -eu

# ABI Cross-Platform Build Tool
# Builds ABI for multiple platforms from a single command.
#
# Usage:
#   tools/crossbuild.sh --all          Build for all supported platforms
#   tools/crossbuild.sh linux          Build for Linux (aarch64 + x86_64)
#   tools/crossbuild.sh macos          Build for macOS (aarch64 + x86_64)
#   tools/crossbuild.sh windows        Build for Windows (x86_64)
#   tools/crossbuild.sh wasm           Build for WASM/WASI
#   tools/crossbuild.sh ios            Build for iOS (aarch64)
#   tools/crossbuild.sh android        Build for Android (aarch64)
#   tools/crossbuild.sh freebsd        Build for FreeBSD (x86_64 + aarch64)
#   tools/crossbuild.sh --list         List available targets
#   tools/crossbuild.sh --clean        Clean all build artifacts
#
# Options:
#   --optimize=Debug|ReleaseSafe|ReleaseFast  (default: Debug)
#   -Dfeat-<name>=false                        Disable feature flags

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ZIGLY="$SCRIPT_DIR/zigly"
OUT_BASE="$REPO_ROOT/zig-out"

OPTIMIZE="Debug"
EXTRA_ARGS=""
BUILT_ARTIFACTS=""

usage() {
    echo "Usage: crossbuild.sh [--all|--list|--clean|<platform>]"
    echo ""
    echo "Platforms:"
    echo "  linux     Linux (aarch64 + x86_64, musl)"
    echo "  macos     macOS (aarch64 + x86_64)"
    echo "  windows   Windows (x86_64, GNU)"
    echo "  wasm      WASM/WASI"
    echo "  ios       iOS (aarch64)"
    echo "  android   Android (aarch64)"
    echo "  freebsd   FreeBSD (x86_64 + aarch64)"
    echo ""
    echo "Options:"
    echo "  --all                        Build all platforms"
    echo "  --list                       List available targets"
    echo "  --clean                      Clean all build artifacts"
    echo "  --optimize=<mode>            Debug, ReleaseSafe, ReleaseFast"
    echo "  -Dfeat-<name>=<true|false>   Pass feature flags to zig build"
    exit 1
}

log() { echo "[crossbuild] $*" >&2; }
err() { echo "[crossbuild] ERROR: $*" >&2; exit 1; }

# Parse arguments
PLATFORM=""
while [ $# -gt 0 ]; do
    case "$1" in
        --all)     PLATFORM="all" ;;
        --list)    PLATFORM="list" ;;
        --clean)   PLATFORM="clean" ;;
        --optimize=*)
            OPTIMIZE="${1#--optimize=}"
            ;;
        -Dfeat-*)
            EXTRA_ARGS="$EXTRA_ARGS $1"
            ;;
        linux|macos|windows|wasm|ios|android|freebsd)
            PLATFORM="$1"
            ;;
        *)
            err "Unknown argument: $1"
            ;;
    esac
    shift
done

[ -z "$PLATFORM" ] && usage

# List targets
if [ "$PLATFORM" = "list" ]; then
    echo "Available cross-build targets:"
    echo ""
    printf "  %-12s %s\n" "Platform" "Targets"
    printf "  %-12s %s\n" "--------" "-------"
    printf "  %-12s %s\n" "linux"    "aarch64-linux-musl, x86_64-linux-musl"
    printf "  %-12s %s\n" "macos"    "aarch64-macos, x86_64-macos"
    printf "  %-12s %s\n" "windows"  "x86_64-windows-gnu"
    printf "  %-12s %s\n" "wasm"     "wasm32-wasi"
    printf "  %-12s %s\n" "ios"      "aarch64-ios"
    printf "  %-12s %s\n" "android"  "aarch64-linux-android"
    printf "  %-12s %s\n" "freebsd"  "x86_64-freebsd, aarch64-freebsd"
    exit 0
fi

# Clean
if [ "$PLATFORM" = "clean" ]; then
    log "Cleaning all build artifacts ..."
    rm -rf "$OUT_BASE" "$REPO_ROOT/.zig-cache"
    log "Done."
    exit 0
fi

# Resolve zig
ZIG="$("$ZIGLY" --status)"

resolve_zig_lib_dir() {
    local zig_bin="$1"
    local zig_real bin_dir candidate fallback

    zig_real="$(readlink -f "$zig_bin" 2>/dev/null || echo "$zig_bin")"
    bin_dir="$(dirname "$zig_real")"
    candidate="$bin_dir/lib"
    if [ -d "$candidate" ]; then
        printf '%s\n' "$candidate"
        return 0
    fi

    fallback="$(dirname "$bin_dir")/lib"
    printf '%s\n' "$fallback"
}

ZIG_LIB="$(resolve_zig_lib_dir "$ZIG")"

log "Using zig: $ZIG"

detect_host_os() {
    case "$(uname -s)" in
        Darwin) echo "macos" ;;
        Linux)  echo "linux" ;;
        *)      echo "unknown" ;;
    esac
}

detect_host_arch() {
    case "$(uname -m)" in
        arm64|aarch64)  echo "aarch64" ;;
        x86_64|amd64)  echo "x86_64" ;;
        *)              echo "unknown" ;;
    esac
}

HOST_OS="$(detect_host_os)"
HOST_ARCH="$(detect_host_arch)"

is_native_target() {
    # $1 = target triple. Returns 0 if this is the host platform.
    case "$1" in
        aarch64-macos)     [ "$HOST_OS" = "macos" ] && [ "$HOST_ARCH" = "aarch64" ] ;;
        x86_64-macos)     [ "$HOST_OS" = "macos" ] && [ "$HOST_ARCH" = "x86_64" ] ;;
        aarch64-linux-*)  [ "$HOST_OS" = "linux" ] && [ "$HOST_ARCH" = "aarch64" ] ;;
        x86_64-linux-*)   [ "$HOST_OS" = "linux" ] && [ "$HOST_ARCH" = "x86_64" ] ;;
        *)                return 1 ;;
    esac
}

# Build a single target
# $1 = target triple
# $2 = feature flags string (e.g. "-Dfeat-gpu=false -Dfeat-network=false")
build_target() {
    target="$1"
    feat_flags="$2"
    out_dir="$OUT_BASE/$target"

    log "Building $target (optimize=$OPTIMIZE) ..."

    # Build args
    args="--zig-lib-dir $ZIG_LIB --global-cache-dir $HOME/.cache/zig --cache-dir .zig-cache"
    args="$args -Dtarget=$target -Doptimize=$OPTIMIZE"
    args="$args $feat_flags $EXTRA_ARGS"

    # On macOS 26.4+, always use build.sh wrapper (LLD can't link the build runner)
    if [ "$HOST_OS" = "macos" ]; then
        log "  Using macOS build wrapper (Apple ld) ..."
        # shellcheck disable=SC2086
        "$REPO_ROOT/build.sh" lib $args 2>&1 | sed 's/^/  /'
        # Move output to target-specific directory
        if [ -d "$REPO_ROOT/zig-out/lib" ]; then
            mkdir -p "$out_dir/lib"
            cp -f "$REPO_ROOT/zig-out/lib/"* "$out_dir/lib/" 2>/dev/null || true
        fi
    else
        # shellcheck disable=SC2086
        "$ZIG" build lib $args --prefix "$out_dir" 2>&1 | sed 's/^/  /'
    fi

    if [ -d "$out_dir" ] || [ -f "$REPO_ROOT/zig-out/lib/"*.a ]; then
        BUILT_ARTIFACTS="$BUILT_ARTIFACTS $target:$out_dir"
        log "  $target -> $out_dir"
    else
        log "  WARNING: $target produced no output"
    fi
}

# WASM feature flags — disable OS-dependent features
WASM_FLAGS="-Dfeat-gpu=false -Dfeat-database=false -Dfeat-network=false -Dfeat-observability=false -Dfeat-web=false -Dfeat-pages=false -Dfeat-cloud=false -Dfeat-storage=false -Dfeat-compute=false -Dfeat-desktop=false -Dfeat-lsp=false -Dfeat-mcp=false"

# iOS feature flags — enable mobile, metal + opengles GPU
IOS_FLAGS="-Dfeat-mobile=true -Dgpu-backend=metal,opengles"

# Android feature flags — enable mobile, vulkan + opengles GPU
ANDROID_FLAGS="-Dfeat-mobile=true -Dgpu-backend=vulkan,opengles"

# Windows feature flags — no metal GPU
WINDOWS_FLAGS="-Dgpu-backend=cuda,vulkan,opengl,opengles,stdgpu"

# Build all targets for a platform
build_linux() {
    log "=== Linux ==="
    build_target "aarch64-linux-musl" ""
    build_target "x86_64-linux-musl" ""
}

build_macos() {
    log "=== macOS ==="
    build_target "aarch64-macos" ""
    build_target "x86_64-macos" ""
}

build_windows() {
    log "=== Windows ==="
    build_target "x86_64-windows-gnu" "$WINDOWS_FLAGS"
}

build_wasm() {
    log "=== WASM/WASI ==="
    build_target "wasm32-wasi" "$WASM_FLAGS"
}

build_ios() {
    log "=== iOS ==="
    build_target "aarch64-ios" "$IOS_FLAGS"
}

build_android() {
    log "=== Android ==="
    build_target "aarch64-linux-android" "$ANDROID_FLAGS"
}

build_freebsd() {
    log "=== FreeBSD ==="
    build_target "x86_64-freebsd" ""
    build_target "aarch64-freebsd" ""
}

# Print summary table
print_summary() {
    echo ""
    echo "=== Cross-Build Summary ==="
    echo ""
    printf "  %-30s %s\n" "Target" "Status"
    printf "  %-30s %s\n" "------------------------------" "------"

    for entry in $BUILT_ARTIFACTS; do
        target="${entry%%:*}"
        dir="${entry##*:}"
        # Find .a files
        libs=$(find "$dir" -name "*.a" -o -name "*.exe" 2>/dev/null | head -5)
        if [ -n "$libs" ]; then
            for lib in $libs; do
                size=$(du -h "$lib" 2>/dev/null | cut -f1)
                printf "  %-30s %s (%s)\n" "$target" "$(basename "$lib")" "$size"
            done
        else
            printf "  %-30s %s\n" "$target" "OK (no artifacts found)"
        fi
    done

    echo ""
    total=$(echo "$BUILT_ARTIFACTS" | wc -w | tr -d ' ')
    echo "  Total: $total target(s) built"
    echo ""
}

# Main dispatch
case "$PLATFORM" in
    all)
        build_linux
        build_macos
        build_windows
        build_wasm
        build_ios
        build_android
        build_freebsd
        ;;
    linux)   build_linux ;;
    macos)   build_macos ;;
    windows) build_windows ;;
    wasm)    build_wasm ;;
    ios)     build_ios ;;
    android) build_android ;;
    freebsd) build_freebsd ;;
    *)       usage ;;
esac

print_summary
