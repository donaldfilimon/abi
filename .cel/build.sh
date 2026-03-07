#!/usr/bin/env bash
set -euo pipefail

# .cel/build.sh — Build a patched Zig toolchain from source
#
# Usage:
#   ./cel/build.sh              Build patched Zig from source
#   ./cel/build.sh --clean      Wipe .src/ and rebuild from scratch
#   ./cel/build.sh --patch-only Clone and apply patches without building

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh"

SRC_DIR="$CEL_DIR/.src"
PATCHES_DIR="$CEL_DIR/patches"
BIN_DIR="$CEL_DIR/bin"
BUILD_DIR="$SRC_DIR/build"

# Bootstrap LLVM artifacts (reuse from zig-bootstrap-emergency if available)
BOOTSTRAP_LLVM_DIR="$CEL_DIR/../zig-bootstrap-emergency/out/build-llvm-host"

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

info()  { printf "\033[1;34m[cel]\033[0m %s\n" "$*"; }
warn()  { printf "\033[1;33m[cel]\033[0m %s\n" "$*"; }
error() { printf "\033[1;31m[cel]\033[0m %s\n" "$*" >&2; }
die()   { error "$@"; exit 1; }

usage() {
    cat <<'USAGE'
Usage: .cel/build.sh [OPTIONS]

Options:
  --clean        Remove .src/ directory and rebuild from scratch
  --patch-only   Clone repo and apply patches, but do not build
  -h, --help     Show this help message

Environment:
  CMAKE_JOBS     Number of parallel build jobs (default: nproc/2)
USAGE
    exit 0
}

# --------------------------------------------------------------------------- #
# Parse arguments
# --------------------------------------------------------------------------- #

CLEAN=false
PATCH_ONLY=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --clean)      CLEAN=true; shift ;;
        --patch-only) PATCH_ONLY=true; shift ;;
        -h|--help)    usage ;;
        *)            die "Unknown option: $1 (use --help)" ;;
    esac
done

# --------------------------------------------------------------------------- #
# Prerequisites
# --------------------------------------------------------------------------- #

for cmd in git cmake cc c++; do
    command -v "$cmd" >/dev/null 2>&1 || die "Required command not found: $cmd"
done

# --------------------------------------------------------------------------- #
# Clean (if requested)
# --------------------------------------------------------------------------- #

if $CLEAN; then
    info "Cleaning .src/ directory..."
    rm -rf "$SRC_DIR"
    rm -rf "$BIN_DIR"
    info "Clean complete."
fi

# --------------------------------------------------------------------------- #
# Clone upstream at pinned commit
# --------------------------------------------------------------------------- #

if [[ ! -d "$SRC_DIR/.git" ]]; then
    info "Cloning Zig upstream at commit $ZIG_UPSTREAM_COMMIT..."
    git clone "$ZIG_UPSTREAM_REPO" "$SRC_DIR" 2>&1 | tail -1
    (
        cd "$SRC_DIR"
        git checkout "$ZIG_UPSTREAM_COMMIT" 2>&1 | tail -1
    )
    info "Clone complete."
else
    info "Source already cloned at $SRC_DIR"
    # Verify we're on the right commit
    CURRENT_COMMIT="$(cd "$SRC_DIR" && git rev-parse --short=9 HEAD)"
    if [[ "$CURRENT_COMMIT" != "$ZIG_UPSTREAM_COMMIT"* ]]; then
        warn "Source is at $CURRENT_COMMIT, expected $ZIG_UPSTREAM_COMMIT"
        warn "Run with --clean to re-clone, or manually checkout the correct commit."
    fi
fi

# --------------------------------------------------------------------------- #
# Apply patches in lexicographic order
# --------------------------------------------------------------------------- #

if [[ -d "$PATCHES_DIR" ]]; then
    PATCH_COUNT=0
    for patch in "$PATCHES_DIR"/*.patch; do
        [[ -f "$patch" ]] || continue

        # Skip placeholder patches (files that contain only comments)
        if ! grep -q '^[^#]' "$patch" 2>/dev/null; then
            info "Skipping placeholder patch: $(basename "$patch")"
            continue
        fi

        info "Applying patch: $(basename "$patch")"
        (
            cd "$SRC_DIR"
            git apply --check "$patch" 2>/dev/null || {
                warn "Patch $(basename "$patch") does not apply cleanly (may already be applied)."
                return 0 2>/dev/null || true
            }
            git apply "$patch"
        )
        PATCH_COUNT=$((PATCH_COUNT + 1))
    done
    info "Applied $PATCH_COUNT patch(es)."
else
    warn "No patches directory found at $PATCHES_DIR"
fi

if $PATCH_ONLY; then
    info "Patch-only mode: skipping build."
    info "Source is ready at: $SRC_DIR"
    exit 0
fi

# --------------------------------------------------------------------------- #
# Configure cmake
# --------------------------------------------------------------------------- #

mkdir -p "$BUILD_DIR"
mkdir -p "$BIN_DIR"

CMAKE_ARGS=(
    -S "$SRC_DIR"
    -B "$BUILD_DIR"
    -DCMAKE_BUILD_TYPE=Release
    -DCMAKE_INSTALL_PREFIX="$BIN_DIR"
)

# Detect and reuse bootstrap LLVM artifacts if available
if [[ -d "$BOOTSTRAP_LLVM_DIR" ]]; then
    info "Found bootstrap LLVM artifacts at $BOOTSTRAP_LLVM_DIR"
    CMAKE_ARGS+=(
        -DZIG_STATIC_LLVM=ON
        -DLLVM_DIR="$BOOTSTRAP_LLVM_DIR/lib/cmake/llvm"
        -DLLD_DIR="$BOOTSTRAP_LLVM_DIR/lib/cmake/lld"
        -DCLANG_DIR="$BOOTSTRAP_LLVM_DIR/lib/cmake/clang"
    )
else
    info "No bootstrap LLVM found; cmake will find system LLVM."
    info "  (For static builds, run zig-bootstrap-emergency first, or install llvm via Homebrew.)"

    # Try Homebrew LLVM as fallback on macOS
    if [[ "$(uname -s)" == "Darwin" ]]; then
        BREW_LLVM="$(brew --prefix llvm 2>/dev/null || true)"
        if [[ -n "$BREW_LLVM" && -d "$BREW_LLVM" ]]; then
            info "Using Homebrew LLVM at $BREW_LLVM"
            CMAKE_ARGS+=(
                -DLLVM_DIR="$BREW_LLVM/lib/cmake/llvm"
                -DLLD_DIR="$BREW_LLVM/lib/cmake/lld"
                -DCLANG_DIR="$BREW_LLVM/lib/cmake/clang"
            )
        fi
    fi
fi

info "Configuring cmake..."
cmake "${CMAKE_ARGS[@]}" || die "cmake configuration failed. Check LLVM availability."

# --------------------------------------------------------------------------- #
# Build
# --------------------------------------------------------------------------- #

# Default to half of available cores
JOBS="${CMAKE_JOBS:-$(( $(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4) / 2 ))}"
[[ "$JOBS" -lt 1 ]] && JOBS=1

info "Building Zig with $JOBS parallel jobs..."
cmake --build "$BUILD_DIR" --parallel "$JOBS" || die "Build failed."

# --------------------------------------------------------------------------- #
# Install
# --------------------------------------------------------------------------- #

info "Installing to $BIN_DIR..."
cmake --install "$BUILD_DIR" --prefix "$BIN_DIR" || die "Install failed."

# Verify the binary exists
if [[ -x "$BIN_DIR/bin/zig" ]]; then
    # Move zig binary up so .cel/bin/zig works directly
    mv "$BIN_DIR/bin/zig" "$BIN_DIR/zig" 2>/dev/null || true
    # Also keep lib/ accessible
    if [[ -d "$BIN_DIR/lib" ]]; then
        info "Zig lib directory: $BIN_DIR/lib"
    fi
    info "Build successful!"
    info "Zig binary: $BIN_DIR/zig"
    "$BIN_DIR/zig" version 2>/dev/null && true
elif [[ -x "$BIN_DIR/zig" ]]; then
    info "Build successful!"
    info "Zig binary: $BIN_DIR/zig"
    "$BIN_DIR/zig" version 2>/dev/null && true
else
    die "Build appeared to succeed, but no zig binary found at $BIN_DIR"
fi

info "Done. To use: export PATH=\"$BIN_DIR:\$PATH\""
