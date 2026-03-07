#!/usr/bin/env bash
set -euo pipefail

# .cel/build.sh — Build a patched Zig toolchain from source
#
# Usage:
#   .cel/build.sh              Build patched Zig from source
#   .cel/build.sh --clean      Wipe .src/ and rebuild from scratch
#   .cel/build.sh --patch-only Clone and apply patches without building
#   .cel/build.sh --verify     Check if .cel/bin/zig exists and print version
#   .cel/build.sh --status     Show source, patches, binary, and version info

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
  --verify       Check if .cel/bin/zig exists and print its version
  --status       Show source dir, patches, binary, and version status
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
VERIFY_ONLY=false
STATUS_ONLY=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --clean)      CLEAN=true; shift ;;
        --patch-only) PATCH_ONLY=true; shift ;;
        --verify)     VERIFY_ONLY=true; shift ;;
        --status)     STATUS_ONLY=true; shift ;;
        -h|--help)    usage ;;
        *)            die "Unknown option: $1 (use --help)" ;;
    esac
done

# --------------------------------------------------------------------------- #
# --verify: check binary exists and print version, then exit
# --------------------------------------------------------------------------- #

if $VERIFY_ONLY; then
    if [[ -x "$BIN_DIR/zig" ]]; then
        info "Zig binary found: $BIN_DIR/zig"
        "$BIN_DIR/zig" version
        exit 0
    else
        die "No zig binary at $BIN_DIR/zig — run .cel/build.sh to build it."
    fi
fi

# --------------------------------------------------------------------------- #
# --status: show comprehensive toolchain status, then exit
# --------------------------------------------------------------------------- #

if $STATUS_ONLY; then
    echo ""
    info "=== .cel Toolchain Status ==="
    echo ""

    # Source directory
    if [[ -d "$SRC_DIR/.git" ]]; then
        SRC_COMMIT="$(cd "$SRC_DIR" && git rev-parse --short=9 HEAD)"
        SRC_DIRTY="$(cd "$SRC_DIR" && git diff --quiet && echo "clean" || echo "dirty")"
        info "Source dir:  $SRC_DIR (commit $SRC_COMMIT, $SRC_DIRTY)"
    else
        warn "Source dir:  NOT CLONED ($SRC_DIR)"
    fi

    # Expected commit
    info "Pinned commit: $ZIG_UPSTREAM_COMMIT (version $ZIG_VERSION)"

    # Commit match
    if [[ -d "$SRC_DIR/.git" ]]; then
        if [[ "$SRC_COMMIT" == "$ZIG_UPSTREAM_COMMIT"* ]]; then
            info "Commit match: YES"
        else
            warn "Commit match: NO (source is at $SRC_COMMIT, expected $ZIG_UPSTREAM_COMMIT)"
        fi
    fi

    # Patches
    if [[ -d "$PATCHES_DIR" ]]; then
        REAL_PATCHES=0
        for p in "$PATCHES_DIR"/*.patch; do
            [[ -f "$p" ]] || continue
            if grep -q '^[^#]' "$p" 2>/dev/null; then
                REAL_PATCHES=$((REAL_PATCHES + 1))
                info "  Patch: $(basename "$p")"
            fi
        done
        info "Patches:     $REAL_PATCHES active patch(es) in $PATCHES_DIR"
    else
        warn "Patches:     No patches directory"
    fi

    # Binary
    if [[ -x "$BIN_DIR/zig" ]]; then
        BUILT_VERSION="$("$BIN_DIR/zig" version 2>/dev/null || echo "unknown")"
        info "Binary:      $BIN_DIR/zig (version $BUILT_VERSION)"
        if [[ "$BUILT_VERSION" == "$ZIG_VERSION" ]]; then
            info "Version match: YES"
        else
            warn "Version match: NO (built=$BUILT_VERSION, expected=$ZIG_VERSION)"
        fi
    else
        warn "Binary:      NOT FOUND at $BIN_DIR/zig"
    fi

    echo ""
    exit 0
fi

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

    # Try Homebrew LLVM as fallback on macOS, with arch-aware path detection
    if [[ "$(uname -s)" == "Darwin" ]]; then
        BREW_LLVM=""
        ARCH="$(uname -m)"

        # Check arch-specific Homebrew prefix first, then fall back
        if [[ "$ARCH" == "arm64" ]]; then
            # Apple Silicon (M1/M2/M3/M4)
            if [[ -d "/opt/homebrew/opt/llvm" ]]; then
                BREW_LLVM="/opt/homebrew/opt/llvm"
            fi
        else
            # Intel Mac
            if [[ -d "/usr/local/opt/llvm" ]]; then
                BREW_LLVM="/usr/local/opt/llvm"
            fi
        fi

        # Fall back to `brew --prefix llvm` if arch-specific path wasn't found
        if [[ -z "$BREW_LLVM" ]]; then
            BREW_LLVM="$(brew --prefix llvm 2>/dev/null || true)"
        fi

        if [[ -n "$BREW_LLVM" && -d "$BREW_LLVM" ]]; then
            info "Found Homebrew LLVM at $BREW_LLVM ($ARCH)"
            CMAKE_ARGS+=(
                -DCMAKE_PREFIX_PATH="$BREW_LLVM"
                -DLLVM_DIR="$BREW_LLVM/lib/cmake/llvm"
                -DLLD_DIR="$BREW_LLVM/lib/cmake/lld"
                -DCLANG_DIR="$BREW_LLVM/lib/cmake/clang"
            )
        else
            warn "No Homebrew LLVM found. Checked:"
            if [[ "$ARCH" == "arm64" ]]; then
                warn "  /opt/homebrew/opt/llvm (Apple Silicon)"
            else
                warn "  /usr/local/opt/llvm (Intel)"
            fi
            warn "  Install with: brew install llvm"
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

# --------------------------------------------------------------------------- #
# Post-build verification: ensure the built zig can compile a trivial program
# --------------------------------------------------------------------------- #

ZIG_BIN="$BIN_DIR/zig"
if [[ -x "$ZIG_BIN" ]]; then
    info "Verifying built zig can compile a test program..."
    VERIFY_DIR="$(mktemp -d)"
    VERIFY_SRC="$VERIFY_DIR/verify.zig"
    cat > "$VERIFY_SRC" <<'ZIG'
pub fn main() void {}
ZIG
    if "$ZIG_BIN" build-exe "$VERIFY_SRC" --name verify -femit-bin="$VERIFY_DIR/verify" 2>/dev/null; then
        if [[ -x "$VERIFY_DIR/verify" ]]; then
            info "Verification passed: built zig can compile and link."
        else
            warn "Compilation ran but produced no executable — linking may still be broken."
        fi
    else
        warn "Verification FAILED: built zig could not compile a trivial program."
        warn "The binary exists but may have linker issues on this platform."
    fi
    rm -rf "$VERIFY_DIR"
fi

info "Done. To use: export PATH=\"$BIN_DIR:\$PATH\""
