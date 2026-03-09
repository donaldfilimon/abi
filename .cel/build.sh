#!/usr/bin/env bash
set -euo pipefail

# .cel/build.sh — Build the repo-local CEL Zig toolchain and ZLS.
#
# Usage:
#   .cel/build.sh              Build Zig, then build ZLS with that Zig
#   .cel/build.sh --zig-only   Build only the CEL Zig toolchain
#   .cel/build.sh --zls-only   Build only ZLS using .cel/bin/zig
#   .cel/build.sh --patch-only Clone Zig source and apply patches only
#   .cel/build.sh --verify     Print the current Zig and ZLS status
#   .cel/build.sh --status     Print source/pin/binary status
#   .cel/build.sh --clean      Remove generated state before building

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh"

ZIG_SRC_DIR="$CEL_DIR/.src"
ZLS_SRC_DIR="$CEL_DIR/.zls-src"
PATCHES_DIR="$CEL_DIR/patches"
BIN_DIR="$CEL_DIR/bin"
LIB_DIR="$CEL_DIR/lib"
BUILD_DIR="$ZIG_SRC_DIR/build"
BOOTSTRAP_LLVM_DIR="$CEL_DIR/../zig-bootstrap-emergency/out/build-llvm-host"

info()  { printf "\033[1;34m[cel]\033[0m %s\n" "$*"; }
warn()  { printf "\033[1;33m[cel]\033[0m %s\n" "$*"; }
error() { printf "\033[1;31m[cel]\033[0m %s\n" "$*" >&2; }
die()   { error "$@"; exit 1; }

usage() {
    cat <<'USAGE'
Usage: .cel/build.sh [OPTIONS]

Options:
  --clean        Remove generated CEL source/build/bin state first
  --patch-only   Clone Zig source and apply patches without building
  --verify       Print the current Zig and ZLS status
  --status       Show source, patches, pins, and binary status
  --zig-only     Build only the CEL Zig toolchain
  --zls-only     Build only ZLS using .cel/bin/zig
  -h, --help     Show this help text

Environment:
  CMAKE_JOBS     Number of parallel Zig build jobs
USAGE
    exit 0
}

binary_name() {
    if [[ "$(uname -s)" == "MINGW"* || "$(uname -s)" == "MSYS"* || "$(uname -s)" == "CYGWIN"* ]]; then
        printf '%s.exe' "$1"
    else
        printf '%s' "$1"
    fi
}

ZIG_EXE="$(binary_name zig)"
ZLS_EXE="$(binary_name zls)"
ZIG_BIN="$BIN_DIR/$ZIG_EXE"
ZLS_BIN="$BIN_DIR/$ZLS_EXE"

CLEAN=false
PATCH_ONLY=false
VERIFY_ONLY=false
STATUS_ONLY=false
BUILD_ZIG=true
BUILD_ZLS=true

while [[ $# -gt 0 ]]; do
    case "$1" in
        --clean) CLEAN=true; shift ;;
        --patch-only) PATCH_ONLY=true; shift ;;
        --verify) VERIFY_ONLY=true; shift ;;
        --status) STATUS_ONLY=true; shift ;;
        --zig-only) BUILD_ZIG=true; BUILD_ZLS=false; shift ;;
        --zls-only) BUILD_ZIG=false; BUILD_ZLS=true; shift ;;
        -h|--help) usage ;;
        *) die "Unknown option: $1" ;;
    esac
done

print_binary_status() {
    local label="$1"
    local path="$2"
    local version_args=("${@:3}")

    if [[ -x "$path" ]]; then
        local version
        version="$("$path" "${version_args[@]}" 2>/dev/null | tr -d '\r' | head -n 1 || true)"
        if [[ -n "$version" ]]; then
            info "$label: $path ($version)"
        else
            info "$label: $path"
        fi
    else
        warn "$label: not built ($path)"
    fi
}

print_status() {
    printf '\n'
    info "=== CEL Toolchain Status ==="
    printf '\n'

    info "Pinned Zig repo:    $ZIG_UPSTREAM_REPO"
    info "Pinned Zig commit:  $ZIG_UPSTREAM_COMMIT"
    info "Pinned Zig version: $ZIG_VERSION"
    info "Pinned ZLS repo:    $ZLS_UPSTREAM_REPO"

    if [[ -d "$ZIG_SRC_DIR/.git" ]]; then
        local src_commit
        local src_dirty
        src_commit="$(git -C "$ZIG_SRC_DIR" rev-parse --short=9 HEAD)"
        if git -C "$ZIG_SRC_DIR" diff --quiet && git -C "$ZIG_SRC_DIR" diff --cached --quiet; then
            src_dirty="clean"
        else
            src_dirty="dirty"
        fi
        info "Zig source:         $ZIG_SRC_DIR ($src_commit, $src_dirty)"
    else
        warn "Zig source:         not cloned ($ZIG_SRC_DIR)"
    fi

    if [[ -d "$ZLS_SRC_DIR/.git" ]]; then
        local zls_commit
        zls_commit="$(git -C "$ZLS_SRC_DIR" rev-parse --short=9 HEAD)"
        info "ZLS source:         $ZLS_SRC_DIR ($zls_commit)"
    else
        warn "ZLS source:         not cloned ($ZLS_SRC_DIR)"
    fi

    if [[ -d "$PATCHES_DIR" ]]; then
        local patch_count=0
        local patch
        for patch in "$PATCHES_DIR"/*.patch; do
            [[ -f "$patch" ]] || continue
            patch_count=$((patch_count + 1))
            info "Patch:              $(basename "$patch")"
        done
        info "Patch count:        $patch_count"
    else
        warn "Patch directory:    missing ($PATCHES_DIR)"
    fi

    print_binary_status "CEL Zig" "$ZIG_BIN" version
    print_binary_status "CEL ZLS" "$ZLS_BIN" --version
    printf '\n'
}

if $VERIFY_ONLY || $STATUS_ONLY; then
    print_status
    if $VERIFY_ONLY; then
        [[ -x "$ZIG_BIN" ]] || die "No Zig binary at $ZIG_BIN"
    fi
    exit 0
fi

for cmd in git cmake cc c++; do
    command -v "$cmd" >/dev/null 2>&1 || die "Required command not found: $cmd"
done

if $CLEAN; then
    info "Cleaning CEL-generated state..."
    rm -rf "$ZIG_SRC_DIR" "$ZLS_SRC_DIR" "$BIN_DIR" "$LIB_DIR"
fi

mkdir -p "$BIN_DIR"

clone_or_sync_zig_source() {
    if [[ ! -d "$ZIG_SRC_DIR/.git" ]]; then
        info "Cloning Zig source at $ZIG_UPSTREAM_COMMIT..."
        git clone "$ZIG_UPSTREAM_REPO" "$ZIG_SRC_DIR" >/dev/null 2>&1 || die "Failed to clone Zig source."
        git -C "$ZIG_SRC_DIR" checkout "$ZIG_UPSTREAM_COMMIT" >/dev/null 2>&1 || die "Failed to checkout $ZIG_UPSTREAM_COMMIT."
        return
    fi

    local current_commit
    current_commit="$(git -C "$ZIG_SRC_DIR" rev-parse --short=9 HEAD)"
    if [[ "$current_commit" == "$ZIG_UPSTREAM_COMMIT"* ]]; then
        info "Zig source already present at $current_commit"
        return
    fi

    if ! git -C "$ZIG_SRC_DIR" diff --quiet || ! git -C "$ZIG_SRC_DIR" diff --cached --quiet; then
        die "Zig source at $ZIG_SRC_DIR is dirty and not at $ZIG_UPSTREAM_COMMIT. Re-run with --clean."
    fi

    info "Updating Zig source from $current_commit to $ZIG_UPSTREAM_COMMIT..."
    git -C "$ZIG_SRC_DIR" fetch --depth 1 origin "$ZIG_UPSTREAM_COMMIT" >/dev/null 2>&1 || die "Failed to fetch Zig commit $ZIG_UPSTREAM_COMMIT."
    git -C "$ZIG_SRC_DIR" checkout "$ZIG_UPSTREAM_COMMIT" >/dev/null 2>&1 || die "Failed to checkout $ZIG_UPSTREAM_COMMIT."
}

apply_patches() {
    local patch
    for patch in "$PATCHES_DIR"/*.patch; do
        [[ -f "$patch" ]] || continue
        info "Applying patch: $(basename "$patch")"
        if git -C "$ZIG_SRC_DIR" apply --check "$patch" >/dev/null 2>&1; then
            git -C "$ZIG_SRC_DIR" apply "$patch" || die "Failed to apply $(basename "$patch")"
            continue
        fi
        if git -C "$ZIG_SRC_DIR" apply --reverse --check "$patch" >/dev/null 2>&1; then
            warn "Patch already applied: $(basename "$patch")"
            continue
        fi
        die "Patch does not apply cleanly: $(basename "$patch")"
    done
}

configure_cmake_args() {
    CMAKE_ARGS=(
        -S "$ZIG_SRC_DIR"
        -B "$BUILD_DIR"
        -DCMAKE_BUILD_TYPE="${CEL_BUILD_TYPE:-Release}"
        -DCMAKE_INSTALL_PREFIX="$CEL_DIR"
    )

    local bootstrap_llvm_config=""
    if [[ -f "$BOOTSTRAP_LLVM_DIR/bin/llvm-config" && -x "$BOOTSTRAP_LLVM_DIR/bin/llvm-config" ]]; then
        bootstrap_llvm_config="$BOOTSTRAP_LLVM_DIR/bin/llvm-config"
    elif [[ -f "$BOOTSTRAP_LLVM_DIR/tools/llvm-config" && -x "$BOOTSTRAP_LLVM_DIR/tools/llvm-config" ]]; then
        bootstrap_llvm_config="$BOOTSTRAP_LLVM_DIR/tools/llvm-config"
    fi

    if [[ -d "$BOOTSTRAP_LLVM_DIR" && -n "$bootstrap_llvm_config" ]]; then
        info "Reusing bootstrap LLVM artifacts from $BOOTSTRAP_LLVM_DIR"
        local bootstrap_llvm_path=""
        if [[ -d "$BOOTSTRAP_LLVM_DIR/tools" ]]; then
            bootstrap_llvm_path="$BOOTSTRAP_LLVM_DIR/tools"
        fi
        if [[ -d "$BOOTSTRAP_LLVM_DIR/bin" ]]; then
            if [[ -n "$bootstrap_llvm_path" ]]; then
                bootstrap_llvm_path="$bootstrap_llvm_path:$BOOTSTRAP_LLVM_DIR/bin"
            else
                bootstrap_llvm_path="$BOOTSTRAP_LLVM_DIR/bin"
            fi
        fi
        if [[ -n "$bootstrap_llvm_path" ]]; then
            export PATH="$bootstrap_llvm_path:$PATH"
        fi
        CMAKE_ARGS+=(
            -DZIG_STATIC_LLVM=ON
            -DLLVM_DIR="$BOOTSTRAP_LLVM_DIR/lib/cmake/llvm"
            -DLLD_DIR="$BOOTSTRAP_LLVM_DIR/lib/cmake/lld"
            -DCLANG_DIR="$BOOTSTRAP_LLVM_DIR/lib/cmake/clang"
        )
        return
    elif [[ -d "$BOOTSTRAP_LLVM_DIR" ]]; then
        warn "Ignoring bootstrap LLVM at $BOOTSTRAP_LLVM_DIR because no executable llvm-config was found."
    fi

    if [[ "$(uname -s)" == "Darwin" ]]; then
        local brew_llvm=""
        local candidate=""
        for candidate in \
            "/opt/homebrew/opt/llvm@21" \
            "/usr/local/opt/llvm@21" \
            "/opt/homebrew/opt/llvm" \
            "/usr/local/opt/llvm"; do
            if [[ -d "$candidate" ]]; then
                brew_llvm="$candidate"
                break
            fi
        done

        if [[ -z "$brew_llvm" ]] && command -v brew >/dev/null 2>&1; then
            for candidate in llvm@21 llvm; do
                brew_llvm="$(brew --prefix "$candidate" 2>/dev/null || true)"
                if [[ -n "$brew_llvm" && -d "$brew_llvm" ]]; then
                    break
                fi
            done
        fi

        if [[ -n "$brew_llvm" && -d "$brew_llvm" ]]; then
            if [[ "$brew_llvm" == *"llvm@21"* ]]; then
                info "Using Homebrew LLVM 21 at $brew_llvm"
            else
                warn "Using Homebrew LLVM at $brew_llvm; Zig $ZIG_UPSTREAM_COMMIT expects LLVM 21.x."
            fi
            CMAKE_ARGS+=(
                -DCMAKE_PREFIX_PATH="$brew_llvm;/opt/homebrew"
                -DLLVM_DIR="$brew_llvm/lib/cmake/llvm"
                -DLLD_DIR="$brew_llvm/lib/cmake/lld"
                -DCLANG_DIR="$brew_llvm/lib/cmake/clang"
                -DCMAKE_EXE_LINKER_FLAGS="-L/opt/homebrew/lib"
                -DCMAKE_SHARED_LINKER_FLAGS="-L/opt/homebrew/lib"
            )
        else
            warn "No bootstrap LLVM or compatible Homebrew LLVM detected. Install llvm@21 or build zig-bootstrap-emergency first."
        fi
    fi
}

build_zig() {
    clone_or_sync_zig_source
    apply_patches

    if $PATCH_ONLY; then
        info "Patch-only mode complete."
        return
    fi

    mkdir -p "$BUILD_DIR"
    configure_cmake_args

    info "Configuring Zig build..."
    cmake "${CMAKE_ARGS[@]}" || die "cmake configuration failed."

    local jobs
    jobs="${CMAKE_JOBS:-$(( ($(sysctl -n hw.ncpu 2>/dev/null || nproc 2>/dev/null || echo 4) + 1) / 2 ))}"
    [[ "$jobs" -lt 1 ]] && jobs=1

    info "Building Zig with $jobs parallel jobs..."
    cmake --build "$BUILD_DIR" --parallel "$jobs" || die "Zig build failed."

    info "Installing Zig into $CEL_DIR..."
    cmake --install "$BUILD_DIR" --prefix "$CEL_DIR" || die "Zig install failed."

    [[ -x "$ZIG_BIN" ]] || die "Zig build finished without $ZIG_BIN"
    info "Built CEL Zig: $("$ZIG_BIN" version 2>/dev/null || echo unknown)"

    local verify_dir
    verify_dir="$(mktemp -d)"
    local verify_src="$verify_dir/verify.zig"
    cat > "$verify_src" <<'ZIG'
pub fn main() void {}
ZIG
    if "$ZIG_BIN" build-exe "$verify_src" --name verify -femit-bin="$verify_dir/verify" >/dev/null 2>&1; then
        info "Verified CEL Zig can compile a trivial program."
    else
        warn "CEL Zig built, but trivial compile verification failed."
    fi
    rm -rf "$verify_dir"
}

clone_or_sync_zls_source() {
    if [[ ! -d "$ZLS_SRC_DIR/.git" ]]; then
        info "Cloning ZLS source..."
        git clone "$ZLS_UPSTREAM_REPO" "$ZLS_SRC_DIR" >/dev/null 2>&1 || die "Failed to clone ZLS source."
        return
    fi

    if ! git -C "$ZLS_SRC_DIR" diff --quiet || ! git -C "$ZLS_SRC_DIR" diff --cached --quiet; then
        die "ZLS source at $ZLS_SRC_DIR is dirty. Re-run with --clean."
    fi

    info "Refreshing ZLS source..."
    git -C "$ZLS_SRC_DIR" fetch --depth 1 origin >/dev/null 2>&1 || die "Failed to fetch ZLS source."
    git -C "$ZLS_SRC_DIR" pull --ff-only >/dev/null 2>&1 || die "Failed to fast-forward ZLS source."
}

build_zls() {
    [[ -x "$ZIG_BIN" ]] || die "Build Zig first; expected compiler at $ZIG_BIN"
    clone_or_sync_zls_source

    info "Building ZLS with CEL Zig..."
    (
        cd "$ZLS_SRC_DIR"
        "$ZIG_BIN" build -Doptimize=ReleaseFast
    ) || die "ZLS build failed."

    [[ -x "$ZLS_SRC_DIR/zig-out/bin/$ZLS_EXE" ]] || die "ZLS build finished without zig-out/bin/$ZLS_EXE"
    cp "$ZLS_SRC_DIR/zig-out/bin/$ZLS_EXE" "$ZLS_BIN"
    chmod +x "$ZLS_BIN"
    info "Built CEL ZLS: $("$ZLS_BIN" --version 2>/dev/null | head -n 1 || echo unknown)"
}

if $BUILD_ZIG; then
    build_zig
else
    clone_or_sync_zig_source
    apply_patches
fi

if $PATCH_ONLY; then
    exit 0
fi

if $BUILD_ZLS; then
    build_zls
fi

print_status
info "Done. To activate: eval \"\$(./tools/scripts/use_cel.sh)\""
