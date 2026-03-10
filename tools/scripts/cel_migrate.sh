#!/usr/bin/env bash
# cel_migrate.sh — Migrate from stock Zig to CEL toolchain
#
# This script handles the full migration path:
#   1. Check platform requirements
#   2. Verify prerequisites (git, cmake, LLVM)
#   3. Build the .cel toolchain (if not already built)
#   4. Verify the build
#   5. Activate CEL on PATH
#   6. Run basic validation
#
# Usage:
#   ./tools/scripts/cel_migrate.sh              # Full migration
#   ./tools/scripts/cel_migrate.sh --check      # Check only (no build)
#   ./tools/scripts/cel_migrate.sh --activate   # Activate only (already built)
#   ./tools/scripts/cel_migrate.sh --clean      # Clean rebuild
#
# Source this to activate in current shell:
#   eval "$(./tools/scripts/cel_migrate.sh --activate)"
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
CEL_DIR="$REPO_ROOT/.cel"
CEL_ZIG="$CEL_DIR/bin/zig"
CEL_ZLS="$CEL_DIR/bin/zls"
BOOTSTRAP_HOST_ZIG="$REPO_ROOT/zig-bootstrap-emergency/out/host/bin/zig"
EXPECTED_ZIG_VERSION="$(tr -d '[:space:]' < "$REPO_ROOT/.zigversion" 2>/dev/null || true)"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[1;34m'
NC='\033[0m' # No Color

info()  { printf "${BLUE}[cel-migrate]${NC} %s\n" "$*"; }
ok()    { printf "${GREEN}[cel-migrate]${NC} %s\n" "$*"; }
warn()  { printf "${YELLOW}[cel-migrate]${NC} %s\n" "$*"; }
error() { printf "${RED}[cel-migrate]${NC} %s\n" "$*" >&2; }
die()   { error "$@"; exit 1; }

stock_zig_path() {
    if command -v zig >/dev/null 2>&1; then
        command -v zig
    else
        return 1
    fi
}

stock_zig_version() {
    local path
    path="$(stock_zig_path)" || return 1
    "$path" version 2>/dev/null | tr -d '\r' | head -n 1
}

probe_stock_build_runner() {
    local output
    if output="$(cd "$REPO_ROOT" && zig build --help 2>&1 1>/dev/null)"; then
        printf 'ok'
        return 0
    fi

    if [[ "$output" == *"__availability_version_check"* || "$output" == *"undefined symbol:"* ]]; then
        printf 'darwin-linker'
    else
        printf 'failing'
    fi
}

report_stock_zig() {
    info "Step 2a: Inspecting stock Zig"

    if ! command -v zig >/dev/null 2>&1; then
        warn "  zig: not found on PATH"
        return
    fi

    local path version state
    path="$(stock_zig_path)"
    version="$(stock_zig_version)"
    ok "  zig: found ($path)"
    info "  zig version: $version"

    if [[ -n "$EXPECTED_ZIG_VERSION" && "$version" != "$EXPECTED_ZIG_VERSION" ]]; then
        warn "  zig pin mismatch: expected $EXPECTED_ZIG_VERSION"
    else
        ok "  zig pin matches .zigversion"
    fi

    state="$(probe_stock_build_runner)"
    case "$state" in
        ok)
            ok "  build runner: stock zig can start ABI build steps"
            ;;
        darwin-linker)
            warn "  build runner: blocked by Darwin linker failure"
            warn "  use CEL/bootstrap for repo validation on this host"
            ;;
        failing)
            warn "  build runner: stock zig failed before ABI gates could run"
            ;;
    esac
}

# Parse arguments
CHECK_ONLY=false
ACTIVATE_ONLY=false
CLEAN=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --check)    CHECK_ONLY=true; shift ;;
        --activate) ACTIVATE_ONLY=true; shift ;;
        --clean)    CLEAN=true; shift ;;
        -h|--help)
            cat <<'USAGE'
Usage: cel_migrate.sh [OPTIONS]

Options:
  --check      Check migration prerequisites only (no build)
  --activate   Activate an already-built CEL toolchain
  --clean      Clean rebuild of the CEL toolchain
  -h, --help   Show this help message

Migration Steps:
  1. Check platform: macOS 26+ requires CEL for binary output
  2. Verify prerequisites: git, cmake, cc, LLVM
  3. Build .cel toolchain: .cel/build.sh
  4. Activate: export PATH=".cel/bin:$PATH"
  5. Validate: zig build full-check
USAGE
            exit 0
            ;;
        *) die "Unknown option: $1" ;;
    esac
done

# ── Step 1: Platform check ──────────────────────────────────────────────
info "Step 1: Platform check"

OS="$(uname -s)"
if [[ "$OS" != "Darwin" ]]; then
    ok "Not on macOS — stock Zig should work fine."
    ok "CEL migration not needed."
    exit 0
fi

OS_VER="$(sw_vers -productVersion 2>/dev/null || echo 'unknown')"
MAJOR="$(echo "$OS_VER" | cut -d. -f1)"

info "macOS version: $OS_VER"

if [[ "$MAJOR" -ge 26 ]] 2>/dev/null; then
    warn "macOS 26+ detected — CEL toolchain REQUIRED for binary builds"
else
    info "macOS $MAJOR — stock Zig may work, CEL is optional"
fi

# ── Step 2: Prerequisites ───────────────────────────────────────────────
info "Step 2: Checking prerequisites"

MISSING=()
for cmd in git cmake cc c++; do
    if command -v "$cmd" >/dev/null 2>&1; then
        ok "  $cmd: found"
    else
        error "  $cmd: MISSING"
        MISSING+=("$cmd")
    fi
done

# Check LLVM
LLVM_FOUND=false
if [[ -d "$REPO_ROOT/zig-bootstrap-emergency/out/build-llvm-host" ]]; then
    ok "  LLVM: bootstrap artifacts"
    LLVM_FOUND=true
elif [[ -d "/opt/homebrew/opt/llvm@21" ]]; then
    ok "  LLVM: Homebrew llvm@21 (Apple Silicon)"
    LLVM_FOUND=true
elif [[ -d "/usr/local/opt/llvm@21" ]]; then
    ok "  LLVM: Homebrew llvm@21 (Intel)"
    LLVM_FOUND=true
elif command -v brew >/dev/null 2>&1; then
    BREW_LLVM21="$(brew --prefix llvm@21 2>/dev/null || true)"
    if [[ -n "$BREW_LLVM21" && -d "$BREW_LLVM21" ]]; then
        ok "  LLVM: Homebrew llvm@21 ($BREW_LLVM21)"
        LLVM_FOUND=true
    fi
fi

if ! $LLVM_FOUND && command -v llvm-config >/dev/null 2>&1; then
    LLVM_VER="$(llvm-config --version 2>/dev/null || echo 'unknown')"
    if [[ "$LLVM_VER" == 21.* ]]; then
        ok "  LLVM: system ($LLVM_VER)"
        LLVM_FOUND=true
    else
        warn "  LLVM: found system llvm-config $LLVM_VER, but CEL pin expects LLVM 21.x"
    fi
fi

if ! $LLVM_FOUND && [[ -d "/opt/homebrew/opt/llvm" ]]; then
    warn "  LLVM: Homebrew llvm found, but CEL pin expects llvm@21"
elif ! $LLVM_FOUND && [[ -d "/usr/local/opt/llvm" ]]; then
    warn "  LLVM: Homebrew llvm found, but CEL pin expects llvm@21"
fi

if [[ -x "$BOOTSTRAP_HOST_ZIG" ]]; then
    BOOTSTRAP_VER="$("$BOOTSTRAP_HOST_ZIG" version 2>/dev/null || echo 'unknown')"
    ok "  bootstrap zig: $BOOTSTRAP_VER ($BOOTSTRAP_HOST_ZIG)"
elif [[ -d "$REPO_ROOT/zig-bootstrap-emergency/zig" ]]; then
    info "  bootstrap zig: source present, host binary not built yet"
fi

report_stock_zig

if [[ ${#MISSING[@]} -gt 0 ]]; then
    die "Missing prerequisites: ${MISSING[*]}"
fi

if ! $LLVM_FOUND; then
    warn "No compatible LLVM found. .cel/build.sh will look for llvm@21 or bootstrap LLVM."
    warn "If the build fails, install LLVM 21: brew install llvm@21"
fi

if [[ "$MAJOR" -ge 26 ]] 2>/dev/null && [[ ! -x "$BOOTSTRAP_HOST_ZIG" ]]; then
    info "macOS 26+ note: .cel/build.sh now prefers a bootstrap-host Zig when available."
    info "If stage3 still cannot start, run 'abi toolchain bootstrap' to refresh zig-bootstrap-emergency."
fi

if $CHECK_ONLY; then
    echo ""
    ok "Prerequisites check passed."
    if [[ -x "$CEL_ZIG" ]]; then
        CEL_VER="$("$CEL_ZIG" version 2>/dev/null || echo 'unknown')"
        ok "CEL toolchain already built: $CEL_VER"
    else
        info "CEL toolchain not yet built. Run without --check to build."
        if [[ "$MAJOR" -ge 26 ]] 2>/dev/null && [[ -x "$BOOTSTRAP_HOST_ZIG" ]]; then
            info "Next action: ./.cel/build.sh"
        elif [[ "$MAJOR" -ge 26 ]] 2>/dev/null && [[ -d "$REPO_ROOT/zig-bootstrap-emergency/zig" ]]; then
            info "Next action: abi toolchain bootstrap"
        fi
    fi
    exit 0
fi

# ── Step 3: Activate only ──────────────────────────────────────────────
if $ACTIVATE_ONLY; then
    if [[ ! -x "$CEL_ZIG" ]]; then
        die "CEL toolchain not built. Run without --activate first."
    fi
    export PATH="$CEL_DIR/bin:$PATH"
    CEL_VER="$(zig version 2>/dev/null)"
    ok "Activated CEL toolchain: $CEL_VER"
    ok "PATH updated: $CEL_DIR/bin is first"
    
    # Print eval-friendly output for sourcing
    echo "export PATH=\"$CEL_DIR/bin:\$PATH\""
    exit 0
fi

# ── Step 4: Build ──────────────────────────────────────────────────────
info "Step 4: Building CEL toolchain"

if [[ ! -f "$CEL_DIR/build.sh" ]]; then
    die ".cel/build.sh not found. Ensure .cel/ directory is in the repo checkout."
fi

if $CLEAN; then
    info "Clean rebuild requested"
    "$CEL_DIR/build.sh" --clean
else
    if [[ -x "$CEL_ZIG" ]]; then
        CEL_VER="$("$CEL_ZIG" version 2>/dev/null || echo 'unknown')"
        EXPECTED=""
        if [[ -f "$REPO_ROOT/.zigversion" ]]; then
            EXPECTED="$(cat "$REPO_ROOT/.zigversion" | tr -d '[:space:]')"
        fi
        
        if [[ "$CEL_VER" == "$EXPECTED" ]]; then
            ok "CEL toolchain already built and up-to-date: $CEL_VER"
        else
            warn "CEL version ($CEL_VER) doesn't match .zigversion ($EXPECTED)"
            info "Rebuilding..."
            "$CEL_DIR/build.sh" --clean
        fi
    else
        "$CEL_DIR/build.sh"
    fi
fi

# ── Step 5: Verify ─────────────────────────────────────────────────────
info "Step 5: Verifying CEL build"

if [[ ! -x "$CEL_ZIG" ]]; then
    die "Build appeared to succeed but .cel/bin/zig not found"
fi

if [[ ! -x "$CEL_ZLS" ]]; then
    warn "Build appeared to succeed but .cel/bin/zls not found (ZLS may not be built)"
fi

CEL_VER="$("$CEL_ZIG" version 2>/dev/null || echo 'unknown')"
ok "CEL Zig version: $CEL_VER"

if [[ -x "$CEL_ZLS" ]]; then
    CEL_ZLS_VER="$("$CEL_ZLS" --version 2>/dev/null || echo 'unknown')"
    ok "CEL ZLS version: $CEL_ZLS_VER"
fi

# Version consistency check
if [[ -f "$REPO_ROOT/.zigversion" ]]; then
    EXPECTED="$(cat "$REPO_ROOT/.zigversion" | tr -d '[:space:]')"
    if [[ "$CEL_VER" == "$EXPECTED" ]]; then
        ok "Version matches .zigversion"
    else
        warn "Version mismatch: cel=$CEL_VER, .zigversion=$EXPECTED"
    fi
fi

# ── Step 6: Activate ───────────────────────────────────────────────────
info "Step 6: Activating CEL toolchain"
export PATH="$CEL_DIR/bin:$PATH"
ok "PATH updated: $CEL_DIR/bin is first"

# ── Step 7: Quick validation ────────────────────────────────────────────
info "Step 7: Quick validation"

# Check zig can parse build.zig
if zig build --help >/dev/null 2>&1; then
    ok "zig build: OK (build system parses)"
else
    warn "zig build --help failed — the build runner may still have linker issues"
fi

echo ""
ok "═══════════════════════════════════════"
ok "  CEL migration complete!"
ok ""
ok "  To activate in your current shell:"
ok "    eval \"\$(./tools/scripts/use_cel.sh)\""
ok ""
ok "  To run the full gate:"
ok "    zig build full-check"
ok "═══════════════════════════════════════"
echo ""
