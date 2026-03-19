#!/usr/bin/env bash
# build.sh — Top-level build entry point for the ABI Framework.
#
# Resolves the correct Zig compiler and delegates to the appropriate
# build mechanism:
#   - On Darwin 26+: delegates to run_build.sh (Apple ld relink)
#   - On version mismatch: prints clear instructions
#   - Otherwise: runs zig build directly
#
# Usage: ./build.sh [zig build args...]
#   e.g. ./build.sh test --summary all
#        ./build.sh full-check --summary all
#        ./build.sh lint

set -Eeuo pipefail
unset CDPATH

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
source "$SCRIPT_DIR/tools/scripts/zig_toolchain.sh"

log() { echo "[build] $*" >&2; }

# ── Resolve Zig ──────────────────────────────────────────────────────────

ZIG="$(abi_toolchain_resolve_active_zig)" || {
    log "No usable Zig found. Run ./tools/scripts/bootstrap_host_zig.sh"
    exit 1
}
EXPECTED_VER="$(abi_toolchain_expected_version)"
ACTUAL_VER="$("$ZIG" version 2>/dev/null || echo unknown)"

# ── Version check ────────────────────────────────────────────────────────

if [[ "$ACTUAL_VER" != "$EXPECTED_VER" ]]; then
    # Extract dev build number (e.g. "2905" from "0.16.0-dev.2905+...")
    dev_num="${ACTUAL_VER#*dev.}"
    dev_num="${dev_num%%+*}"
    if [[ "$dev_num" =~ ^[0-9]+$ ]] && (( dev_num < 2000 )); then
        log "ERROR: Zig version too old for this build system."
        log "  Required: $EXPECTED_VER"
        log "  Found:    $ACTUAL_VER (dev build $dev_num < 2000)"
        log ""
        log "  Options:"
        log "    1. Download the pinned version from ziglang.org/builds"
        log "    2. Run: ./tools/scripts/bootstrap_host_zig.sh"
        exit 1
    else
        log "WARNING: Zig version mismatch — expected $EXPECTED_VER, got $ACTUAL_VER"
    fi
fi

# ── Darwin 26+ detection ─────────────────────────────────────────────────

if [[ "$(uname -s)" == "Darwin" ]]; then
    macos_major="$(sw_vers -productVersion 2>/dev/null | cut -d. -f1 || echo 0)"
    if (( macos_major >= 26 )); then
        log "Darwin 26+ detected — delegating to run_build.sh"
        exec "$SCRIPT_DIR/tools/scripts/run_build.sh" "$@"
    fi
fi

# ── Normal build ─────────────────────────────────────────────────────────

exec "$ZIG" build "$@"
