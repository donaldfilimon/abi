#!/bin/bash
set -euo pipefail

# patch-sdk.sh — Patch macOS SDK TBD files for Zig arm64 compatibility
#
# macOS 26+ SDK TBD files list "arm64e-macos" but NOT "arm64-macos".
# Zig's self-hosted Mach-O linker targets plain arm64 and cannot match arm64e,
# causing all system symbol resolution to fail.
#
# This script creates an overlay directory that mirrors the SDK structure with
# patched TBD files that include both arm64e-macos and arm64-macos targets.
#
# Usage:
#   tools/patch-sdk.sh [--force]
#   Prints the overlay path on success.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJ_DIR="$(dirname "$SCRIPT_DIR")"
PATCH_DIR="$PROJ_DIR/.darwin-sdk-overlay"
FORCE=false

for arg in "$@"; do
    [ "$arg" = "--force" ] && FORCE=true
done

# Only needed on macOS with Darwin 25+
DARWIN_MAJOR="$(uname -r | cut -d. -f1)"
if [ "$(uname -s)" != "Darwin" ] || [ "$DARWIN_MAJOR" -lt 25 ] 2>/dev/null; then
    exit 0
fi

SYSROOT="$(xcrun --show-sdk-path 2>/dev/null || echo "")"
if [ -z "$SYSROOT" ] || [ ! -d "$SYSROOT" ]; then
    echo "ERROR: Cannot find macOS SDK" >&2
    exit 1
fi

# Check if overlay already exists and is up-to-date
STAMP="$PATCH_DIR/.stamp"
if [ "$FORCE" != true ] && [ -f "$STAMP" ]; then
    SAVED_SDK="$(cat "$STAMP")"
    if [ "$SAVED_SDK" = "$SYSROOT" ]; then
        echo "$PATCH_DIR"
        exit 0
    fi
fi

# Clean and recreate
rm -rf "$PATCH_DIR"
mkdir -p "$PATCH_DIR"

# --- Helper: patch a single TBD file ---
patch_tbd() {
    local src="$1"
    local dst="$2"
    sed 's/arm64e-macos/arm64e-macos, arm64-macos/g; s/arm64e-maccatalyst/arm64e-maccatalyst, arm64-maccatalyst/g' \
        "$src" > "$dst"
}

# --- Build the overlay directory structure ---

# Top-level: symlink everything
for item in "$SYSROOT"/*; do
    ln -sf "$item" "$PATCH_DIR/$(basename "$item")"
done

# usr/ — real directory
rm -f "$PATCH_DIR/usr"
mkdir -p "$PATCH_DIR/usr"
for item in "$SYSROOT/usr/"*; do
    ln -sf "$item" "$PATCH_DIR/usr/$(basename "$item")"
done

# usr/lib/ — real directory with patched TBDs
rm -f "$PATCH_DIR/usr/lib"
mkdir -p "$PATCH_DIR/usr/lib"
for item in "$SYSROOT/usr/lib/"*; do
    bn="$(basename "$item")"
    case "$bn" in
        system)
            ;; # handle below
        *.tbd)
            if grep -q "arm64e-macos" "$item" 2>/dev/null; then
                patch_tbd "$item" "$PATCH_DIR/usr/lib/$bn"
            else
                ln -sf "$item" "$PATCH_DIR/usr/lib/$bn"
            fi
            ;;
        *)
            ln -sf "$item" "$PATCH_DIR/usr/lib/$bn"
            ;;
    esac
done

# usr/lib/system/ — real directory with patched TBDs
mkdir -p "$PATCH_DIR/usr/lib/system"
for item in "$SYSROOT/usr/lib/system/"*; do
    bn="$(basename "$item")"
    case "$bn" in
        *.tbd)
            if grep -q "arm64e-macos" "$item" 2>/dev/null; then
                patch_tbd "$item" "$PATCH_DIR/usr/lib/system/$bn"
            else
                ln -sf "$item" "$PATCH_DIR/usr/lib/system/$bn"
            fi
            ;;
        *)
            ln -sf "$item" "$PATCH_DIR/usr/lib/system/$bn"
            ;;
    esac
done

# System/Library/Frameworks/ — patch needed frameworks
rm -f "$PATCH_DIR/System"
mkdir -p "$PATCH_DIR/System"
for item in "$SYSROOT/System/"*; do
    ln -sf "$item" "$PATCH_DIR/System/$(basename "$item")"
done

rm -f "$PATCH_DIR/System/Library"
mkdir -p "$PATCH_DIR/System/Library"
for item in "$SYSROOT/System/Library/"*; do
    ln -sf "$item" "$PATCH_DIR/System/Library/$(basename "$item")"
done

rm -f "$PATCH_DIR/System/Library/Frameworks"
mkdir -p "$PATCH_DIR/System/Library/Frameworks"

# Symlink all frameworks first
for item in "$SYSROOT/System/Library/Frameworks/"*; do
    ln -sf "$item" "$PATCH_DIR/System/Library/Frameworks/$(basename "$item")"
done

# Patch specific frameworks used by the project
patch_framework() {
    local fw_name="$1"
    local fw_src="$SYSROOT/System/Library/Frameworks/${fw_name}.framework"
    local fw_dst="$PATCH_DIR/System/Library/Frameworks/${fw_name}.framework"

    [ ! -d "$fw_src" ] && return

    # Remove symlink, create real dir
    rm -f "$fw_dst"
    mkdir -p "$fw_dst"

    # Symlink all contents
    for item in "$fw_src/"*; do
        ln -sf "$item" "$fw_dst/$(basename "$item")"
    done

    # Patch top-level TBD files
    for tbd in "$fw_src/"*.tbd; do
        [ ! -f "$tbd" ] && continue
        bn="$(basename "$tbd")"
        if grep -q "arm64e-macos" "$tbd" 2>/dev/null; then
            rm -f "$fw_dst/$bn"
            patch_tbd "$tbd" "$fw_dst/$bn"
        fi
    done

    # Handle Versions/A/ if it exists
    if [ -d "$fw_src/Versions/A" ]; then
        rm -f "$fw_dst/Versions"
        mkdir -p "$fw_dst/Versions/A"
        for item in "$fw_src/Versions/"*; do
            bn="$(basename "$item")"
            [ "$bn" = "A" ] && continue
            ln -sf "$item" "$fw_dst/Versions/$bn"
        done
        for item in "$fw_src/Versions/A/"*; do
            bn="$(basename "$item")"
            case "$bn" in
                *.tbd)
                    if grep -q "arm64e-macos" "$item" 2>/dev/null; then
                        patch_tbd "$item" "$fw_dst/Versions/A/$bn"
                    else
                        ln -sf "$item" "$fw_dst/Versions/A/$bn"
                    fi
                    ;;
                *)
                    ln -sf "$item" "$fw_dst/Versions/A/$bn"
                    ;;
            esac
        done
    fi
}

# Frameworks used by the ABI project
for fw in IOKit CoreFoundation CoreGraphics Accelerate Metal MetalPerformanceShaders Security; do
    patch_framework "$fw"
done

# Record which SDK we patched
echo "$SYSROOT" > "$STAMP"

echo "$PATCH_DIR"
