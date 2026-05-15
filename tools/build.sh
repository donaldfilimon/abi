#!/usr/bin/env bash
set -euo pipefail

# ABI Build Wrapper for macOS 26.4+ (Darwin 25.x)
# Always use this on macOS to ensure proper linking.

ZIG_BIN=$(command -v zig)
if [ -f ".zigversion" ]; then
    # In a real environment, we'd use zigly or zvm to switch,
    # but here we just assume the system zig is correct or report it.
    echo "Using Zig: $ZIG_BIN"
fi

# Detect macOS version
OS_VER=$(sw_vers -productVersion 2>/dev/null || printf '0.0.0')
MAJOR_VER=${OS_VER%%.*}

EXTRA_FLAGS=""
if [ "$MAJOR_VER" -ge 26 ] || [[ "$OSTYPE" == "darwin"* ]]; then
    echo "macOS detected: adding stability flags."
    EXTRA_FLAGS="-Dfeat-gpu=false"
fi

case "$1" in
    "check")
        "$ZIG_BIN" build check --summary all $EXTRA_FLAGS "${@:2}"
        ;;
    "test")
        "$ZIG_BIN" build test --summary all $EXTRA_FLAGS "${@:2}"
        ;;
    "cli")
        "$ZIG_BIN" build cli $EXTRA_FLAGS "${@:2}"
        ;;
    "mcp")
        "$ZIG_BIN" build mcp $EXTRA_FLAGS "${@:2}"
        ;;
    "full-check")
        "$ZIG_BIN" build full-check --summary all $EXTRA_FLAGS "${@:2}"
        ;;
    *)
        "$ZIG_BIN" build "$@" $EXTRA_FLAGS
        ;;
esac
