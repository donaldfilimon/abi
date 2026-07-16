#!/usr/bin/env bash
set -euo pipefail

# ABI build wrapper. This keeps the documented Darwin entrypoint stable and
# normalizes common build commands around the pinned Zig workflow.

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
    echo "macOS detected: build.zig handles Metal linking when feat-gpu=true."
fi

if [ "$#" -eq 0 ]; then
    "$ZIG_BIN" build $EXTRA_FLAGS
    exit 0
fi

case "$1" in
    "-l"|"--list"|"list")
        "$ZIG_BIN" build -l
        ;;
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
    "test-cli")
        "$ZIG_BIN" build test-cli $EXTRA_FLAGS "${@:2}"
        ;;
    "test-plugins")
        "$ZIG_BIN" build test-plugins $EXTRA_FLAGS "${@:2}"
        ;;
    "test-feature-contracts")
        "$ZIG_BIN" build test-feature-contracts $EXTRA_FLAGS "${@:2}"
        ;;
    "test-integration")
        "$ZIG_BIN" build test-integration $EXTRA_FLAGS "${@:2}"
        ;;
    "benchmarks")
        "$ZIG_BIN" build benchmarks $EXTRA_FLAGS "${@:2}"
        ;;
    "test-mcp-contracts")
        "$ZIG_BIN" build test-mcp-contracts $EXTRA_FLAGS "${@:2}"
        ;;
    "test-mcp-server")
        "$ZIG_BIN" build test-mcp-server $EXTRA_FLAGS "${@:2}"
        ;;
    "test-contracts")
        "$ZIG_BIN" build test-contracts $EXTRA_FLAGS "${@:2}"
        ;;
    "lint")
        "$ZIG_BIN" build lint $EXTRA_FLAGS "${@:2}"
        ;;
    "fix")
        "$ZIG_BIN" build fix $EXTRA_FLAGS "${@:2}"
        ;;
    "check-parity")
        "$ZIG_BIN" build check-parity $EXTRA_FLAGS "${@:2}"
        ;;
    "cross-smoke")
        "$ZIG_BIN" build cross-smoke $EXTRA_FLAGS "${@:2}"
        ;;
    *)
        "$ZIG_BIN" build "$@" $EXTRA_FLAGS
        ;;
esac
