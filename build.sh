#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

usage() {
    cat <<EOF
Usage: ./build.sh [command] [options]

Commands:
    (default)         Build the library
    test              Run all tests
    cli               Build CLI binary
    lib               Build static library
    mcp               Build MCP server
    lint              Check formatting
    fix               Auto-format
    check             Full gate (lint + test + parity)
    check-parity      Verify mod/stub declaration parity
    --link            Install and link Zig + ZLS to ~/.local/bin
    --status          Show Zig toolchain status
    --bootstrap       Full setup: install Zig + link + build
    --help            Show this help

Options:
    -Dfeat-*          Feature flags (e.g., -Dfeat-gpu=false)
    -Dgpu-backend=*  GPU backend (metal, cuda, vulkan, etc.)

Examples:
    ./build.sh                    # Build library
    ./build.sh cli                # Build CLI
    ./build.sh test --summary all # Run tests
    ./build.sh --link             # Link Zig to PATH
    ./build.sh --bootstrap        # Full setup
EOF
    exit "${1:-0}"
}

LINK_MODE=false
BOOTSTRAP_MODE=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --link)
            LINK_MODE=true
            shift
            ;;
        --bootstrap)
            BOOTSTRAP_MODE=true
            shift
            ;;
        --help|-h)
            usage 0
            ;;
        --status)
            exec tools/zigly status
            ;;
        test|cli|lib|mcp|lint|fix|check|check-parity)
            COMMAND="$1"
            shift
            ;;
        -D*|--*)
            break
            ;;
        *)
            echo "Unknown option: $1"
            usage 1
            ;;
    esac
done

REMAINING_ARGS=("$@")

resolve_zig() {
    local candidate
    for candidate in \
        "$HOME/.zvm/bin/zig" \
        "$HOME/.local/bin/zig" \
        "$HOME/.zigly/versions/"*/bin/zig \
        "$(command -v zig 2>/dev/null || true)"; do
        if [ -n "$candidate" ] && [ -x "$candidate" ]; then
            readlink -f "$candidate" 2>/dev/null || echo "$candidate"
            return 0
        fi
    done
    return 1
}

ZIG="$(resolve_zig || true)"
if [ -z "$ZIG" ]; then
    echo "Error: Zig not found. Run './build.sh --bootstrap' or 'tools/zigly bootstrap' first." >&2
    exit 1
fi

echo "Using Zig: $ZIG"
echo ""

if [ "$LINK_MODE" = true ]; then
    echo "Linking Zig + ZLS to ~/.local/bin..."
    tools/zigly use || tools/zigly install
    exit $?
fi

if [ "$BOOTSTRAP_MODE" = true ]; then
    echo "Bootstrapping: installing and linking Zig + ZLS..."
    tools/zigly bootstrap
    echo ""
    echo "Building project..."
fi

COMMAND="${COMMAND:-}"
if [ -z "$COMMAND" ]; then
    echo "Building library..."
    exec "$ZIG" build "${REMAINING_ARGS[@]}"
else
    exec "$ZIG" build "$COMMAND" "${REMAINING_ARGS[@]}"
fi