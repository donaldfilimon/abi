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
    tools             Install abi + abi-mcp to ~/.local/bin
    lint              Check formatting
    fix               Auto-format
    check             Full gate (lint + test + parity)
    check-parity      Verify mod/stub declaration parity
    --link            Install/link Zig + ZLS, then install abi tools to ~/.local/bin
    --status          Show Zig toolchain status
    --bootstrap       Full setup: bootstrap Zig/ZLS, install abi tools, build
    --help            Show this help

Options:
    -Dfeat-*          Feature flags (e.g., -Dfeat-gpu=false)
    -Dgpu-backend=*   GPU backend (metal, cuda, vulkan, etc.)

Examples:
    ./build.sh                    # Build library
    ./build.sh cli                # Build CLI
    ./build.sh test --summary all # Run tests
    ./build.sh tools              # Install abi + abi-mcp to ~/.local/bin
    ./build.sh --link             # Link Zig/ZLS and install abi tools
    ./build.sh --bootstrap        # Full setup
EOF
    exit "${1:-0}"
}

LINK_MODE=false
BOOTSTRAP_MODE=false
COMMAND=""

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
        test|cli|lib|mcp|tools|lint|fix|check|check-parity)
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

is_macos_26_4_or_newer() {
    if [ "$(uname -s)" != "Darwin" ]; then
        return 1
    fi

    local darwin_major
    darwin_major="$(uname -r | cut -d. -f1)"
    [ "${darwin_major:-0}" -ge 25 ]
}

has_feat_gpu_flag() {
    local arg
    for arg in "$@"; do
        case "$arg" in
            -Dfeat-gpu=*)
                return 0
                ;;
        esac
    done
    return 1
}

install_local_tools() {
    local zig_bin="$1"

    mkdir -p "$HOME/.local/bin"
    echo "Installing abi + abi-mcp to $HOME/.local/bin..."
    "$zig_bin" build tools --prefix "$HOME/.local"
    echo ""
    echo 'Add to PATH if needed: export PATH="$HOME/.local/bin:$PATH"'
}

ZIG="$(resolve_zig || true)"

if [ "$LINK_MODE" = true ]; then
    echo "Linking Zig + ZLS to ~/.local/bin..."
    tools/zigly --link
    ZIG="$(resolve_zig || true)"
    if [ -z "$ZIG" ]; then
        echo "Error: Zig not found after tools/zigly --link." >&2
        exit 1
    fi
    echo "Using Zig: $ZIG"
    echo ""
    install_local_tools "$ZIG"
    exit 0
fi

if [ "$BOOTSTRAP_MODE" = true ]; then
    echo "Bootstrapping: installing and linking Zig + ZLS..."
    tools/zigly --bootstrap
    ZIG="$(resolve_zig || true)"
    if [ -z "$ZIG" ]; then
        echo "Error: Zig not found after tools/zigly --bootstrap." >&2
        exit 1
    fi
    echo "Using Zig: $ZIG"
    echo ""
    install_local_tools "$ZIG"
    echo "Building project..."
fi

if [ -z "$ZIG" ]; then
    echo "Error: Zig not found. Run './build.sh --bootstrap' or 'tools/zigly --bootstrap' first." >&2
    exit 1
fi

echo "Using Zig: $ZIG"
echo ""

if [ "$COMMAND" = "test" ] && is_macos_26_4_or_newer && ! has_feat_gpu_flag "$@"; then
    echo "macOS 26.4+ detected: adding -Dfeat-gpu=false for test stability."
    echo ""
    set -- -Dfeat-gpu=false "$@"
fi

if [ -z "$COMMAND" ]; then
    echo "Building library..."
    exec "$ZIG" build "$@"
else
    exec "$ZIG" build "$COMMAND" "$@"
fi