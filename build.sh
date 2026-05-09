#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

usage() {
    cat <<EOF
Usage: ./build.sh [command] [options]

Commands:
    (default)         Build the library
    dev               Build fast developer targets (typecheck + cli + mcp)
    quick             Run fast validation (fmt + typecheck + parity)
    ci                Run CI validation (lint + tests + parity + MCP + cross-check)
    test              Run all tests
    cli               Build CLI binary
    lib               Build static library
    mcp               Build MCP stdio/SSE server
    tools             Install abi + abi-mcp to ~/.local/bin
    install           Alias for tools
    lint              Check formatting
    fix               Auto-format
    check             Full gate (lint + test + parity)
    check-parity      Verify mod/stub declaration parity
    doctor            Build/toolchain diagnostics
    mcp-health        Check configured MCP HA health endpoints
    interop           Check MCP-ACP readiness
    acp-endpoints     List/check ACP endpoints from ACP_ENDPOINTS
    --link            Install/link Zig + ZLS, then install abi tools to ~/.local/bin
    --status          Show Zig toolchain status
    --bootstrap       Full setup: bootstrap Zig/ZLS, install abi tools, build
    --help            Show this help

Options:
    -Dfeat-*          Feature flags (e.g., -Dfeat-gpu=false)
    -Dgpu-backend=*   GPU backend (metal, cuda, vulkan, etc.)

Examples:
    ./build.sh                    # Build library
    ./build.sh quick              # Fast local gate
    ./build.sh dev                # Build CLI + MCP after typecheck
    ./build.sh cli                # Build CLI
    ./build.sh test --summary all # Run tests
    ./build.sh install            # Install abi + abi-mcp to ~/.local/bin
    ABI_VERBOSE=1 ./build.sh ci   # Print delegated zig command
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
            exec tools/zigly --status
            ;;
        -D*|--*)
            break
            ;;
        *)
            if [ -z "$COMMAND" ]; then
                COMMAND="$1"
                shift
            else
                break
            fi
            ;;
    esac
done

resolve_zig() {
    if [ ! -x "tools/zigly" ]; then
        echo "Error: tools/zigly is missing or not executable." >&2
        echo "Hint: run from the repository root, or restore tools/zigly before building." >&2
        return 1
    fi

    local candidate
    candidate="$(
        tools/zigly --status 2>/dev/null |
            while IFS= read -r line; do
                line="${line%$'\r'}"
                if [ -n "$line" ] && [ -x "$line" ]; then
                    printf '%s\n' "$line"
                fi
            done |
            tail -n 1
    )"
    if [ -n "$candidate" ] && [ -x "$candidate" ]; then
        printf '%s\n' "$candidate"
        return 0
    fi
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
    if ! "$zig_bin" build tools --prefix "$HOME/.local"; then
        echo "Error: failed to install abi tools with '$zig_bin build tools --prefix $HOME/.local'." >&2
        echo "Hint: run './build.sh dev' to check compile errors before installing." >&2
        exit 1
    fi
    echo ""
    echo 'Add to PATH if needed: export PATH="$HOME/.local/bin:$PATH"'
}

ZIG="$(resolve_zig || true)"

if [ "$LINK_MODE" = true ]; then
    echo "Linking Zig + ZLS to ~/.local/bin..."
    if ! tools/zigly --link; then
        echo "Error: tools/zigly --link failed." >&2
        exit 1
    fi
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
    if ! tools/zigly --bootstrap; then
        echo "Error: tools/zigly --bootstrap failed." >&2
        echo "Hint: run 'tools/zigly --status' for toolchain diagnostics." >&2
        exit 1
    fi
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
    echo "Hint: './build.sh --status' shows the currently resolved Zig path when available." >&2
    exit 1
fi

echo "Using Zig: $ZIG"
echo ""

case "$COMMAND" in
    install)
        COMMAND="tools"
        ;;
    doctor)
        COMMAND="doctor-full"
        ;;
    ""|dev|quick|ci|test|cli|lib|mcp|tools|lint|fix|check|check-parity|doctor-full|mcp-health|interop|acp-endpoints|typecheck|cross-check|mcp-tests|feature-tests|full-check|verify-all|dashboard-smoke|validate-flags|*-tests)
        ;;
    *)
        echo "Error: unknown build command '$COMMAND'." >&2
        echo "Run './build.sh --help' for supported aliases and Zig build steps." >&2
        exit 1
        ;;
esac

if [[ ("$COMMAND" == "test" || "$COMMAND" == "check" || "$COMMAND" == "ci") ]] && is_macos_26_4_or_newer && ! has_feat_gpu_flag "$@"; then
    echo "macOS 26.4+ detected: adding -Dfeat-gpu=false for stability."
    echo ""
    set -- -Dfeat-gpu=false "$@"
fi

if [ -z "$COMMAND" ]; then
    echo "Building library..."
    if [ "${ABI_VERBOSE:-0}" = "1" ]; then
        printf 'Running: %q build' "$ZIG"
        printf ' %q' "$@"
        printf '\n'
    fi
    exec "$ZIG" build "$@"
else
    if [ "${ABI_VERBOSE:-0}" = "1" ]; then
        printf 'Running: %q build %q' "$ZIG" "$COMMAND"
        printf ' %q' "$@"
        printf '\n'
    fi
    exec "$ZIG" build "$COMMAND" "$@"
fi
