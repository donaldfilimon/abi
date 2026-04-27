#!/bin/bash
# Simple helper to send JSON-RPC to abi-mcp
abi_call() {
    local method=$1
    local params=$2
    printf '{"jsonrpc": "2.0", "method": "%s", "params": %s, "id": 1}\n' "$method" "$params" | \
    # In a real setup, pipe to the running server or abi-mcp process
    # For now, print the payload to verify
    cat
}
