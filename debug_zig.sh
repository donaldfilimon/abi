#!/bin/bash
HOME="/Users/donaldfilimon"
for candidate in     "$HOME/.zvm/bin/zig"     "$HOME/.local/bin/zig"     "$HOME/.zigly/versions/"*/bin/zig     "$(command -v zig 2>/dev/null || true)"; do
    echo "Checking candidate: $candidate"
    if [ -n "$candidate" ] && [ -x "$candidate" ]; then
        echo "FOUND: $candidate"
        exit 0
    fi
done
