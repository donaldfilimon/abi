#!/bin/bash
# Load ABI project context at session start
set -euo pipefail

PROJECT_DIR="${CLAUDE_PROJECT_DIR:-.}"

# Export platform detection
if [[ "$(uname -s)" == "Darwin" ]]; then
    MACOS_VER="$(sw_vers -productVersion 2>/dev/null || echo unknown)"
    MACOS_MAJOR="${MACOS_VER%%.*}"
    if [[ "$MACOS_MAJOR" -ge 26 ]] 2>/dev/null; then
        echo "ABI: macOS $MACOS_VER detected (blocked Darwin). Use ./tools/scripts/run_build.sh or /zig-abi:build for builds."
    fi
fi

# Check for lessons learned
if [[ -f "$PROJECT_DIR/tasks/lessons.md" ]]; then
    echo "ABI: tasks/lessons.md exists — review corrections before starting work."
fi

# Check for active tasks
if [[ -f "$PROJECT_DIR/tasks/todo.md" ]]; then
    echo "ABI: tasks/todo.md exists — check for in-progress work."
fi

# Check zig availability
if command -v zig &>/dev/null; then
    ZIG_VER="$(zig version 2>/dev/null || echo unknown)"
    echo "ABI: Zig $ZIG_VER available."
else
    echo "ABI: WARNING — zig not found on PATH."
fi
