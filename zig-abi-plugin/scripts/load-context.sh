#!/bin/bash
# Load ABI project context at session start
set -euo pipefail

PROJECT_DIR="${CLAUDE_PROJECT_DIR:-.}"

# ── Platform detection ───────────────────────────────────────────────────
if [[ "$(uname -s)" == "Darwin" ]]; then
    MACOS_VER="$(sw_vers -productVersion 2>/dev/null || echo unknown)"
    MACOS_MAJOR="${MACOS_VER%%.*}"
    if [[ "$MACOS_MAJOR" -ge 25 ]] 2>/dev/null; then
        echo "ABI: macOS $MACOS_VER detected (Darwin linker blocked). Use ./tools/scripts/run_build.sh or /zig-abi:build for builds."
    fi
fi

# ── Zig version and pin check ────────────────────────────────────────────
PINNED_VER=""
if [[ -f "$PROJECT_DIR/.zigversion" ]]; then
    PINNED_VER="$(cat "$PROJECT_DIR/.zigversion" | tr -d '[:space:]')"
    echo "ABI: Pinned Zig version: $PINNED_VER"
fi

if command -v zig &>/dev/null; then
    ZIG_VER="$(zig version 2>/dev/null || echo unknown)"
    echo "ABI: PATH Zig version: $ZIG_VER"
    if [[ -n "$PINNED_VER" && "$ZIG_VER" != "$PINNED_VER" ]]; then
        echo "ABI: WARNING — PATH Zig ($ZIG_VER) does not match pinned version ($PINNED_VER)."
        echo "ABI:   Run ./tools/scripts/bootstrap_host_zig.sh and prepend the cache to PATH."
    fi
else
    echo "ABI: WARNING — zig not found on PATH."
fi

# ── Host-built Zig cache check ───────────────────────────────────────────
HOST_CACHE="$HOME/.cache/abi-host-zig"
if [[ -d "$HOST_CACHE" ]]; then
    if [[ -n "$PINNED_VER" && -d "$HOST_CACHE/$PINNED_VER/bin" ]]; then
        echo "ABI: Host-built Zig cache found at $HOST_CACHE/$PINNED_VER/"
    elif [[ -n "$PINNED_VER" ]]; then
        echo "ABI: Host-built cache exists but no $PINNED_VER build. Run bootstrap_host_zig.sh to rebuild."
    fi
else
    echo "ABI: No host-built Zig cache at $HOST_CACHE — run bootstrap_host_zig.sh if linker fails."
fi

# ── Task and lessons check ───────────────────────────────────────────────
if [[ -f "$PROJECT_DIR/tasks/lessons.md" ]]; then
    LESSONS_AGE=""
    if command -v git &>/dev/null; then
        LESSONS_AGE="$(git -C "$PROJECT_DIR" log -1 --format='%cr' -- tasks/lessons.md 2>/dev/null || echo "")"
    fi
    if [[ -n "$LESSONS_AGE" ]]; then
        echo "ABI: tasks/lessons.md exists (last updated $LESSONS_AGE) — review corrections before starting work."
    else
        echo "ABI: tasks/lessons.md exists — review corrections before starting work."
    fi
fi

if [[ -f "$PROJECT_DIR/tasks/todo.md" ]]; then
    echo "ABI: tasks/todo.md exists — check for in-progress work."
fi
