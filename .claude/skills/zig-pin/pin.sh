#!/usr/bin/env bash
# zig-pin driver: check that the active Zig toolchain matches the .zigversion
# pin and print the exact command to fix a mismatch.
#
# This is a read/diagnose script — it never edits .zigversion and never
# auto-runs the select command. Toolchain switching is the user's call.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../../.." && pwd)
cd "$REPO_ROOT"

PIN=$(tr -d ' \t\r\n' < .zigversion 2>/dev/null || echo "")
if ! command -v zig >/dev/null 2>&1; then
    echo "FATAL: zig not found on PATH."
    echo "Install a version manager (zvm or zigup), then:"
    echo "  zvm install $PIN && zvm use $PIN"
    exit 1
fi
ACTIVE=$(zig version)

echo "repo pin (.zigversion): ${PIN:-<none>}"
echo "active zig:             $ACTIVE"

if [ "$ACTIVE" = "$PIN" ]; then
    echo "RESULT: MATCH — active Zig matches the repo pin ($PIN)."
    exit 0
fi

echo ""
echo "MISMATCH: active ($ACTIVE) ≠ pinned ($PIN)"
echo ""

if command -v zvm >/dev/null 2>&1; then
    echo "Fix with zvm:"
    echo "  zvm install $PIN    # only needed if not cached"
    echo "  zvm use $PIN"
elif command -v zigup >/dev/null 2>&1; then
    echo "Fix with zigup:"
    echo "  zigup $PIN"
else
    echo "No version manager found (zvm or zigup)."
    echo "Install one, or manually fetch and place the pinned build:"
    echo "  $PIN"
fi
exit 2
