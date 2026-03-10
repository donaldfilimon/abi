#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BOOTSTRAP_BIN="$REPO_ROOT/.zig-bootstrap/bin"

if [[ ! -x "$BOOTSTRAP_BIN/zig" ]]; then
    echo "Error: bootstrap Zig toolchain not found at $BOOTSTRAP_BIN/zig" >&2
    echo "Build it first:" >&2
    echo "  cd $REPO_ROOT && ./.zig-bootstrap/build.sh" >&2
    exit 1
fi

export PATH="$BOOTSTRAP_BIN:$PATH"
printf 'export PATH="%s:$PATH"\n' "$BOOTSTRAP_BIN"
