#!/usr/bin/env bash
# One-shot quarantine of out-of-scope plugin pollution. No loops, no background PIDs.
set -euo pipefail

if [ -n "${ROOT:-}" ] && [ -d "$ROOT" ]; then
  :
else
  ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
fi
cd "$ROOT"
SCRATCH="${SCRATCH:-}"
if [ -z "$SCRATCH" ]; then
  echo "quarantine-out-of-scope: set SCRATCH" >&2
  exit 2
fi
QDIR="$SCRATCH/out-of-scope-quarantine"
mkdir -p "$QDIR"

# Move (prefer) or remove zig-self-improve
if [ -e src/plugins/zig-self-improve ]; then
  rm -rf "$QDIR/zig-self-improve"
  mv src/plugins/zig-self-improve "$QDIR/zig-self-improve" 2>/dev/null \
    || { cp -a src/plugins/zig-self-improve "$QDIR/zig-self-improve"; rm -rf src/plugins/zig-self-improve; }
  echo "quarantined zig-self-improve -> $QDIR/zig-self-improve"
fi
rm -f src/plugins/zig-self-improve.QUARANTINED 2>/dev/null || true

# Move any other untracked residual under src/plugins/
while IFS= read -r p; do
  [ -z "$p" ] && continue
  dest="$QDIR/$(basename "$p")-$(date +%s)"
  mv "$p" "$dest" 2>/dev/null || { cp -a "$p" "$dest"; rm -rf "$p"; }
  echo "quarantined residual: $p -> $dest"
done < <(git ls-files --others --exclude-standard src/plugins/ 2>/dev/null || true)

# Restore tracked plugin/build/contract surfaces to HEAD
git restore --source=HEAD --worktree --staged -- \
  src/plugins/ \
  src/plugin_registry.zig \
  build.zig \
  tools/generate_plugin_registry.zig \
  tests/contracts/ 2>/dev/null \
  || git checkout HEAD -- \
    src/plugins/ \
    src/plugin_registry.zig \
    build.zig \
    tools/generate_plugin_registry.zig \
    tests/contracts/

test ! -e src/plugins/zig-self-improve
if ! git diff --quiet HEAD -- src/plugins/ src/plugin_registry.zig build.zig tools/generate_plugin_registry.zig tests/contracts/; then
  echo "quarantine-out-of-scope: FAIL still dirty after restore" >&2
  git diff --stat HEAD -- src/plugins/ src/plugin_registry.zig build.zig tools/generate_plugin_registry.zig tests/contracts/ >&2
  exit 1
fi

echo "quarantine-out-of-scope: OK"
