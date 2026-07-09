#!/usr/bin/env bash
# Hard scope gate for modern-refactor Phase 1 goal.
# Fails if any dirty/untracked path is outside the allowlist, or if
# zig-self-improve / src/plugins pollution reappears.
set -euo pipefail

# Prefer explicit ROOT (sandbox/worktree closeout); else repo containing this script.
if [ -n "${ROOT:-}" ] && [ -d "$ROOT" ]; then
  :
else
  ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
fi
cd "$ROOT"

fail=0

# Explicit filesystem denials (plugin discovery surface)
# Empty dirs still poison filesystem plugin discovery
if [ -e src/plugins/zig-self-improve ] || [ -d src/plugins/zig-self-improve ]; then
  echo "DENY: src/plugins/zig-self-improve exists on disk" >&2
  ls -la src/plugins/zig-self-improve 2>&1 || true
  fail=1
fi
# Any stray untracked under src/plugins/
while IFS= read -r p; do
  [ -z "$p" ] && continue
  echo "DENY: untracked under src/plugins: $p" >&2
  fail=1
done < <(git ls-files --others --exclude-standard src/plugins/ 2>/dev/null || true)
if ! git diff --quiet HEAD -- src/plugins/ 2>/dev/null; then
  echo "DENY: dirty paths under src/plugins/" >&2
  git diff --stat HEAD -- src/plugins/ >&2 || true
  fail=1
fi
if ! git diff --quiet HEAD -- build.zig src/plugin_registry.zig tools/generate_plugin_registry.zig tests/contracts/ 2>/dev/null; then
  echo "DENY: dirty build/registry/contract paths" >&2
  git diff --stat HEAD -- build.zig src/plugin_registry.zig tools/generate_plugin_registry.zig tests/contracts/ >&2 || true
  fail=1
fi

# Every short-status path must match allowlist
# Format: XY path  or ?? path
while IFS= read -r line; do
  [ -z "$line" ] && continue
  # skip branch header
  case "$line" in
    \#\#*) continue ;;
  esac
  path="${line:3}"
  path="${path#\"}"
  path="${path%\"}"
  # renames: "old -> new"
  case "$path" in
    *' -> '*) path="${path##* -> }" ;;
  esac

  allowed=0
  case "$path" in
    .gitignore) allowed=1 ;;
    tasks/todo.md) allowed=1 ;;
    tools/goal_capture.sh) allowed=1 ;;
    tools/check_feature_stubs.sh) allowed=1 ;;
    docs/*) allowed=1 ;;
    modern-refactor|modern-refactor/*) allowed=1 ;;
  esac

  if [ "$allowed" -ne 1 ]; then
    echo "DENY out-of-scope dirty/untracked: $path (status line: $line)" >&2
    fail=1
  fi
done < <(git status --short --untracked-files=all)

# Legacy CLI names must not appear as added lines in the Phase-1 diff
if git diff HEAD -- .gitignore docs tools/goal_capture.sh tools/check_feature_stubs.sh tasks/todo.md 2>/dev/null \
  | rg -q '^\+[^+].*\b(version|doctor|features|platform|connectors|search|info|chat|db|serve)\b'; then
  echo "DENY: legacy CLI name hit in Phase-1 added diff lines" >&2
  fail=1
fi

if [ "$fail" -ne 0 ]; then
  echo "verify-phase1-scope: FAIL" >&2
  exit 1
fi
echo "verify-phase1-scope: OK"
git status --short --branch
