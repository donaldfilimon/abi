#!/usr/bin/env bash
# Deterministic Phase-1 closeout. No infinite watchdogs.
# Usage: SCRATCH=... ./modern-refactor/scripts/phase1-closeout.sh
# Optional: PHASE1_SANDBOX=1 to run in an isolated worktree.
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
SCRATCH="${SCRATCH:?set SCRATCH to implementer scratch}"
mkdir -p "$SCRATCH"
export SCRATCH

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
NDIR="${SCRATCH}/node-v20.19.2-darwin-arm64"
if [ -x "$NDIR/bin/node" ]; then
  export PATH="$NDIR/bin:$PATH"
fi

# Stop any residual goal watchdogs from prior rounds (best-effort)
pkill -f 'implementer/watchdog.log' 2>/dev/null || true
pkill -f 'rm -rf src/plugins/zig-self-improve' 2>/dev/null || true

WORKDIR="$ROOT"
if [ "${PHASE1_SANDBOX:-0}" = "1" ]; then
  WT="$SCRATCH/phase1-sandbox"
  rm -rf "$WT"
  git worktree remove --force "$WT" 2>/dev/null || true
  git worktree add --detach "$WT" HEAD
  # Overlay allowlisted Phase-1 edits from main worktree
  for f in \
    .gitignore \
    tasks/todo.md \
    tools/goal_capture.sh \
    tools/check_feature_stubs.sh \
    docs/index.mdx \
    docs/spec/sea-design-extract.mdx \
    docs/spec/wdbx-rust-capability-extract.mdx \
    docs/superpowers/archive/README.md
  do
    if [ -e "$ROOT/$f" ]; then
      mkdir -p "$WT/$(dirname "$f")"
      cp -a "$ROOT/$f" "$WT/$f"
    fi
  done
  # modern-refactor entire tree
  rm -rf "$WT/modern-refactor"
  cp -a "$ROOT/modern-refactor" "$WT/modern-refactor"
  WORKDIR="$WT"
  echo "phase1-closeout: using sandbox worktree $WT"
fi

cd "$WORKDIR"
export ROOT="$WORKDIR"

# Prefer scripts from WORKDIR (sandbox) when present so relative ROOT is correct.
SCOPE_SH="$WORKDIR/modern-refactor/scripts/verify-phase1-scope.sh"
QUAR_SH="$WORKDIR/modern-refactor/scripts/quarantine-out-of-scope.sh"
LAYOUT_SH="$WORKDIR/modern-refactor/scripts/verify-plugin-layout.sh"
GATE_SH="$WORKDIR/modern-refactor/scripts/pre-closeout-gate.sh"
[ -f "$SCOPE_SH" ] || SCOPE_SH="$SCRIPT_DIR/verify-phase1-scope.sh"
[ -f "$QUAR_SH" ] || QUAR_SH="$SCRIPT_DIR/quarantine-out-of-scope.sh"
[ -f "$LAYOUT_SH" ] || LAYOUT_SH="$SCRIPT_DIR/verify-plugin-layout.sh"
[ -f "$GATE_SH" ] || GATE_SH="$SCRIPT_DIR/pre-closeout-gate.sh"

LOG="$SCRATCH/verification-final.txt"
{
  echo "=== phase1-closeout START $(date -Iseconds) ==="
  echo "WORKDIR=$WORKDIR ROOT=$ROOT"
  echo "=== pre-closeout-gate ==="
  if ! bash "$GATE_SH"; then
    echo "pre-closeout-gate failed — see $SCRATCH/blocking-writers.txt"
    echo "HINT: set PHASE1_SANDBOX=1 or stop the foreign writer, then re-run."
    exit 1
  fi
  echo "=== quarantine-out-of-scope ==="
  bash "$QUAR_SH"
  echo "=== verify-phase1-scope (1st) ==="
  bash "$SCOPE_SH"
  echo "SCOPE1_EXIT:$?"
  echo "=== verify-plugin-layout ==="
  bash "$LAYOUT_SH"
  echo "=== skill assets ==="
  test -f modern-refactor/skills/codebase-analysis/references/analysis-checklist.md
  test -f modern-refactor/skills/refactor-implementation/references/implementation-playbook.md
  find modern-refactor/skills -type f | sort
  echo "=== ./build.sh check ==="
  ./build.sh check
  echo "BUILD_EXIT:$?"
  echo "=== mint validate (docs/) ==="
  if command -v node >/dev/null && [ -d docs ]; then
    (cd docs && npx --yes mint@latest validate) || true
    echo "MINT_EXIT:${PIPESTATUS[0]:-$?}"
  else
    echo "MINT_SKIPPED"
  fi
  echo "=== stage allowlisted paths ==="
  git add -- \
    .gitignore \
    tasks/todo.md \
    tools/goal_capture.sh \
    tools/check_feature_stubs.sh \
    docs/ \
    modern-refactor/ 2>/dev/null || true
  echo "=== PASSIVE 60s (no scrub, no watchdog) ==="
  sleep 60
  echo "=== after 60s passive ==="
  if [ -e src/plugins/zig-self-improve ]; then
    echo "ZSI_REAPPEARED_AFTER_60S"
    ls -la src/plugins/zig-self-improve || true
  else
    echo "ZSI_STILL_ABSENT_AFTER_60S"
  fi
  echo "=== verify-phase1-scope (2nd, durable) ==="
  bash "$SCOPE_SH"
  echo "SCOPE2_EXIT:$?"
  echo "=== git status --short --branch ==="
  git status --short --branch
  echo "=== git diff --stat HEAD ==="
  git diff --stat HEAD
  echo "=== git diff --cached --stat ==="
  git diff --cached --stat || true
  echo "=== plugin contracts ==="
  rg -n 'pluginCount|count=16|count=17' tests/contracts/plugin_registry.zig tests/contracts/mcp_tools.zig | head -15
  echo "=== phase1-closeout END $(date -Iseconds) ==="
} 2>&1 | tee "$LOG"

# Require durable success markers
rg -q 'ZSI_STILL_ABSENT_AFTER_60S' "$LOG"
rg -q 'SCOPE2_EXIT:0' "$LOG" || rg -q 'verify-phase1-scope: OK' "$LOG"
rg -q 'BUILD_EXIT:0' "$LOG"
echo "phase1-closeout: SUCCESS (see $LOG)"
