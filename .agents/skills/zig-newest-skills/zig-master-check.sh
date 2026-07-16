#!/usr/bin/env bash
# zig-newest-skills driver: switch the abi repo to the newest Zig master
# nightly (via zvm), build the real binaries, and verify the tree still
# compiles + runs against bleeding-edge std — surfacing API drift the repo's
# pinned toolchain (.zigversion) would otherwise hide.
#
# This is the inverse of "pin-safety": instead of guaranteeing the known-good
# version, it proves (or disproves) that abi keeps working on the latest Zig.
#
# Usage:
#   .agents/skills/zig-newest-skills/zig-master-check.sh            # use already-installed master
#   .agents/skills/zig-newest-skills/zig-master-check.sh --update   # re-fetch latest master first
#   .agents/skills/zig-newest-skills/zig-master-check.sh --smoke    # also run the full run-abi smoke
#   .agents/skills/zig-newest-skills/zig-master-check.sh --revert   # switch back to the .zigversion pin and exit
#
# Resolves the repo root from its own location; safe to run from any cwd.
set -uo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../../.." && pwd)
cd "$REPO_ROOT"

UPDATE=0; SMOKE=0; REVERT=0
for a in "$@"; do
    case "$a" in
        --update) UPDATE=1 ;;
        --smoke)  SMOKE=1 ;;
        --revert) REVERT=1 ;;
        *) echo "unknown flag: $a" >&2; exit 2 ;;
    esac
done

PIN=$(tr -d ' \t\r\n' < "$REPO_ROOT/.zigversion" 2>/dev/null || echo "")
ZVM_HOME="${ZVM_PATH:-$HOME/.zvm}"
say() { printf '\n=== %s ===\n' "$*"; }
# Presence check by on-disk dir, NOT `zvm list` — its output is ANSI-colored
# (e.g. "\x1b[32mmaster"), which defeats word-boundary greps and would cause a
# needless ~54MB re-download every run.
have_ver() { [ -x "$ZVM_HOME/$1/zig" ]; }

if ! command -v zvm >/dev/null 2>&1; then
    echo "FATAL: zvm not found on PATH. Install zvm first (https://github.com/tristanisham/zvm)." >&2
    exit 1
fi

# --- revert shortcut -----------------------------------------------------
if [ "$REVERT" -eq 1 ]; then
    say "revert to repo pin ($PIN)"
    [ -n "$PIN" ] || { echo "FATAL: .zigversion is empty; nothing to revert to." >&2; exit 1; }
    # zvm prompts interactively ([y/n]) when a version isn't installed and
    # errors on EOF in a non-tty. Install the pin explicitly first so revert
    # is non-interactive and works even after the pin dir was reclaimed.
    if ! have_ver "$PIN"; then
        echo "pin $PIN not installed locally; attempting to fetch it..."
        if ! zvm install "$PIN" 2>&1; then
            cat >&2 <<EOF

FATAL: cannot revert to $PIN.
zvm only serves the CURRENT master nightly from ziglang.org; old dev/nightly
builds are not archived there, so a pin like this is NOT re-downloadable once
its local ~/.zvm/<version> dir is gone. Options:
  - Stay on master (abi is verified to build + run on it): zvm use master
  - Bump .zigversion to the current master and treat that as the new pin
  - Restore $PIN from your own cached tarball into ~/.zvm/$PIN if you kept one
EOF
            exit 1
        fi
    fi
    zvm use "$PIN" || { echo "FATAL: 'zvm use $PIN' failed" >&2; exit 1; }
    hash -r 2>/dev/null || true
    echo "active zig: $(zig version)"
    exit 0
fi

say "before"
echo "repo pin (.zigversion): ${PIN:-<none>}"
echo "active zig now:         $(zig version 2>/dev/null || echo MISSING)"

# --- ensure master is available + selected -------------------------------
if [ "$UPDATE" -eq 1 ] || ! have_ver master; then
    say "install/update master"
    zvm install master || { echo "FATAL: zvm install master failed" >&2; exit 1; }
fi

say "select master"
zvm use master || { echo "FATAL: zvm use master failed" >&2; exit 1; }
hash -r 2>/dev/null || true
MASTER_VER=$(zig version)
echo "active zig: $MASTER_VER"
if [ -n "$PIN" ] && [ "$MASTER_VER" = "$PIN" ]; then
    echo "note: master currently EQUALS the repo pin ($PIN) — no drift to test today."
else
    echo "note: master ($MASTER_VER) differs from pin ($PIN) — this run tests forward drift."
fi

fail=0
gate() { # gate <label> -- <cmd...>
    local label="$1"; shift; [ "$1" = "--" ] && shift
    say "$label"
    echo "\$ $*"
    if "$@"; then echo "[ok] $label"; else echo "[FAIL] $label (exit $?)"; fail=$((fail+1)); fi
}

# --- gates, cheapest first ----------------------------------------------
# check-parity is std-only/host-target: it compiles even when the feature
# graph would not, so it isolates "toolchain can run our tools" from
# "feature graph compiles under this std".
gate "check-parity (std-only host gate)" -- zig build check-parity
gate "build cli"                         -- ./build.sh cli
gate "build mcp"                         -- ./build.sh mcp

# --- prove the binary actually runs (not just links) ---------------------
ABI="$REPO_ROOT/zig-out/bin/abi"
if [ -x "$ABI" ]; then
    gate "run: abi help"              -- "$ABI" help
    gate "run: abi backends"          -- "$ABI" backends
else
    echo "[FAIL] abi binary not produced at $ABI"; fail=$((fail+1))
fi

# --- optional deep smoke (reuses the run-abi harness) --------------------
if [ "$SMOKE" -eq 1 ]; then
    # Prefer canonical .agents path; fall back to Claude mirror layout.
    SMOKE_SH=""
    for candidate in \
        "$REPO_ROOT/.agents/skills/run-abi/smoke.sh" \
        "$REPO_ROOT/.claude/skills/run-abi/smoke.sh"
    do
        if [ -x "$candidate" ]; then
            SMOKE_SH="$candidate"
            break
        fi
    done
    if [ -n "$SMOKE_SH" ]; then
        gate "run-abi smoke (full CLI+MCP)" -- "$SMOKE_SH"
    else
        echo "note: --smoke requested but run-abi/smoke.sh not found/executable under .agents or .claude; skipping."
    fi
fi

# --- summary -------------------------------------------------------------
say "summary"
echo "tested zig:   $MASTER_VER"
echo "repo pin:     ${PIN:-<none>}"
echo "failed gates: $fail"
if [ "$fail" -eq 0 ]; then
    echo "RESULT: PASS — abi builds + runs on the newest Zig master."
else
    echo "RESULT: FAIL — master drift broke $fail gate(s)."
    echo "Revert to the known-good pin with:"
    echo "  $SCRIPT_DIR/$(basename "$0") --revert"
fi
exit "$fail"
