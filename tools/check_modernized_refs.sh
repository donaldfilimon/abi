#!/usr/bin/env bash
# check_modernized_refs.sh — stale-reference scan for the Phase D scaffold.
#
# Wired into `zig build full-check` (release-readiness gate); also runs
# standalone. It greps every Markdown file under `modernized/` for fenced
# `src/...` path spans, resolves each against the repo root, and exits 1
# on any path that no longer exists. This keeps the scaffold honest
# without making `modernized/` a build root: if a `src/` leaf is renamed
# or removed, the scaffold's pointers are flagged before a release.
#
# No-op cleanly when `modernized/README.md` is absent (e.g. the scaffold
# has not been created yet, or was removed) — exit 0 with a one-line note.
#
# Portable on macOS system bash 3.2 (no mapfile).
#
# Usage:
#   tools/check_modernized_refs.sh          # scan; exit 1 on any stale ref
#
# Environment:
#   ZIG                      unused (kept for parity with bench_regress.sh)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCAFFOLD="$REPO_ROOT/modernized/README.md"

# --- no-op when the scaffold is absent ---
if [[ ! -f "$SCAFFOLD" ]]; then
    echo "check_modernized_refs: modernized/README.md absent — nothing to scan."
    exit 0
fi

# --- collect + resolve fenced `src/...` spans from modernized/ Markdown ---
# The grep pattern matches a backtick-fenced path that starts with `src/`
# and runs to the closing backtick (no spaces inside). Duplicate paths
# across files are collapsed with `sort -u`. Stream via while-read so we
# stay bash-3.2-safe (no mapfile).
stale=0
found=0
while IFS= read -r ref || [[ -n "$ref" ]]; do
    [[ -z "$ref" ]] && continue
    found=$((found + 1))
    # Strip the surrounding backticks.
    path="${ref//\`/}"
    full="$REPO_ROOT/$path"
    if [[ ! -e "$full" ]]; then
        echo "check_modernized_refs: STALE reference — $path (resolved from modernized/ Markdown)"
        stale=$((stale + 1))
    fi
done < <(grep -rhoE '`src/[^` ]+`' "$REPO_ROOT/modernized/" 2>/dev/null | sort -u || true)

if [[ "$found" -eq 0 ]]; then
    echo "check_modernized_refs: no fenced src/... references found in modernized/."
    exit 0
fi

if [[ "$stale" -gt 0 ]]; then
    echo "check_modernized_refs: FAIL — $stale stale src/ reference(s) in modernized/ scaffold."
    echo "  Update or remove the dangling pointer(s) in modernized/packages/*/README.md"
    echo "  or modernized/README.md so the scaffold stays honest against src/."
    exit 1
fi

echo "check_modernized_refs: PASS — all fenced src/ references in modernized/ resolve."
