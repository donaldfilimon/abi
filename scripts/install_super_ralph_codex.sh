#!/usr/bin/env bash
# Install Super Ralph skill into Codex (~/.codex/skills/super-ralph).
# Run from ABI repo root: ./scripts/install_super_ralph_codex.sh

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
codex_skill_dir="${CODEX_SKILLS_DIR:-$HOME/.codex/skills}"
target="$codex_skill_dir/super-ralph"
source_dir="$repo_root/codex/super-ralph"

if [[ ! -d "$source_dir" ]]; then
    echo "ERROR: Source not found: $source_dir" >&2
    echo "Run this script from the ABI repo root." >&2
    exit 1
fi

mkdir -p "$codex_skill_dir"
if [[ -d "$target" ]]; then
    echo "Updating existing $target"
    cp -Rf "$source_dir"/* "$target"/
else
    echo "Installing Super Ralph to $target"
    cp -R "$source_dir" "$target"
fi
echo "OK: Super Ralph installed. Codex can use skill: super-ralph"
