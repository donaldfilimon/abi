#!/usr/bin/env bash
# Cleanup stray nested Git repositories inside the current repo, excluding the root repo.
set -euo pipefail

# Determine repo root
ROOT=$(git rev-parse --show-toplevel 2>/dev/null || pwd)
echo "Scanning for stray nested Git repos inside: $ROOT"

# Find all .git directories and evaluate their parents
find . -type d -name ".git" -print0 | while IFS= read -r -d '' gitdir; do
  repo_dir=$(dirname "$gitdir")
  # Normalize path
  repo_dir=$(cd "$repo_dir" && pwd)
  # Skip the repository root
  if [ "$repo_dir" = "$ROOT" ]; then
    continue
  fi
  if [ -d "$repo_dir/.git" ]; then
    echo "Removing stray nested repo: $repo_dir"
    rm -rf "$repo_dir"
  else
    echo "Warning: Found .git at $gitdir but parent $repo_dir does not contain a .git; skipping."
  fi
done

echo "Done."
