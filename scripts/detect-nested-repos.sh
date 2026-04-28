#!/usr/bin/env bash
set -euo pipefail

# Detect stray nested Git repositories inside the repo (excluding root).
ROOT=$(git rev-parse --show-toplevel 2>/dev/null || pwd)
echo "[preflight] Scanning for stray nested repos under: ${ROOT}"

found=0
while IFS= read -r -d '' gitdir; do
  dir=$(dirname "$gitdir")
  dir=$(cd "$dir" && pwd)
  if [ "$dir" = "$ROOT" ]; then
    continue
  fi
  if [ -d "$dir/.git" ]; then
    echo "[preflight] Detected stray nested repo: $dir"
    found=1
  else
    echo "[preflight] Found .git at $gitdir but parent $dir missing .git; skipping."
  fi
done < <(find . -type d -name ".git" -print0)

if [ "$found" -eq 1 ]; then
  echo "[preflight] Stray nested repos detected. Aborting CI." >&2
  exit 1
fi

echo "[preflight] No stray nested repos detected."
exit 0
