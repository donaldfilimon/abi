#!/bin/bash
set -euo pipefail

## merge_workspaces.sh
## Recursively merge histories of multiple workspaces into the current repo.
## This is a safe, read-only by-default operation. Pass --execute to actually perform merges.

WORKSPACES=()
EXECUTE=false

usage() {
  cat <<'USAGE'
Usage: tools/merge_workspaces.sh [--ws <path>]... [--execute]
Merge the histories of multiple local workspaces into this repository.
Notes:
- Workspaces must be Git repositories.
- This will create temporary remotes (ws-*) pointing to the given paths and merge their default branches
- By default the script only prints the planned actions. Use --execute to run them.
USAGE
  exit 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --ws)
      shift; if [[ -z "${1-}" ]]; then usage; fi
      WORKSPACES+=("$1"); shift ;; 
    --execute)
      EXECUTE=true; shift ;;
    *) usage ;;
  esac
done

if [ ${#WORKSPACES[@]} -eq 0 ]; then
  echo "No workspaces provided. Use --ws <path> to add workspaces." >&2
  usage
fi

ROOT=$(git rev-parse --show-toplevel 2>/dev/null || true)
if [ -z "$ROOT" ]; then
  echo "Error: not inside a git repository. Run from the root of the target repo." >&2
  exit 2
fi

BRANCH="merge-workspaces-review"
if $EXECUTE; then
  git checkout -b "$BRANCH" 2>/dev/null || git checkout "$BRANCH"
else
  echo "[DRY-RUN] Would switch to branch '$BRANCH' for merges"
fi

set -e

for wp in "${WORKSPACES[@]}"; do
  if [ ! -d "$wp" ]; then
    echo "Workspace path not found: $wp" >&2; exit 3
  fi
  if [ ! -d "$wp/.git" ]; then
    echo "Workspace is not a git repo: $wp" >&2; exit 4
  fi
  # Sanitize remote name
  name=$(basename "$wp" | tr -c '[:alnum:]' '-' )
  remote_name="ws_${name}"
  if git remote | grep -q "^$remote_name$"; then
    echo "Remote $remote_name already exists, skipping add."
  else
    echo "Adding remote $remote_name -> $wp"
    if $EXECUTE; then
      git remote add "$remote_name" "$wp" || { echo "Failed to add remote $remote_name"; exit 5; }
      git fetch "$remote_name" || { echo "Failed to fetch $remote_name"; exit 6; }
      # Determine default branch of the workspace remote
      default_branch=$(git remote show "$remote_name" | grep 'HEAD branch' | awk '{print $NF}')
      default_branch=${default_branch:-main}
      echo "Merging $remote_name/$default_branch into current branch"
      git merge "$remote_name/$default_branch" --allow-unrelated-histories -m "Merge workspace '$wp' from $remote_name" || {
        echo "Merge conflict while merging $remote_name/$default_branch"; exit 7; }
    else
      echo "Would add remote $remote_name -> $wp and fetch; would merge $remote_name/$(git -C "$ROOT" remote show "$remote_name" | grep 'HEAD branch' | awk '{print $NF}') with unrelated histories"
    fi
  fi
done

if $EXECUTE; then
  echo "Merges complete. Branch: $BRANCH"
else
  echo "Dry-run complete. No changes pushed. Switch to '$BRANCH' and review conflicts if any."
fi
