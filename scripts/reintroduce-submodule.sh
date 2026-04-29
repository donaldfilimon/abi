#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 2 ]; then
  echo "Usage: $0 <path> <url> [branch-or-commit]" >&2
  exit 2
fi

SUBPATH="$1"
URL="$2"
PIN="${3:-}"

echo "Re introducing submodule at '$SUBPATH' from '$URL'"

if [ -d "$SUBPATH" ]; then
  echo "Backing up existing directory at $SUBPATH to ${SUBPATH}.bak"
  mv "$SUBPATH" "${SUBPATH}.bak" || { echo "Backup failed"; exit 1; }
  git rm -r --cached "$SUBPATH" || true
fi

git submodule add "$URL" "$SUBPATH"
git commit -m "submodule: reintroduce ${SUBPATH} from ${URL}"

echo "Done. If you want to pin to a specific commit, run:
  cd $SUBPATH
  git fetch --all
  git checkout <commit-or-tag>
  cd ..
  git add $SUBPATH
  git commit -m 'submodule: pin ${SUBPATH} to <commit-or-tag>'"
