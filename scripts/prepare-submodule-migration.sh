#!/usr/bin/env bash
set -euo pipefail

# Prepare the protocol-parity submodule migration
# Usage:
#   ./scripts/prepare-submodule-migration.sh -u <url> [-c <commit|tag|branch>] [-b <branch>] [-t <tag>]

usage() {
  echo "Usage: $0 -u <url> [-c <commit|tag|branch>] [-b <branch>] [-t <tag>]" >&2
  exit 1
}

URL=""
PIN=""
BRANCH=""
TAG=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    -u|--url)
      URL="$2"; shift 2;;
    -c|--commit-or-tag)
      PIN="$2"; shift 2;;
    -b|--branch)
      BRANCH="$2"; shift 2;;
    -t|--tag)
      TAG="$2"; shift 2;;
    *)
      usage;;
  esac
done

if [[ -z "$URL" ]]; then
  echo "Error: --url is required" >&2
  usage
fi

echo "Config: URL=$URL PIN_OR_TAG=$PIN BRANCH=$BRANCH TAG=$TAG"

# Update .gitmodules with the real URL
if [[ -f .gitmodules ]]; then
  if grep -q "url = https://REPLACE-WITH-REMOTE/protocol-parity.git" .gitmodules; then
    sed -i '' "s#url = https://REPLACE-WITH-REMOTE/protocol-parity.git#url = ${URL}#" .gitmodules
  else
    # Fallback: replace any protocol-parity submodule url line
    sed -i '' "/submodule \"protocol-parity\"/!b;n;c\turl = ${URL}" .gitmodules
  fi
else
  echo "Warning: .gitmodules not found in repo root" >&2
fi

echo "Updated .gitmodules with new URL. Syncing submodules..."
git submodule sync

if [[ -d protocol-parity ]]; then
  echo "protocol-parity directory exists; ensuring URL is set..."
  git config -f .gitmodules submodule.protocol-parity.url "$URL"
  git submodule update --init --recursive protocol-parity
else
  echo "Adding protocol-parity as a new submodule..."
  git submodule add "$URL" protocol-parity
fi

if [[ -n "$PIN" ]]; then
  echo "Note: pinning to commit/tag/branch not automated in this script; please ensure ${PIN} exists in the submodule."
fi

git add .gitmodules protocol-parity || true
COMMIT_MSG="chore: migrate protocol-parity to real submodule at $URL"
git commit -m "$COMMIT_MSG" || true

echo "Submodule migration scaffolding completed."
