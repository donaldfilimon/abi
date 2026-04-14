#!/bin/bash
set -euo pipefail

## crossbuild.sh — Modernized lightweight cross-build helper
## This is a safe, minimal interface to enumerate targets and perform a dry-run cross-build.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
OUT_BASE="$REPO_ROOT/zig-out"

usage() {
  cat <<'EOS'
Usage: crossbuild.sh [--list | <target> | --help]

Supported targets (dry-run):
  linux-x86_64
  linux-arm64
  macos-x86_64
  macos-arm64
  windows-x86_64
  all
EOS
  exit 1
}

if [ "${1-}" = "--list" ]; then
  usage
fi

if [ "${1-}" = "--help" ]; then
  usage
fi

targets=(
  "linux-x86_64"
  "linux-arm64"
  "macos-x86_64"
  "macos-arm64"
  "windows-x86_64"
  "all"
)

if [ -z "${1-}" ]; then
  echo "No target provided. Use --list to view targets." 1>&2
  exit 2
fi

target="$1"
if [ "$target" = "all" ]; then
  echo "Dry-run: Cross-building all targets (no actions performed in this environment)."
  for t in "${targets[@]}"; do
    if [ "$t" != "all" ]; then
      echo " - would cross-build: $t"
    fi
  done
  exit 0
fi

case "$target" in
  linux-x86_64|linux-arm64|macos-x86_64|macos-arm64|windows-x86_64)
    echo "Dry-run: would cross-build for target: $target"
    echo "Note: actual cross-compile commands depend on the project build.zig and environment."
    ;;
  *)
    echo "Unknown target: $target" 1>&2
    usage
    ;;
esac

exit 0
