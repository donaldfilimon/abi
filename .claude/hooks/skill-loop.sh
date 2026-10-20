#!/usr/bin/env bash
# Optional integration: hooks must never install packages or access a registry.
# Install skill-loop-claude separately to enable it; otherwise this is a no-op.
set -uo pipefail
if [ "$#" -ne 1 ]; then
  echo 'Expected pre-hook or post-hook' >&2
  exit 1
fi
case "${1:-}" in
  pre-hook|post-hook) ;;
  *) echo 'Expected pre-hook or post-hook' >&2; exit 1 ;;
esac
if command -v skill-loop-claude >/dev/null 2>&1; then
  exec skill-loop-claude "$1"
fi
