#!/usr/bin/env bash
# Fail if .zigversion drifts from CI's ZIG_VERSION.
# Wired into `zig build check` so pin edits cannot land without a matching CI bump.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PIN_FILE="$ROOT/.zigversion"
CI_FILE="$ROOT/.github/workflows/ci.yml"

if [[ ! -f "$PIN_FILE" ]]; then
  echo "check_zigversion: missing $PIN_FILE" >&2
  exit 1
fi
if [[ ! -f "$CI_FILE" ]]; then
  echo "check_zigversion: missing $CI_FILE" >&2
  exit 1
fi

PIN="$(tr -d ' \t\r\n' < "$PIN_FILE")"
if [[ -z "$PIN" ]]; then
  echo "check_zigversion: .zigversion is empty" >&2
  exit 1
fi

CI_PIN="$(grep -E '^[[:space:]]*ZIG_VERSION:' "$CI_FILE" | head -n1 | sed -E 's/^[[:space:]]*ZIG_VERSION:[[:space:]]*//' | tr -d ' \t\r\n"')"
if [[ -z "$CI_PIN" ]]; then
  echo "check_zigversion: could not parse ZIG_VERSION from $CI_FILE" >&2
  exit 1
fi

if [[ "$PIN" != "$CI_PIN" ]]; then
  echo "check_zigversion: pin drift" >&2
  echo "  .zigversion:              $PIN" >&2
  echo "  ci.yml ZIG_VERSION:       $CI_PIN" >&2
  echo "Keep them identical (see AGENTS.md)." >&2
  exit 1
fi

ACTIVE="$(zig version 2>/dev/null || true)"
if [[ -n "$ACTIVE" && "$ACTIVE" != "$PIN" ]]; then
  echo "check_zigversion: warning: active zig ($ACTIVE) ≠ pin ($PIN)" >&2
  echo "  Select the pin with: zvm use $PIN   (or zigup $PIN)" >&2
  # Soft warn only — CI installs the pin; local PATH may differ during exploration.
fi

echo "check_zigversion: ok ($PIN)"
