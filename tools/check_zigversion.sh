#!/usr/bin/env bash
# Verify that PATH zig matches the repo .zigversion pin.
# Optional: if ZIG_VERSION is set (e.g. CI workflow env for mlugg/setup-zig),
# also assert it equals .zigversion so the two cannot silently drift.
#
# Usage:
#   ./tools/check_zigversion.sh
#   ./tools/check_zigversion.sh --github-actions
set -euo pipefail

GITHUB_ACTIONS_MODE=0
for arg in "$@"; do
  case "${arg}" in
    --github-actions) GITHUB_ACTIONS_MODE=1 ;;
    -h | --help)
      echo "Usage: $0 [--github-actions]"
      exit 0
      ;;
    *)
      echo "error: unknown argument: ${arg}" >&2
      exit 2
      ;;
  esac
done

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PIN_FILE="${REPO_ROOT}/.zigversion"

emit_error() {
  if ((GITHUB_ACTIONS_MODE)); then
    echo "::error::$*"
  else
    echo "error: $*" >&2
  fi
}

if [[ ! -f "${PIN_FILE}" ]]; then
  emit_error ".zigversion missing at ${PIN_FILE}"
  exit 1
fi

PIN="$(tr -d '[:space:]' <"${PIN_FILE}")"
if [[ -z "${PIN}" ]]; then
  emit_error ".zigversion is empty"
  exit 1
fi

if [[ -n "${ZIG_VERSION:-}" && "${ZIG_VERSION}" != "${PIN}" ]]; then
  emit_error "workflow ZIG_VERSION (${ZIG_VERSION}) != .zigversion pin (${PIN})"
  exit 1
fi

if ! command -v zig >/dev/null 2>&1; then
  emit_error "zig not on PATH. Install pin ${PIN} (zvm/zigup) and ensure it is on PATH."
  exit 1
fi

ACTUAL="$(zig version)"
if [[ "${ACTUAL}" != "${PIN}" ]]; then
  emit_error "PATH zig is ${ACTUAL}, expected pin ${PIN}. Update the toolchain and restart any runner service."
  exit 1
fi

echo "Using zig ${ACTUAL} (matches .zigversion)"
exit 0
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
