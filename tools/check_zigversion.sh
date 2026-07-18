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
