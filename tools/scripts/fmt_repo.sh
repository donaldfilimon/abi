#!/usr/bin/env bash
set -Eeuo pipefail
shopt -s inherit_errexit 2>/dev/null || true
unset CDPATH

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"

cd "$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  echo "Usage: ./tools/scripts/fmt_repo.sh [zig fmt args...]"
  echo "Formats only ABI-owned sources and skips vendored bootstrap fixtures."
  echo "Examples:"
  echo "  ./tools/scripts/fmt_repo.sh"
  echo "  ./tools/scripts/fmt_repo.sh --check"
  exit 0
fi

exec zig fmt "$@" build.zig build/ src/ tools/ examples/ tests/ bindings/ lang/
