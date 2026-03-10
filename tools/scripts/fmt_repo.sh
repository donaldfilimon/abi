#!/usr/bin/env bash
set -euo pipefail

cd "$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  echo "Usage: ./tools/scripts/fmt_repo.sh [zig fmt args...]"
  echo "Formats only ABI-owned sources and skips vendored bootstrap fixtures."
  echo "Examples:"
  echo "  ./tools/scripts/fmt_repo.sh"
  echo "  ./tools/scripts/fmt_repo.sh --check"
  exit 0
fi

ZIG="${ZIG_REAL:-${ZIG:-zig}}"

exec "$ZIG" fmt "$@" build.zig build/ src/ tools/ examples/
