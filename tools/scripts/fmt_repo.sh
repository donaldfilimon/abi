#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
source "$SCRIPT_DIR/zig_toolchain.sh"

cd "$(abi_toolchain_repo_root)"

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  echo "Usage: ./tools/scripts/fmt_repo.sh [zig fmt args...]"
  echo "Formats only ABI-owned sources and skips vendored bootstrap fixtures."
  echo "Examples:"
  echo "  ./tools/scripts/fmt_repo.sh"
  echo "  ./tools/scripts/fmt_repo.sh --check"
  exit 0
fi

ZIG="$(abi_toolchain_resolve_active_zig)"

exec "$ZIG" fmt "$@" build.zig build/ src/ tools/ examples/ tests/ bindings/ lang/
