#!/usr/bin/env bash
# Compatibility entrypoint after the Rust rewrite.
# Prefer ./tools/check.sh and ./tools/cargo.sh directly.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cmd="${1:-check}"
shift || true
case "$cmd" in
  check|full-check)
    exec "$ROOT/tools/check.sh" "$@"
    ;;
  cli)
    exec "$ROOT/tools/cargo.sh" build -p abi-cli "$@"
    ;;
  mcp)
    exec "$ROOT/tools/cargo.sh" build -p abi-mcp "$@"
    ;;
  test)
    exec "$ROOT/tools/cargo.sh" test --workspace "$@"
    ;;
  fmt)
    exec "$ROOT/tools/cargo.sh" fmt --all "$@"
    ;;
  -l|list|--list)
    cat <<'EOF'
Rust rewrite entrypoints:
  ./tools/check.sh          primary gate (fmt, clippy, build, test, docs)
  ./tools/cargo.sh …        rustup nightly cargo wrapper
  ./build.sh check          → ./tools/check.sh
  ./build.sh cli            → cargo build -p abi-cli
  ./build.sh mcp            → cargo build -p abi-mcp
  ./build.sh test           → cargo test --workspace
  ./build.sh fmt            → cargo fmt --all
EOF
    ;;
  *)
    echo "unknown target: $cmd (try ./build.sh -l)" >&2
    exit 2
    ;;
esac
