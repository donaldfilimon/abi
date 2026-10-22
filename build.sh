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
    # Tests must run with stdin at EOF: `abi auth signin` reads a secret from
    # stdin when ABI_AUTH_TOKEN is unset and stdin is not a TTY, so the
    # no-token path test blocks forever on an inherited open stdin. Redirect
    # here so this alias cannot hang the same way tools/check.sh does.
    exec "$ROOT/tools/cargo.sh" test --workspace "$@" < /dev/null
    ;;
  fmt)
    exec "$ROOT/tools/cargo.sh" fmt --all "$@"
    ;;
  -l|list|--list)
    cat <<'EOF'
Rust rewrite entrypoints:
  ./tools/check.sh          primary gate (fmt, clippy, build, test, docs)
  ./tools/cargo.sh …        rustup stable-1.99.1 cargo wrapper
  ./build.sh check          → ./tools/check.sh
  ./build.sh cli            → ./tools/cargo.sh build -p abi-cli
  ./build.sh mcp            → ./tools/cargo.sh build -p abi-mcp
  ./build.sh test           → ./tools/cargo.sh test --workspace  (stdin is EOF)
  ./build.sh fmt            → ./tools/cargo.sh fmt --all
EOF
    ;;
  *)
    echo "unknown target: $cmd (try ./build.sh -l)" >&2
    exit 2
    ;;
esac
