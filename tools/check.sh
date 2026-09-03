#!/usr/bin/env bash
# The repository gate — successor to `./build.sh check`.
#
# Covers the same ground the Zig gate did: format, lint, build, tests, and the
# frozen-contract checks. CI runs this same script on the self-hosted macOS
# ARM64 runner for trusted events and on a GitHub-hosted macOS runner for fork
# PRs; a separate GitHub-hosted Windows job checks credential ACLs. Treat a red
# result as blocking.
#
# Always invoked through tools/cargo.sh, never bare `cargo` — see that script for
# why that distinction matters here.
set -euo pipefail

cd "$(dirname "$0")/.."
CARGO="./tools/cargo.sh"

step() { printf '\n\033[1m==> %s\033[0m\n' "$1"; }

step "rustc / cargo versions"
"${CARGO}" --version
"${CARGO}" rustc --version -- --version 2>/dev/null | head -1 || true

step "repository policy tests"
python3 -m unittest discover -s tools/tests -p 'test_*.py' -v

step "CI contract (xtask Rust port - must match Python oracle)"
"${CARGO}" run -p xtask -- ci verify

step "Abbey contract corpus"
python3 tools/abbey_contracts.py verify contracts/abbey

step "Rust source size limits"
bash ./tools/check_rust_sizes.sh

step "format (check only)"
"${CARGO}" fmt --all -- --check

step "clippy (warnings are errors)"
"${CARGO}" clippy --workspace --all-targets -- -D warnings

step "build"
"${CARGO}" build --workspace --all-targets

step "tests"
# Tests run with stdin at EOF. `abi auth signin` reads a secret from stdin when
# ABI_AUTH_TOKEN is unset and stdin is not a TTY, so a test that exercises the
# no-token path (app::tests::auth_signin_without_token_fails_honestly) blocks
# forever on an inherited stdin that stays open. Redirecting here enforces the
# non-TTY-empty-stdin precondition those tests document instead of inheriting
# whatever the caller happened to have open.
"${CARGO}" test --workspace < /dev/null

step "local model device features"
if [[ "$(uname -s)" == "Darwin" ]]; then
  "${CARGO}" test -p abi-model-runtime --features metal < /dev/null
  RUSTDOCFLAGS="-D warnings" "${CARGO}" doc -p abi-model-runtime --features metal --no-deps --document-private-items --quiet
else
  printf 'Metal runtime evidence unavailable: this gate is not running on macOS\n'
fi
if command -v nvcc >/dev/null 2>&1; then
  "${CARGO}" check -p abi-model-runtime --features cuda
else
  printf 'CUDA feature compilation unavailable: nvcc is not installed\n'
fi

step "benchmark regression (same-system local guard)"
./tools/bench_regress.sh

step "docs (broken intra-doc links are errors)"
RUSTDOCFLAGS="-D warnings" "${CARGO}" doc --workspace --no-deps --quiet

printf '\n\033[1;32mcheck: all green\033[0m\n'
