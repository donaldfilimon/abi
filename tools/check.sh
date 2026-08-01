#!/usr/bin/env bash
# The repository gate — successor to `./build.sh check`.
#
# Covers the same ground the Zig gate did: format, lint, build, tests, and the
# frozen-contract checks. GitHub Actions on this repo is billing-locked, so this
# is the only gate that actually runs; treat a red result as blocking.
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

step "format (check only)"
"${CARGO}" fmt --all -- --check

step "clippy (warnings are errors)"
"${CARGO}" clippy --workspace --all-targets -- -D warnings

step "build"
"${CARGO}" build --workspace --all-targets

step "tests"
"${CARGO}" test --workspace

step "benchmark regression (same-system local guard)"
./tools/bench_regress.sh

step "docs (broken intra-doc links are errors)"
RUSTDOCFLAGS="-D warnings" "${CARGO}" doc --workspace --no-deps --quiet

printf '\n\033[1;32mcheck: all green\033[0m\n'
