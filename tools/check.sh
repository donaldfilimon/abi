#!/usr/bin/env bash
# The repository gate — successor to `./build.sh check`.
#
# Covers the same ground the Zig gate did: format, lint, build, tests, and the
# frozen-contract checks. CI runs this same script on the self-hosted macOS
# ARM64 runner for trusted events; fork PRs get no CI job. Treat a red result
# as blocking.
#
# Always invoked through tools/cargo.sh, never bare `cargo` — see that script for
# why that distinction matters here.
set -euo pipefail

cd "$(dirname "$0")/.."
CARGO="./tools/cargo.sh"

# The Python policy tooling needs 3.11+ (tomllib, PEP 585 generics at
# runtime). macOS still ships /usr/bin/python3 at 3.9, and /usr/bin precedes
# the Homebrew interpreter on the default PATH, so a bare `python3` can resolve
# to an interpreter that dies with `ModuleNotFoundError: No module named
# 'tomllib'`. Select a capable interpreter explicitly and fail with an
# actionable message instead of a traceback.
PYTHON=""
candidates=()
[[ -n "${ABI_PYTHON:-}" ]] && candidates+=("${ABI_PYTHON}")
while IFS= read -r versioned; do
  # Pure-bash filter: no external grep/sort, so a stripped PATH cannot
  # silently drop every candidate and misreport a capable interpreter as
  # missing.
  [[ "$versioned" =~ ^python3\.[0-9]+$ ]] && candidates+=("$versioned")
done < <(compgen -c 'python3.' 2>/dev/null)
candidates+=(python3 python)
for candidate in "${candidates[@]}"; do
  [[ -n "$candidate" ]] || continue
  resolved="$(command -v "$candidate" 2>/dev/null || true)"
  [[ -n "$resolved" ]] || continue
  # DISTINGUISHING exit status, not "exit 0": a program that ignores its
  # `-c` argument and exits 0 (echo, true, a broken shim) must not be
  # accepted as an interpreter, or the policy tests and the Abbey Python
  # oracle would silently become no-ops while the gate still reported green.
  probe_status=0
  "$resolved" -c 'import sys, tomllib; sys.exit(3 if sys.version_info >= (3, 11) else 4)' \
    >/dev/null 2>&1 || probe_status=$?
  if [[ "$probe_status" -eq 3 ]]; then
    PYTHON="$resolved"
    break
  fi
done
if [[ -z "$PYTHON" ]]; then
  printf 'check: no Python 3.11+ interpreter found (tomllib is required)\n' >&2
  printf 'check: install one, or set ABI_PYTHON=/path/to/python3.11+\n' >&2
  exit 1
fi

step() { printf '\n\033[1m==> %s\033[0m\n' "$1"; }

step "rustc / cargo versions"
"${CARGO}" --abi-qualify-toolchain

step "repository policy tests"
"${PYTHON}" -m unittest discover -s tools/tests -p 'test_*.py' -v

step "CI contract (xtask Rust port - must match Python oracle)"
"${CARGO}" run --locked -p xtask -- ci verify

step "Abbey contract corpus (Python oracle + Rust port - both must pass; Python authoritative until byte-identical)"
"${PYTHON}" tools/abbey_contracts.py verify contracts/abbey
"${CARGO}" run --locked -p xtask -- abbey verify contracts/abbey

step "required dependency-security policy"
ABI_DEP_SCAN_REQUIRE=1 ./tools/security/run-dep-scan.sh

step "Rust source size limits"
bash ./tools/check_rust_sizes.sh

step "format (check only)"
"${CARGO}" fmt --all -- --check

step "clippy (warnings are errors)"
"${CARGO}" clippy --locked --workspace --all-targets -- -D warnings

step "build"
"${CARGO}" build --locked --workspace --all-targets

step "tests"
# Tests run with stdin at EOF. `abi auth signin` reads a secret from stdin when
# ABI_AUTH_TOKEN is unset and stdin is not a TTY, so a test that exercises the
# no-token path (app::tests::auth_signin_without_token_fails_honestly) blocks
# forever on an inherited stdin that stays open. Redirecting here enforces the
# non-TTY-empty-stdin precondition those tests document instead of inheriting
# whatever the caller happened to have open.
"${CARGO}" test --locked --workspace < /dev/null

step "required sibling WDBX contract conformance"
"${CARGO}" test --locked --manifest-path ../wdbx/Cargo.toml -p abi-wdbx --test abbey_contracts --test v3_cross_language_commitment --test v3_cross_language_episode < /dev/null

step "local model device features"
if [[ "$(uname -s)" == "Darwin" ]]; then
  "${CARGO}" test --locked -p abi-model-runtime --features metal < /dev/null
  RUSTDOCFLAGS="-D warnings" "${CARGO}" doc --locked -p abi-model-runtime --features metal --no-deps --document-private-items --quiet
else
  printf 'Metal runtime evidence unavailable: this gate is not running on macOS\n'
fi
if command -v nvcc >/dev/null 2>&1; then
  "${CARGO}" check --locked -p abi-model-runtime --features cuda
else
  printf 'CUDA feature compilation unavailable: nvcc is not installed\n'
fi

step "benchmark regression (same-system local guard)"
./tools/bench_regress.sh

step "docs (broken intra-doc links are errors)"
RUSTDOCFLAGS="-D warnings" "${CARGO}" doc --locked --workspace --no-deps --quiet

step "release binaries"
"${CARGO}" build --locked --release -p abi-cli -p abi-mcp

printf '\n\033[1;32mcheck: all green\033[0m\n'
