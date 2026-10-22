#!/usr/bin/env bash
# Invoke cargo on the repository-selected toolchain.
#
# Two separate PATH hazards, both caused by Homebrew installing real rust
# binaries under /opt/homebrew/bin rather than rustup shims:
#
#  1. Bare `cargo` resolves to Homebrew's stable cargo, which neither honours
#     `rust-toolchain.toml` nor understands rustup toolchain selectors.
#
#  2. Less obviously, `rustup run <toolchain> cargo` is NOT sufficient either.
#     `rustup run` resolves the *named* command against the toolchain but does
#     not put the toolchain's bin directory ahead of Homebrew on PATH. Cargo
#     then looks `rustc` up through PATH, finds Homebrew's stable rustc, and
#     reports the baffling "rustc 1.97.1 is not supported by ... requires
#     rustc 1.99" — a mismatched cargo driving a different rustc.
#
# Resolving the toolchain bin directory and prepending it fixes rustc, rustdoc,
# clippy-driver and rustfmt in one move.
set -euo pipefail

if [[ "${1:-}" == "--abi-qualify-toolchain" ]]; then
  # Preserve the gate's Python interpreter before /usr/bin linker precedence.
  # It must be 3.11+: the policy script imports tomllib, which macOS' system
  # /usr/bin/python3 (3.9) does not ship. Probe versioned interpreters
  # generically so a host carrying only e.g. python3.15 still qualifies.
  #
  # The probe must use a DISTINGUISHING exit status, not "exit 0": a program
  # that ignores its `-c` argument and exits 0 (echo, true, a broken shim)
  # would otherwise be accepted as an interpreter, and the policy script would
  # never run — silently disabling the compiler-substitution check this
  # qualification exists to enforce. Exit 3 means "a real 3.11+ interpreter".
  POLICY_PYTHON=""
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
    probe_status=0
    "$resolved" -c 'import sys, tomllib; sys.exit(3 if sys.version_info >= (3, 11) else 4)' \
      >/dev/null 2>&1 || probe_status=$?
    if [[ "$probe_status" -eq 3 ]]; then
      POLICY_PYTHON="$resolved"
      break
    fi
  done
  if [[ -z "$POLICY_PYTHON" ]]; then
    printf 'stable qualification requires a Python 3.11+ interpreter (tomllib)\n' >&2
    printf 'install one, or set ABI_PYTHON=/path/to/python3.11+\n' >&2
    exit 1
  fi
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# Only resolution changes directory: Cargo retains the caller's working directory.
TOOLCHAIN_CARGO="$(cd "${REPO_ROOT}" && rustup which cargo)"
TOOLCHAIN_BIN="$(dirname "${TOOLCHAIN_CARGO}")"

# Swiftly installs a `cc`/`clang` shim that refuses to link when the repo's
# `.swift-version` pins a toolchain that is not installed. That is fatal for
# every Rust crate that links (including pure-Rust ones that never touch
# Swift), because rustc looks up `cc` on PATH rather than only honouring $CC.
# Put the repository-selected toolchain and /usr/bin ahead of Swiftly so the system
# compiler is the one that links.
# `$HOME` is dereferenced twice below: the required dependency scanner
# (cargo-audit/cargo-deny) lives in `~/.cargo/bin`, and rustup's dyld fallback
# list includes `~/lib`. `set -u` is active, so an unset HOME must not abort
# the wrapper before any work happens (a minimal container or an `env -i` job
# has no HOME). Derive both entries from a guarded HOME and skip them when it
# is missing, exactly as the sibling `tools/security/run-dep-scan.sh` does.
HOME_ROOT="${HOME:-}"
if [[ -z "${HOME_ROOT}" || "${HOME_ROOT}" == "/" ]]; then
  HOME_ROOT=""
fi
CARGO_USER_BIN=""
HOME_LIB=""
if [[ -n "${HOME_ROOT}" ]]; then
  CARGO_USER_BIN="${HOME_ROOT}/.cargo/bin"
  HOME_LIB="${HOME_ROOT}/lib"
fi
export PATH="${TOOLCHAIN_BIN}:/usr/bin${CARGO_USER_BIN:+:${CARGO_USER_BIN}}:${PATH}"
if [[ -z "${CC:-}" && -x /usr/bin/cc ]]; then
  export CC=/usr/bin/cc
fi
if [[ -z "${CXX:-}" && -x /usr/bin/c++ ]]; then
  export CXX=/usr/bin/c++
fi

# Exec'ing the toolchain's cargo directly also skips the one piece of
# environment the rustup proxy would have set: DYLD_FALLBACK_LIBRARY_PATH with
# the toolchain's lib/ directory first. rustc on Apple targets strips release
# binaries with lib/rustlib/<host>/bin/rust-objcopy, which links
# @rpath/libLLVM.dylib through an rpath of @loader_path/../lib, and the rustc
# component only ships libLLVM.dylib in lib/. Without the fallback path every
# release link prints "stripping debug info with `rust-objcopy` failed: signal: 6
# (SIGABRT)" and leaves the binary unstripped (rust-lld has the same
# dependency). Mirror rustup's exact value: the variable replaces dyld's default
# fallback list, so the system entries must stay on it.
TOOLCHAIN_LIB="${TOOLCHAIN_BIN%/bin}/lib"
if [[ -n "${DYLD_FALLBACK_LIBRARY_PATH:-}" ]]; then
  export DYLD_FALLBACK_LIBRARY_PATH="${TOOLCHAIN_LIB}:${DYLD_FALLBACK_LIBRARY_PATH}"
else
  export DYLD_FALLBACK_LIBRARY_PATH="${TOOLCHAIN_LIB}${HOME_LIB:+:${HOME_LIB}}:/usr/local/lib:/usr/lib"
fi

if [[ "${1:-}" == "--abi-qualify-toolchain" ]]; then
  # Developer overrides remain available for ordinary builds; qualification is
  # deliberately stricter and must describe the compiler Cargo will execute.
  "${POLICY_PYTHON}" "${REPO_ROOT}/tools/toolchain_policy.py"
  printf 'compiler: %s\n' "${TOOLCHAIN_BIN}/rustc"
  COMPILER_VERSION="$("${TOOLCHAIN_BIN}/rustc" -vV)"
  printf '%s\n' "${COMPILER_VERSION}"
  if ! printf '%s\n' "${COMPILER_VERSION}" | /usr/bin/grep -qx 'release: 1.99.1'; then
    printf 'stable qualification requires rustc release 1.99.1\n' >&2
    exit 1
  fi
  "${TOOLCHAIN_BIN}/cargo" --version
  exit 0
fi

exec "${TOOLCHAIN_BIN}/cargo" "$@"
