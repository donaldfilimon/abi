#!/usr/bin/env bash
# Invoke cargo on the pinned nightly toolchain.
#
# Two separate PATH hazards, both caused by Homebrew installing real rust
# binaries under /opt/homebrew/bin rather than rustup shims:
#
#  1. Bare `cargo` resolves to Homebrew's stable cargo, which neither honours
#     `rust-toolchain.toml` nor understands `+nightly`.
#
#  2. Less obviously, `rustup run <toolchain> cargo` is NOT sufficient either.
#     `rustup run` resolves the *named* command against the toolchain but does
#     not put the toolchain's bin directory ahead of Homebrew on PATH. Cargo
#     then looks `rustc` up through PATH, finds Homebrew's stable rustc, and
#     reports the baffling "rustc 1.97.1 is not supported by ... requires
#     rustc 1.99" — a nightly cargo driving a stable rustc.
#
# Resolving the toolchain bin directory and prepending it fixes rustc, rustdoc,
# clippy-driver and rustfmt in one move.
set -euo pipefail

TOOLCHAIN_BIN="$(dirname "$(rustup which cargo)")"

# Swiftly installs a `cc`/`clang` shim that refuses to link when the repo's
# `.swift-version` pins a toolchain that is not installed. That is fatal for
# every Rust crate that links (including pure-Rust ones that never touch
# Swift), because rustc looks up `cc` on PATH rather than only honouring $CC.
# Put the repository-selected toolchain and /usr/bin ahead of Swiftly so the system
# compiler is the one that links.
export PATH="${TOOLCHAIN_BIN}:/usr/bin:${PATH}"
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
  export DYLD_FALLBACK_LIBRARY_PATH="${TOOLCHAIN_LIB}:${HOME}/lib:/usr/local/lib:/usr/lib"
fi

exec "${TOOLCHAIN_BIN}/cargo" "$@"
