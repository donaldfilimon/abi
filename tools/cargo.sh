#!/usr/bin/env bash
# Invoke cargo on the pinned nightly toolchain.
#
# Two separate PATH hazards, both caused by Homebrew installing real rust
# binaries under /opt/homebrew/bin rather than rustup shims:
#
#  1. Bare `cargo` resolves to Homebrew's stable cargo, which neither honours
#     `rust-toolchain.toml` nor understands `+nightly`.
#
#  2. Less obviously, `rustup run nightly cargo` is NOT sufficient either.
#     `rustup run` resolves the *named* command against the toolchain but does
#     not put the toolchain's bin directory ahead of Homebrew on PATH. Cargo
#     then looks `rustc` up through PATH, finds Homebrew's stable rustc, and
#     reports the baffling "rustc 1.97.1 is not supported by ... requires
#     rustc 1.99" — a nightly cargo driving a stable rustc.
#
# Resolving the toolchain bin directory and prepending it fixes rustc, rustdoc,
# clippy-driver and rustfmt in one move.
set -euo pipefail

TOOLCHAIN_BIN="$(dirname "$(rustup which --toolchain nightly cargo)")"
export PATH="${TOOLCHAIN_BIN}:${PATH}"

# Swiftly installs a `cc`/`clang` shim that refuses to link when the repo's
# `.swift-version` pins a toolchain that is not installed. That is fatal for
# every Rust crate that links (including pure-Rust ones that never touch
# Swift), because `cc` is the default linker driver. Prefer the system C
# toolchain for linking unless the caller already set CC/CXX.
if [[ -z "${CC:-}" && -x /usr/bin/cc ]]; then
  export CC=/usr/bin/cc
fi
if [[ -z "${CXX:-}" && -x /usr/bin/c++ ]]; then
  export CXX=/usr/bin/c++
fi

exec "${TOOLCHAIN_BIN}/cargo" "$@"
