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

exec "${TOOLCHAIN_BIN}/cargo" "$@"
