#!/usr/bin/env bash
# PostToolUse: format one Rust edit using this repository's selected toolchain.
set -uo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)" || exit 1
file="$(jq -r '.tool_input.file_path // .tool_response.filePath // empty')" || exit 1
case "$file" in
  *.rs) ;;
  *) exit 0 ;;
esac
# Resolve the edit against the caller's directory before selecting our toolchain.
case "$file" in
  /*) ;;
  *) file="$PWD/$file" ;;
esac
cd -- "$root" || exit 1
# rustup resolves rust-toolchain.toml here; do not select the floating nightly.
formatter="$(rustup which rustfmt)" || exit 1
exec "$formatter" --edition 2024 "$file"
