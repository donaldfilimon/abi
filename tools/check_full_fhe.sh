#!/usr/bin/env bash
# Compile and execute the optional TFHE-rs integration in release mode.
set -euo pipefail

repo_dir="$(cd "$(dirname "$0")/.." && pwd)"
cd "$repo_dir"

./tools/cargo.sh test --release -p abi-wdbx \
  --features full-fhe,experimental-dghv-bootstrap \
  fhe::tests::educational_refresh_has_complete_truth_tables_and_fresh_ciphertexts
./tools/cargo.sh test --release -p abi-wdbx \
  --features full-fhe \
  fhe::tfhe_demo::tests::pinned_tfhe_apis_execute_with_ephemeral_keys
./tools/cargo.sh test --release -p abi-cli \
  --features full-fhe,experimental-dghv-bootstrap \
  wdbx::tests::secure_additive_feature_surfaces_are_explicit_when_disabled

echo "full-fhe gate: PASS (TFHE-rs integration substrate; ABI has no independent cryptographic audit)"
