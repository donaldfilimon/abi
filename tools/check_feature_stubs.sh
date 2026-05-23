#!/usr/bin/env bash
set -euo pipefail

# Compile CLI with each feature disabled to ensure mod/stub trees build.
FLAGS=(
  feat-ai
  feat-gpu
  feat-tui
  feat-accelerator
  feat-shader
  feat-mlir
  feat-mobile
  feat-wdbx
  feat-os-control
)

for flag in "${FLAGS[@]}"; do
  echo "check_feature_stubs: zig build cli -D${flag}=false"
  zig build cli "-D${flag}=false"
done

echo "check_feature_stubs: ok"
