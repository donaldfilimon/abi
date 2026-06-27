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
  feat-hash
  feat-metrics
  feat-telemetry
  feat-sea
)

for flag in "${FLAGS[@]}"; do
  echo "check_feature_stubs: zig build cli -D${flag}=false"
  zig build cli "-D${flag}=false"
done

# Feature-off paths have behavioral contracts, not just compile contracts.
# Keep this focused on feature module APIs so unrelated filesystem/network tests are not
# re-run once per flag under full-check parallelism.
for flag in "${FLAGS[@]}"; do
  echo "check_feature_stubs: zig build test-feature-contracts -D${flag}=false"
  zig build test-feature-contracts "-D${flag}=false"
done

# Public contracts are feature-aware too; run them under every disabled flag so
# MCP, CLI surface, and generated registry expectations stay honest.
for flag in "${FLAGS[@]}"; do
  echo "check_feature_stubs: zig build test-contracts -D${flag}=false"
  zig build test-contracts "-D${flag}=false"
done

# Mobile defaults to disabled; smoke the real mobile module too.
echo "check_feature_stubs: zig build test-feature-contracts -Dfeat-mobile=true"
zig build test-feature-contracts -Dfeat-mobile=true
echo "check_feature_stubs: zig build test-contracts -Dfeat-mobile=true"
zig build test-contracts -Dfeat-mobile=true

echo "check_feature_stubs: ok"
