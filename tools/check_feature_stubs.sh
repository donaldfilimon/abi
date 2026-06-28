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

# Default-off pure-Zig features (sea, metrics, mobile) also need an ENABLED pass:
# their real modules — and SEA's evidence/learn_loop E2E tests, reached via the
# main test root's refAllDecls only when feat-sea=true — are otherwise never
# exercised by the gate (the disabled matrix above only proves the stub trees
# build). The main `test` step is required because the feature contract suites do
# not run the modules' own inline tests. (feat-foundationmodels is macOS + Swift
# -shim gated, so it is intentionally omitted here.)
ENABLED="-Dfeat-sea=true -Dfeat-metrics=true -Dfeat-mobile=true"
echo "check_feature_stubs: zig build test ${ENABLED}"
zig build test ${ENABLED}
echo "check_feature_stubs: zig build test-feature-contracts ${ENABLED}"
zig build test-feature-contracts ${ENABLED}
echo "check_feature_stubs: zig build test-contracts ${ENABLED}"
zig build test-contracts ${ENABLED}

echo "check_feature_stubs: ok"
