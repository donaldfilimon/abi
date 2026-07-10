#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=feature_flags.sh
source "$ROOT/feature_flags.sh"

# Disabled-matrix flags are read from build.zig (feat-foundationmodels excluded; see below).
FLAGS=()
while IFS= read -r flag; do
  [[ -n "$flag" ]] && FLAGS+=("$flag")
done < <(abi_read_disabled_feature_flags "$ROOT/../build.zig")

if [[ ${#FLAGS[@]} -eq 0 ]]; then
  echo "error: no feat-* flags read from build.zig" >&2
  exit 1
fi

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

# These pure-Zig feature implementations also get an explicit enabled pass:
# SEA's evidence/learn_loop E2E tests, reached via the main test root's
# refAllDecls only when feat-sea=true, need coverage beyond the disabled matrix
# that proves only stub trees build. The main `test` step is required because the
# feature contract suites do not run the modules' own inline tests.
# (feat-foundationmodels is macOS + Swift-shim gated, so it is intentionally
# omitted here.)
ENABLED="-Dfeat-sea=true -Dfeat-metrics=true -Dfeat-mobile=true"
echo "check_feature_stubs: zig build test ${ENABLED}"
zig build test ${ENABLED}
echo "check_feature_stubs: zig build test-feature-contracts ${ENABLED}"
zig build test-feature-contracts ${ENABLED}
echo "check_feature_stubs: zig build test-contracts ${ENABLED}"
zig build test-contracts ${ENABLED}

# feat-foundationmodels now defaults ON, so the default gate above only ever
# compiles the real connector path. Force one FM-off CLI build so the
# feature-disabled connector path (fm_enabled=false: empty fm_fns, FMUnavailable)
# still gets compiled by the gate. This needs no Swift toolchain (the shim is only
# built when the flag is on), so it is safe on every host.
echo "check_feature_stubs: zig build cli -Dfeat-foundationmodels=false"
zig build cli -Dfeat-foundationmodels=false

echo "check_feature_stubs: ok"
