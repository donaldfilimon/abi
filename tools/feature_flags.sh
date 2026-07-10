#!/usr/bin/env bash
# Feature flags for stub/contract gates — names must match build.zig -Dfeat-* options.
# feat-foundationmodels is intentionally excluded from the disabled matrix loop;
# check_feature_stubs.sh runs a dedicated FM-off CLI build after the matrix.

abi_read_disabled_feature_flags() {
  local build_zig="${1:-build.zig}"
  if [[ ! -f "$build_zig" ]]; then
    echo "error: missing $build_zig" >&2
    return 1
  fi
  grep -E 'b\.option\(bool, "feat-' "$build_zig" \
    | sed -E 's/.*b\.option\(bool, "(feat-[^"]+)".*/\1/' \
    | grep -v '^feat-foundationmodels$' \
    | sort -u
}