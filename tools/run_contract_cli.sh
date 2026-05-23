#!/usr/bin/env bash
set -euo pipefail

ABI="${ABI_EXE:-zig-out/bin/abi}"

if [[ ! -x "$ABI" ]]; then
  echo "error: $ABI not found; run 'zig build cli' first" >&2
  exit 1
fi

require_substring() {
  local output="$1"
  local needle="$2"
  if ! grep -Fq "$needle" <<<"$output"; then
    echo "expected substring missing: $needle" >&2
    echo "output:" >&2
    echo "$output" >&2
    exit 1
  fi
}

complete_out="$("$ABI" complete "hello world" 2>&1)"
require_substring "$complete_out" "model=abi-local"
require_substring "$complete_out" "audit_passed="

backends_out="$("$ABI" backends 2>&1)"
require_substring "$backends_out" "GPU:"

plugins_out="$("$ABI" plugin list 2>&1)"
require_substring "$plugins_out" "example-plugin"

auth_out="$("$ABI" auth status 2>&1)"
require_substring "$auth_out" "Authentication Status:"
require_substring "$auth_out" "OpenAI:"

echo "run_contract_cli: ok"
