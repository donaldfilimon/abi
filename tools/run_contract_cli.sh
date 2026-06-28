#!/usr/bin/env bash
set -euo pipefail

# Keep the CLI contract hermetic: the default-ON durable WDBX store would
# otherwise persist into ~/.abi/wdbx during the run. Force in-memory.
export ABI_WDBX_PERSIST=0

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
require_substring "$complete_out" "model=claude-fable-5"
require_substring "$complete_out" "audit_passed="
require_substring "$complete_out" "persisted="
require_substring "$complete_out" "wdbx kv_entries="

backends_out="$("$ABI" backends 2>&1)"
require_substring "$backends_out" "GPU:"

plugins_out="$("$ABI" plugin list 2>&1)"
require_substring "$plugins_out" "Installed Plugins ("
require_substring "$plugins_out" "example-plugin"
require_substring "$plugins_out" "(mod.zig)"

auth_out="$("$ABI" auth status 2>&1)"
require_substring "$auth_out" "Authentication Status:"
require_substring "$auth_out" "OpenAI:"

scheduler_out="$("$ABI" scheduler status 2>&1)"
require_substring "$scheduler_out" "scheduler status"
require_substring "$scheduler_out" "source=cli-scheduler-status"
require_substring "$scheduler_out" "completed=1"

# `abi nn` is the miniature character-level demo trainer. Assert the honest
# help framing and a real inline training run (stable label substrings only,
# never float loss values).
nn_help_out="$("$ABI" help nn 2>&1)"
require_substring "$nn_help_out" "character-level demo trainer"
require_substring "$nn_help_out" "not a production/LLM/distributed trainer"

nn_train_out="$("$ABI" nn train "hello hello hello " 2>&1)"
require_substring "$nn_train_out" "nn train: initial_loss="
require_substring "$nn_train_out" "improved="

echo "run_contract_cli: ok"
