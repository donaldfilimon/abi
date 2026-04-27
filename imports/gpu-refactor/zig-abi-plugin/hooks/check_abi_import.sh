#!/bin/bash
# Blocks @import("abi") from being written inside src/ (not test/).
# This causes a circular "no module named 'abi'" error at compile time.
#
# Only runs on Edit/Write tool calls targeting .zig files under src/.
set -euo pipefail

input=$(cat)
tool_name=$(echo "$input" | jq -r '.tool_name')
file_path=$(echo "$input" | jq -r '.tool_input.file_path // empty')

# Skip non-.zig files
if [[ "$file_path" != *.zig ]]; then exit 0; fi

# Skip files outside src/, or inside test/ (test/ is allowed to use @import("abi"))
if [[ "$file_path" != */src/* ]] || [[ "$file_path" == */test/* ]]; then exit 0; fi

# Extract content being written
content=""
if [[ "$tool_name" == "Write" ]]; then
    content=$(echo "$input" | jq -r '.tool_input.content // empty')
elif [[ "$tool_name" == "Edit" ]]; then
    content=$(echo "$input" | jq -r '.tool_input.new_string // empty')
fi

# Check for the circular import pattern
if echo "$content" | grep -qF '@import("abi")'; then
    cat >&2 <<'JSON'
{
  "hookSpecificOutput": { "permissionDecision": "deny" },
  "systemMessage": "BLOCKED: @import(\"abi\") inside src/ causes a circular import error (\"no module named 'abi' available within module 'abi'\"). Use relative imports instead, e.g. @import(\"../../foundation/mod.zig\"). For cross-feature imports, use the comptime gate: if (build_options.feat_X) @import(\"../X/mod.zig\") else @import(\"../X/stub.zig\")."
}
JSON
    exit 2
fi

exit 0
