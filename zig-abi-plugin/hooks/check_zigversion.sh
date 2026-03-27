#!/bin/bash
# Warns when .zigversion is about to be edited.
# The zig version pin must be updated atomically across three files:
#   .zigversion, build.zig.zon, and .github/workflows/ci.yml
set -euo pipefail

input=$(cat)
file_path=$(echo "$input" | jq -r '.tool_input.file_path // empty')

if [[ "$file_path" == *".zigversion" ]]; then
    cat <<'JSON'
{
  "systemMessage": "NOTE: .zigversion is being modified. Remember to update the version pin atomically across all three locations:\n  1. .zigversion (this file)\n  2. build.zig.zon (version field)\n  3. .github/workflows/ci.yml (zig version in CI)\nAfter updating, run: zigly --install"
}
JSON
fi

exit 0
