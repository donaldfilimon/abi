#!/bin/bash
# Reminds about test/mod.zig wiring when a new integration test file is created.
# Triggers on Write to test/integration/*_test.zig files.
set -euo pipefail

input=$(cat)
tool_name=$(echo "$input" | jq -r '.tool_name')
file_path=$(echo "$input" | jq -r '.tool_input.file_path // empty')

# Only trigger on Write (new file creation) for integration tests
if [[ "$tool_name" != "Write" ]]; then exit 0; fi
if [[ "$file_path" != */test/integration/*_test.zig ]]; then exit 0; fi

# Extract just the filename
basename=$(basename "$file_path")

cat <<JSON
{
  "systemMessage": "New integration test file '${basename}' created. Remember to:\n1. Import it from test/mod.zig: const ${basename%_test.zig}_tests = @import(\"integration/${basename}\");\n2. If adding a focused test lane, create test/${basename%.zig}.zig and wire in build/validation.zig\n3. Run: zig build test --summary all"
}
JSON

exit 0
