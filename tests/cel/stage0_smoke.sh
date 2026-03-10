#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TMP_FILE="$REPO_ROOT/.cel-stage0/format-smoke.cel"
trap 'rm -f "$TMP_FILE"' EXIT

cd "$REPO_ROOT"

./cel check >/dev/null
./cel run >/dev/null
./cel test >/dev/null
./cel emit-c >/dev/null
./cel check examples/cel/hello.cel >/dev/null
./cel test tests/cel/stage0_tests.cel >/dev/null

cp tests/cel/format_input.cel "$TMP_FILE"
./cel fmt -w "$TMP_FILE"
diff -u "$TMP_FILE" - >/dev/null <<'EOF'
module tests.cel.format_input;
import std.prelude;

fn main() {
    print("fmt");
    defer print("later");
}
EOF
