#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DOC_PATH="$PROJECT_ROOT/docs/reports/cross_platform_testing.md"
TEST_DIR="$PROJECT_ROOT/tests/cross-platform"
MARKER_BEGIN="<!-- BEGIN: cross-platform-test-catalog -->"
MARKER_END="<!-- END: cross-platform-test-catalog -->"

MODE="update"
if [[ ${1-} == "--check" ]]; then
    MODE="check"
    shift
fi

if [ ! -d "$TEST_DIR" ]; then
    echo "cross-platform test directory not found: $TEST_DIR" >&2
    exit 1
fi

UPDATED_CONTENT=$(python3 - "$PROJECT_ROOT" "$DOC_PATH" "$MARKER_BEGIN" "$MARKER_END" <<'PY'
import pathlib
import re
import sys

root = pathlib.Path(sys.argv[1])
doc_path = pathlib.Path(sys.argv[2])
marker_begin = sys.argv[3]
marker_end = sys.argv[4]

if not doc_path.exists():
    raise SystemExit(f"documentation file not found: {doc_path}")

text = doc_path.read_text()
try:
    start = text.index(marker_begin)
    end = text.index(marker_end, start)
except ValueError:
    raise SystemExit("expected catalog markers not found in documentation")

test_dir = root / "tests" / "cross-platform"
entries = []
pattern = re.compile(r'^test\s+"([^"]+)"')
for path in sorted(test_dir.glob("*.zig")):
    relative = path.relative_to(root)
    raw_name = path.stem.replace('_', ' ').replace('-', ' ')
    lowered = raw_name.lower()
    if lowered == 'macos':
        title = 'macOS'
    else:
        title = raw_name.title()
    tests = []
    for line in path.read_text().splitlines():
        match = pattern.match(line.strip())
        if match:
            tests.append(match.group(1))
    entries.append((title, relative.as_posix(), tests))

lines = []
for title, relative, tests in entries:
    lines.append(f"### {title} (`{relative}`)")
    if tests:
        for test in tests:
            lines.append(f"- {test}")
    else:
        lines.append("- _No tests declared_")
    lines.append("")

catalog = "\n".join(lines).strip()
replacement = f"{marker_begin}\n\n{catalog}\n{marker_end}"

updated = text[:start] + replacement + text[end + len(marker_end):]
if not updated.endswith("\n"):
    updated += "\n"
sys.stdout.write(updated)
PY
)

if [[ $UPDATED_CONTENT != *$'\n' ]]; then
    UPDATED_CONTENT+=$'\n'
fi

if [[ "$MODE" == "check" ]]; then
    if ! diff -u --label "$DOC_PATH (expected)" --label "$DOC_PATH (actual)" <(printf '%s' "$UPDATED_CONTENT") "$DOC_PATH" >/dev/null; then
        echo "cross-platform test catalog is stale" >&2
        exit 1
    fi
    echo "Cross-platform test catalog is up to date."
else
    printf '%s' "$UPDATED_CONTENT" > "$DOC_PATH"
    echo "Updated cross-platform test catalog in $DOC_PATH"
fi
