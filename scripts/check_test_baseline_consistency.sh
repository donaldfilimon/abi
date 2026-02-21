#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

if [[ ! -f scripts/project_baseline.env ]]; then
    echo "ERROR: scripts/project_baseline.env not found"
    exit 1
fi

# shellcheck disable=SC1091
source scripts/project_baseline.env

errors=0

check_line() {
    local file="$1"
    local pattern="$2"
    local label="$3"
    if ! grep -Eq "$pattern" "$file"; then
        echo "ERROR: $file missing expected baseline marker for $label"
        errors=$((errors + 1))
    fi
}

# Main docs/rules we keep synchronized with the current verified baseline.
check_line "README.md" "tests-${ABI_TEST_MAIN_PASS}_passing" "README badge main pass"
check_line "README.md" "${ABI_TEST_MAIN_TOTAL} tests \\(${ABI_TEST_MAIN_PASS} passing, ${ABI_TEST_MAIN_SKIP} skip\\)" "README narrative baseline"

check_line "CLAUDE.md" "${ABI_TEST_MAIN_PASS} pass, ${ABI_TEST_MAIN_SKIP} skip \\(${ABI_TEST_MAIN_TOTAL} total\\)" "CLAUDE main baseline"
check_line "CLAUDE.md" "${ABI_TEST_FEATURE_PASS} pass \\(${ABI_TEST_FEATURE_TOTAL} total\\)" "CLAUDE feature baseline"

check_line ".claude/rules/zig.md" "${ABI_TEST_MAIN_PASS} pass, ${ABI_TEST_MAIN_SKIP} skip \\(${ABI_TEST_MAIN_TOTAL} total\\)" "zig.md main baseline"
check_line ".claude/rules/zig.md" "${ABI_TEST_FEATURE_PASS} pass \\(${ABI_TEST_FEATURE_TOTAL} total\\)" "zig.md feature baseline"

# Guard against known stale baselines.
declare -a stale_markers=(
    "1220/1225"
    "1220 pass"
    "1153 pass"
    "1198 pass"
    "1251 pass"
    "1257 pass"
    "1262 total"
    "1213 pass"
    "671 pass"
    "1252 pass"
    "1512 pass"
    "1534 pass"
)

for file in README.md CLAUDE.md .claude/rules/zig.md; do
    for marker in "${stale_markers[@]}"; do
        if grep -q "$marker" "$file"; then
            echo "ERROR: $file contains stale baseline marker '$marker'"
            errors=$((errors + 1))
        fi
    done
done

if [[ "$errors" -gt 0 ]]; then
    echo "FAILED: Test baseline consistency check found $errors issue(s)"
    exit 1
fi

echo "OK: Test baseline consistency checks passed"
