#!/usr/bin/env bash
# validate_test_counts.sh — Run tests and verify counts match project_baseline.env
#
# Usage: bash scripts/validate_test_counts.sh [--main-only | --feature-only]
#
# Reads expected counts from scripts/project_baseline.env, runs the test suite,
# parses the summary line, and fails if actual counts diverge from the baseline.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

if [[ ! -f scripts/project_baseline.env ]]; then
    echo "ERROR: scripts/project_baseline.env not found"
    exit 1
fi

# shellcheck disable=SC1091
source scripts/project_baseline.env

MODE="${1:-all}"
errors=0

parse_summary() {
    # Parse Zig test summary line:
    #   "run test 1270 pass, 5 skip (1275 total) 14s MaxRSS:1G"
    # or "+- run test 1270 pass, 5 skip (1275 total)"
    local output="$1"
    local pass skip total

    pass=$(echo "$output" | grep -oE '[0-9]+ pass' | head -1 | grep -oE '[0-9]+')
    skip=$(echo "$output" | grep -oE '[0-9]+ skip' | head -1 | grep -oE '[0-9]+')
    total=$(echo "$output" | grep -oE '\([0-9]+ total\)' | head -1 | grep -oE '[0-9]+')

    echo "${pass:-0} ${skip:-0} ${total:-0}"
}

validate_main() {
    echo "Running main tests..."
    local output
    output=$(zig build test --summary all 2>&1) || true

    local counts
    counts=$(parse_summary "$output")
    local actual_pass actual_skip actual_total
    actual_pass=$(echo "$counts" | cut -d' ' -f1)
    actual_skip=$(echo "$counts" | cut -d' ' -f2)
    actual_total=$(echo "$counts" | cut -d' ' -f3)

    echo "  Expected: ${ABI_TEST_MAIN_PASS} pass, ${ABI_TEST_MAIN_SKIP} skip (${ABI_TEST_MAIN_TOTAL} total)"
    echo "  Actual:   ${actual_pass} pass, ${actual_skip} skip (${actual_total} total)"

    if [[ "$actual_pass" -lt "$ABI_TEST_MAIN_PASS" ]]; then
        echo "  ERROR: Main test pass count regressed (${actual_pass} < ${ABI_TEST_MAIN_PASS})"
        errors=$((errors + 1))
    elif [[ "$actual_pass" -gt "$ABI_TEST_MAIN_PASS" ]]; then
        echo "  NOTICE: Main pass count increased — update project_baseline.env (${actual_pass} > ${ABI_TEST_MAIN_PASS})"
        errors=$((errors + 1))
    fi

    if [[ "$actual_total" != "$ABI_TEST_MAIN_TOTAL" ]]; then
        echo "  NOTICE: Main total changed (${actual_total} != ${ABI_TEST_MAIN_TOTAL}) — update project_baseline.env"
        errors=$((errors + 1))
    fi
}

validate_feature() {
    echo "Running feature tests..."
    local output
    output=$(zig build feature-tests --summary all 2>&1) || true

    local counts
    counts=$(parse_summary "$output")
    local actual_pass actual_total
    actual_pass=$(echo "$counts" | cut -d' ' -f1)
    actual_total=$(echo "$counts" | cut -d' ' -f3)

    echo "  Expected: ${ABI_TEST_FEATURE_PASS} pass (${ABI_TEST_FEATURE_TOTAL} total)"
    echo "  Actual:   ${actual_pass} pass (${actual_total} total)"

    if [[ "$actual_pass" -lt "$ABI_TEST_FEATURE_PASS" ]]; then
        echo "  ERROR: Feature test pass count regressed (${actual_pass} < ${ABI_TEST_FEATURE_PASS})"
        errors=$((errors + 1))
    elif [[ "$actual_pass" -gt "$ABI_TEST_FEATURE_PASS" ]]; then
        echo "  NOTICE: Feature pass count increased — update project_baseline.env (${actual_pass} > ${ABI_TEST_FEATURE_PASS})"
        errors=$((errors + 1))
    fi
}

case "$MODE" in
    --main-only)  validate_main ;;
    --feature-only) validate_feature ;;
    all)
        validate_main
        validate_feature
        ;;
    *)
        echo "Usage: $0 [--main-only | --feature-only]"
        exit 1
        ;;
esac

if [[ "$errors" -gt 0 ]]; then
    echo ""
    echo "FAILED: Test baseline validation found $errors issue(s)"
    echo "If counts increased, update scripts/project_baseline.env and run /baseline-sync"
    exit 1
fi

echo ""
echo "OK: Test counts match baseline"
