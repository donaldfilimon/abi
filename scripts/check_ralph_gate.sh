#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

input_json="reports/ralph_upgrade_results_openai.json"

if [[ ! -f "$input_json" ]]; then
    echo "ERROR: missing live Ralph results: $input_json"
    echo "Run:"
    echo "  abi ralph super --task 'upgrade analysis' --gate"
    exit 1
fi

# Validate JSON structure and extract average score using jq (or bash fallback).
if command -v jq &>/dev/null; then
    avg=$(jq -r '
        [.results[]?.score // empty] |
        if length == 0 then "none"
        else (add / length)
        end
    ' "$input_json" 2>/dev/null || echo "none")

    if [[ "$avg" == "none" ]]; then
        echo "FAILED: Ralph report present but contains no scored results"
        exit 1
    fi

    # Compare with threshold (0.75) using awk for float comparison.
    pass=$(awk "BEGIN { print ($avg >= 0.75) }")
    if [[ "$pass" == "1" ]]; then
        echo "OK: Ralph gate passed (average score: $avg)"
    else
        echo "FAILED: Ralph gate score $avg < 0.75 threshold"
        exit 1
    fi
else
    echo "ERROR: jq is required to evaluate Ralph gate score"
    exit 1
fi
