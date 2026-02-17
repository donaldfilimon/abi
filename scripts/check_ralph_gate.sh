#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

input_json="reports/ralph_upgrade_results_openai.json"
summary_md="reports/ralph_upgrade_summary.md"

if [[ ! -f "$input_json" ]]; then
    echo "ERROR: missing live Ralph results: $input_json"
    echo "Run:"
    echo "  OPENAI_API_KEY=... python3 $HOME/.codex/skills/ralph-loop/scripts/ralph_loop.py \\"
    echo "    --prompts scripts/ralph_prompts_upgrade.json \\"
    echo "    --out $input_json \\"
    echo "    --model gpt-5 --provider openai"
    exit 1
fi

python3 scripts/score_ralph_results.py \
    --in "$input_json" \
    --out "$summary_md" \
    --require-live \
    --min-average 0.75

echo "OK: Ralph gate passed ($summary_md)."
