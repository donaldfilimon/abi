# Zig 0.16 Multi-Agent Ralph-Loop Rerun Pack

## Purpose
- `2026-02-08-zig016-multi-agent-prompt-set.json`: canonical prompt inputs.
- `2026-02-08-zig016-multi-agent-raw-results.json`: raw runner outputs.
- `2026-02-08-zig016-multi-agent-scored-results.json`: scored envelope (`scoring_metadata`, `overall`, `results`).
- `2026-02-08-zig016-multi-agent-expected-anchors.json`: deterministic expected anchors per prompt id.

## Dry Run Command (Current)
```sh
RALPH_LOOP_RUNNER="${RALPH_LOOP_RUNNER:-$HOME/.codex/skills/ralph-loop/scripts/ralph_loop.py}"
python3 "$RALPH_LOOP_RUNNER" \
  --prompts checkpoints/ralph-loop/2026-02-08-zig016-multi-agent-prompt-set.json \
  --out checkpoints/ralph-loop/2026-02-08-zig016-multi-agent-raw-results.json \
  --model dryrun-zig016-multi-agent-v1 \
  --provider placeholder \
  --temperature 0.2 \
  --max-tokens 700 \
  --top-p 1.0 \
  --seed 20260208 \
  --dry-run
```

## Live Run Command Template (OpenAI)
```sh
export OPENAI_API_KEY="<your-api-key>"
RALPH_LOOP_RUNNER="${RALPH_LOOP_RUNNER:-$HOME/.codex/skills/ralph-loop/scripts/ralph_loop.py}"
zig version | tee checkpoints/ralph-loop/2026-02-08-zig016-multi-agent-live-zig-version.txt
python3 "$RALPH_LOOP_RUNNER" \
  --prompts checkpoints/ralph-loop/2026-02-08-zig016-multi-agent-prompt-set.json \
  --out checkpoints/ralph-loop/2026-02-08-zig016-multi-agent-raw-results.json \
  --model <openai-model> \
  --provider openai \
  --temperature 0.2 \
  --max-tokens 700 \
  --top-p 1.0 \
  --seed 20260208
```

## Post-Run Scoring Workflow
1. Score raw results into `checkpoints/ralph-loop/2026-02-08-zig016-multi-agent-scored-results.json`.
```sh
python3 checkpoints/ralph-loop/score_with_anchors.py \
  --raw checkpoints/ralph-loop/2026-02-08-zig016-multi-agent-raw-results.json \
  --anchors checkpoints/ralph-loop/2026-02-08-zig016-multi-agent-expected-anchors.json \
  --out checkpoints/ralph-loop/2026-02-08-zig016-multi-agent-scored-results.json \
  --rubric-path "$HOME/.codex/skills/ralph-loop/references/eval-rubric.md"
```
2. Keep scored top-level schema aligned with this pack's contract:
   - top-level keys: `scoring_metadata`, `overall`, `results`
   - per-result row fields: `~/.codex/skills/ralph-loop/references/results-schema.md`
3. Validate scored structure:
```sh
jq -e 'has("scoring_metadata") and has("overall") and has("results") and (.results | type == "array")' \
  checkpoints/ralph-loop/2026-02-08-zig016-multi-agent-scored-results.json
```
