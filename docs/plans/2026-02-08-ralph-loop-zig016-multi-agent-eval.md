# Ralph Loop Eval: Zig 0.16.0-dev.2535+b5bd49460 Multi-Agent Prompt Suite (2026-02-08)

## Scope
- Evaluated comprehension-oriented prompts derived from:
  - `docs/plan.md`
  - `prompts/zig-0.16-master-system-prompt.md`
  - `prompts/wdbx-refactor-megaprompt.md`
  - `prompts/wdbx-testing-megaprompt.md`
- Prompt count: 6 (`checkpoints/ralph-loop/2026-02-08-zig016-multi-agent-prompt-set.json`)
- Goal: validate the `ralph_loop.py` evaluation harness and scoring flow for this prompt suite.

## Run Configuration
- Runner: `$RALPH_LOOP_RUNNER` (default: `$HOME/.codex/skills/ralph-loop/scripts/ralph_loop.py`)
- Mode: `--dry-run` (no model API calls)
- Provider: `placeholder`
- Model label: `dryrun-zig016-multi-agent-v1`
- Timestamp (UTC): `2026-02-08T07:20:54Z`
- Run ID: `72e60ab8-f64f-42a4-b207-7fd421139600`
- Zig toolchain capture: `zig version` (record this in future live-run metadata)
- Parameters: `temperature=0.2`, `top_p=1.0`, `max_tokens=700`, `seed=20260208`
- Raw results: `checkpoints/ralph-loop/2026-02-08-zig016-multi-agent-raw-results.json`
- Scored results: `checkpoints/ralph-loop/2026-02-08-zig016-multi-agent-scored-results.json`

## Aggregate Metrics
| Metric | Value |
| --- | --- |
| Prompt count | 6 |
| Dry-run outputs | 6 |
| Accuracy (avg) | 1.0 |
| Completeness (avg) | 1.0 |
| Reasoning (avg) | 1.0 |
| Instruction Following (avg) | 1.0 |
| Safety/Policy (avg) | 5.0 |
| Overall (avg) | 1.8 |
| Pass threshold | 3.5 |
| Pass count | 0 |
| Pass rate | 0.0 |

Scoring method (deterministic): outputs marked as dry-run placeholders are assigned
`accuracy=1`, `completeness=1`, `reasoning=1`, `instruction=1`, `safety=5`, `overall=1.8`.

## Notable Failures / Examples
- `zig016-phases-002`
  - Output excerpt: `[DRY RUN] List Phases 0 through 3 from docs/plan.md...`
  - Failure: no phase/date/exit-criteria extraction; pure prompt echo.
- `zig016-workflow-004`
  - Output excerpt: `[DRY RUN] Compare the phase workflows in the master, refactor, and testing prompts...`
  - Failure: no cross-prompt comparison or artifact mapping; no role-difference analysis.
- `zig016-parity-003`
  - Output excerpt: `[DRY RUN] A feature team modified src/features/web/mod.zig and src/features/web/stub.zig...`
  - Failure: no verification command synthesis; cannot assess instruction-following quality in dry-run mode.

## Recommendation for Next Iteration
- Re-run the same prompt set against a real model backend (non-dry-run) to generate answer-bearing outputs.
- Keep this dry-run scoring as a harness sanity check only; do not use it for quality/readiness decisions.
- Add expected-answer anchors per prompt (required commands, phase dates, artifact names)
  to support stricter automatic scoring beyond placeholder detection.

## Next Run Procedure
For the live run on February 8, 2026, execute the following commands:

1. Export the API key:
```bash
export OPENAI_API_KEY="<your_openai_api_key>"
RALPH_LOOP_RUNNER="${RALPH_LOOP_RUNNER:-$HOME/.codex/skills/ralph-loop/scripts/ralph_loop.py}"
```
2. Capture Zig toolchain and set output paths (raw + scored):
```bash
zig version | tee checkpoints/ralph-loop/2026-02-08-zig016-multi-agent-live-zig-version.txt
RAW_OUT="checkpoints/ralph-loop/2026-02-08-zig016-multi-agent-live-raw-results.json"
SCORED_OUT="checkpoints/ralph-loop/2026-02-08-zig016-multi-agent-live-scored-results.json"
```
3. Run the existing prompt set with the OpenAI provider:
```bash
python3 "$RALPH_LOOP_RUNNER" \
  --prompts checkpoints/ralph-loop/2026-02-08-zig016-multi-agent-prompt-set.json \
  --out "$RAW_OUT" \
  --model gpt-4.1-mini \
  --provider openai \
  --temperature 0.2 \
  --top-p 1.0 \
  --max-tokens 700 \
  --seed 20260208
```
4. Apply the same scoring flow used for this report and write scored output to:
```bash
python3 checkpoints/ralph-loop/score_with_anchors.py \
  --raw "$RAW_OUT" \
  --anchors checkpoints/ralph-loop/2026-02-08-zig016-multi-agent-expected-anchors.json \
  --out "$SCORED_OUT" \
  --rubric-path "$HOME/.codex/skills/ralph-loop/references/eval-rubric.md"
```

## Acceptance Criteria for Live Run
- [ ] `overall_average >= 3.5` in `checkpoints/ralph-loop/2026-02-08-zig016-multi-agent-live-scored-results.json`.
- [ ] No empty `output` fields in `checkpoints/ralph-loop/2026-02-08-zig016-multi-agent-live-raw-results.json`.
- [ ] Command-parity evidence is present in outputs for command-oriented prompts (for example, `zig016-parity-003`).
