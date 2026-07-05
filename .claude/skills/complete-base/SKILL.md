---
name: complete-base
description: 'Build the abi CLI and drive the base completion path — `abi complete "<input>"` with no flags — routing to the local model, running the constitution audit, and recording the completion in WDBX. Use to smoke-test the core local-completion path after touching src/features/ai/ or the model catalog. Fully local: no --live (remote) and no --learn (SEA).'
---

# complete-base — drive abi's base local completion

Driver: **`.claude/skills/complete-base/complete.sh`** (paths relative to repo root).
Builds the CLI and drives `abi complete` on the base (local, non-learning) path.
Evidence is the `RESULT:` line. Fully local, no network.

## Run (agent path)
```bash
.claude/skills/complete-base/complete.sh                                  # default prompt, default model
.claude/skills/complete-base/complete.sh "summarize backends" fable-5     # custom prompt + model alias
```
- `complete "<prompt>"` (or `complete --model <id> "<prompt>"`) → asserts
  `model=`, `audit_passed=true`, `persisted=true`, `wdbx kv_entries=`.
- Model aliases are canonicalized (`fable-5` → `claude-fable-5`); an unrecognized
  id passes through with a one-line stderr warning.

Prints `RESULT: PASS` (exit 0) or a FAIL count.

Historical verification: **PASS** on Zig master `0.17.0-dev.1099` — base completion
routes to `model=claude-fable-5`, passes the constitution audit, and persists the
completion (query/response vectors + block) to WDBX.

## Gotchas
- ⚠️ **Base path is fully local.** No `--live` means no remote provider is
  contacted; the response comes from the local model. `--live` (anthropic
  provider) and on-device `apple-fm` (requires `--confirm`) are out of scope here.
- `--learn` routes through the SEA self-learning loop — that path is covered by
  the `sea-learn-loop` skill, not this one.
- `complete` **appends to your default WDBX store** (a completion block + query/
  response vectors), so `kv_entries`/`vectors`/`blocks` grow each run. This is the
  designed behavior (same as `run-abi`'s completion step), not a leak.
- For source-level reasoning about model routing + the catalog, read
  `src/features/ai/models.zig`; for the SEA path use the `sea-evidence-analyst` subagent.

## Troubleshooting
| Symptom | Fix |
|---|---|
| `build` FAIL | Check `zig version` (see `/zig-pin`), then `./build.sh check`. |
| `audit_passed=true` missing | Constitution/audit regression — check `src/features/ai/` (constitution + completion path). |
| unexpected `model=` | Alias/catalog drift — check `src/features/ai/models.zig`. |
