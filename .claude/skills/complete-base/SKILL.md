---
name: complete-base
description: 'Build the abi CLI and drive the base completion path — `abi complete "<input>"` with no flags — routing to the local model, running the constitution audit, and optionally recording the completion in WDBX. Use to smoke-test the core local-completion path after touching crates/abi-ai or the model catalog. Fully local: no --live (remote) and no --learn (SEA).'
---

# complete-base — drive abi's base local completion

Driver: **`.agents/skills/complete-base/complete.sh`** (paths relative to repo root).
Builds the CLI and drives `abi complete` on the base (local, non-learning) path.
Evidence is the `RESULT:` line. Fully local, no network.

## Run (agent path)
```bash
.agents/skills/complete-base/complete.sh                                  # default prompt, default model
.agents/skills/complete-base/complete.sh "summarize backends" fable-5     # custom prompt + model alias
```
- `complete "<prompt>"` (or `complete --model <id> "<prompt>"`) → asserts
  `model=`, `audit_passed=true`, `wdbx kv_entries=`, and a `persisted=` line
  (`true` when a durable store path is available, else honest `false`).
- Model aliases are canonicalized (`fable-5` → `claude-fable-5`); an unrecognized
  id passes through with a one-line stderr warning.

Prints `RESULT: PASS` (exit 0) or a FAIL count.

## Gotchas
- ⚠️ **Base path is fully local.** No `--live` means no remote provider is
  contacted; the response comes from the local model. `--live` (anthropic
  provider) and on-device `apple-fm` (requires `--confirm`) are out of scope here.
- `--learn` routes through the SEA self-learning loop — that path is covered by
  the `sea-learn-loop` skill, not this one.
- When a durable WDBX path is configured, `complete` may append to that store.
  Without one, the CLI reports `persisted=false` and `wdbx_status=no persistent
  WDBX path configured` — both are success cases for this smoke.
- For routing/catalog source, see `crates/abi-ai`; SEA path is `sea-learn-loop`.

## Troubleshooting
| Symptom | Fix |
|---|---|
| `build` FAIL | Use `./tools/cargo.sh build -p abi-cli`, then `./tools/check.sh`. |
| `audit_passed=true` missing | Constitution/audit regression — check `crates/abi-ai`. |
| unexpected `model=` | Alias/catalog drift — check `crates/abi-ai` models. |
