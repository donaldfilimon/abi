---
name: sea-learn-loop
description: Build the abi CLI and exercise the SEA (Sparse Evidence Attention) self-learning completion path via `abi complete --learn`. Use when working on ai_learn / complete --learn / evidence recall, or to verify the SEA loop runs and persists. SEA defaults on; --sea only forces -Dfeat-sea=true explicitly.
---

# sea-learn-loop — exercise the SEA self-learning completion

Driver: **`.claude/skills/sea-learn-loop/learn.sh`** (paths relative to repo root).
CLI check — evidence is the `RESULT:` line + the `learn=…` status line.

`feat-sea` defaults **ON** with the rest of the `-Dfeat-*` flags. The default
run exercises the real SEA path; `evidence_count=0 adapted=false` can still be
valid when the scratch store has no matching evidence. Use `--sea` only when you
want to force `-Dfeat-sea=true` explicitly while debugging feature flags.

## Prerequisites
- Pinned/master Zig on PATH (see `/zig-newest-skills`). macOS builds via `./build.sh`.

## Run (agent path)
```bash
.claude/skills/sea-learn-loop/learn.sh                       # default SEA-on path
.claude/skills/sea-learn-loop/learn.sh "my custom prompt"    # custom input
.claude/skills/sea-learn-loop/learn.sh --sea                 # explicitly pass -Dfeat-sea=true
```
It builds the CLI, runs `abi complete --learn "<input>"`, and asserts the
markers `learn=true`, `model=`, `evidence_count=` (plus a soft `persisted=true`
check). Prints `RESULT: PASS — SEA learn loop ran.` (exit 0) or a FAIL count.

Historical verification (old feat-sea-off run): **PASS** — `model=claude-fable-5 …
learn=true evidence_count=0 adapted=false`, store reported `kv_entries=2
vectors=4 blocks=2`, on Zig master `0.17.0-dev.1099`.

## Gotchas
- **`evidence_count=0` is not necessarily a failure** — with SEA on, the scratch
  store may simply have no matching evidence yet.
- The model line shows the catalog default (`claude-fable-5`); `--learn` is local
  (WDBX metadata), not a live API call — no credentials needed.
- First build can be a fresh feature-graph compile; later `./build.sh cli` runs
  are incremental.

## Troubleshooting
| Symptom | Fix |
|---|---|
| `build` FAIL (default) | `/zig-build-doctor` or `./build.sh check`. |
| `build` FAIL (`--sea`) | explicit feat-sea graph issue — check `src/features/sea/{mod,stub}.zig` parity. |
| missing `learn=true` | `complete` grammar drifted — check `src/cli/` `complete` handler + `--learn`. |
