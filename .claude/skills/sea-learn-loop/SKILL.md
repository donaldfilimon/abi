---
name: sea-learn-loop
description: Build the abi CLI and exercise the SEA (Sparse Evidence Attention) self-learning completion path via `abi complete --learn`. Use when working on ai_learn / complete --learn / evidence recall, or to verify the SEA loop runs and persists. Has a --sea flag to build with -Dfeat-sea=true for real recall.
---

# sea-learn-loop — exercise the SEA self-learning completion

Driver: **`.claude/skills/sea-learn-loop/learn.sh`** (paths relative to repo root).
CLI check — evidence is the `RESULT:` line + the `learn=…` status line.

`feat-sea` is **OFF by default**, so the default run exercises the *degraded*
path: the `--learn` plumbing runs end-to-end but recall returns nothing
(`evidence_count=0 adapted=false`). That still proves the loop, persistence, and
WDBX wiring work. Use `--sea` to rebuild with `-Dfeat-sea=true` for real
evidence recall + router adaptation.

## Prerequisites
- Pinned/master Zig on PATH (see `/zig-newest-skills`). macOS builds via `./build.sh`.

## Run (agent path)
```bash
.claude/skills/sea-learn-loop/learn.sh                       # degraded path (feat-sea off)
.claude/skills/sea-learn-loop/learn.sh "my custom prompt"    # custom input
.claude/skills/sea-learn-loop/learn.sh --sea                 # build -Dfeat-sea=true, real recall
```
It builds the CLI, runs `abi complete --learn "<input>"`, and asserts the
markers `learn=true`, `model=`, `evidence_count=` (plus a soft `persisted=true`
check). Prints `RESULT: PASS — SEA learn loop ran.` (exit 0) or a FAIL count.

Historical verification (feat-sea off): **PASS** — `model=claude-fable-5 …
learn=true evidence_count=0 adapted=false`, store reported `kv_entries=2
vectors=4 blocks=2`, on Zig master `0.17.0-dev.1099`.

## Gotchas
- **`evidence_count=0` is expected with feat-sea off** — the degraded path stores
  the completion but does no recall. Don't read 0 as a failure; it's the
  documented off-state. Use `--sea` to see non-zero recall once the store has hits.
- The model line shows the catalog default (`claude-fable-5`); `--learn` is local
  (WDBX metadata), not a live API call — no credentials needed.
- First `--sea` build is a fresh feature-graph compile (slower); the default
  `./build.sh cli` is incremental.

## Troubleshooting
| Symptom | Fix |
|---|---|
| `build` FAIL (default) | `/zig-build-doctor` or `./build.sh check`. |
| `build` FAIL (`--sea`) | feat-sea graph issue — check `src/features/sea/{mod,stub}.zig` parity. |
| missing `learn=true` | `complete` grammar drifted — check `src/cli/` `complete` handler + `--learn`. |
