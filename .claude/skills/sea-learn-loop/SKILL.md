---
name: sea-learn-loop
description: Build the abi CLI and exercise the SEA (Sparse Evidence Attention) self-learning completion path via `abi complete --learn`. Use when working on ai_learn / complete --learn / evidence recall, or to verify the SEA loop runs and persists. SEA is always linked in the Rust workspace; --sea is a compatibility no-op.
---

# sea-learn-loop — exercise the SEA self-learning completion

Driver: **`.agents/skills/sea-learn-loop/learn.sh`** (paths relative to repo root).
CLI check — evidence is the `RESULT:` line + the `learn=…` status line.

SEA is always available in the Rust workspace. The default run exercises the
real SEA path; `evidence_count=0 adapted=false` can still be valid when the
store has no matching evidence. `--sea` is accepted for compatibility and does
not change the build path.

## Prerequisites
- Nightly Rust via `./tools/cargo.sh` (never bare Homebrew `cargo`).

## Run (agent path)
```bash
.agents/skills/sea-learn-loop/learn.sh                       # default SEA-on path
.agents/skills/sea-learn-loop/learn.sh "my custom prompt"    # custom input
.agents/skills/sea-learn-loop/learn.sh --sea                 # accepted; no-op on build path
```
It builds the CLI, points `ABI_WDBX_PATH` at a scratch store under
`target/skill-scratch/` (never the live `~/.abi/`), runs
`abi complete --learn "<input>"`, and asserts the markers `learn=true`,
`model=`, `evidence_count=` (plus a soft `persisted=true` check). Prints
`RESULT: PASS — SEA learn loop ran.` (exit 0) or a FAIL count.

## Gotchas
- **`evidence_count=0` is not necessarily a failure** — with SEA on, the scratch
  store may simply have no matching evidence yet.
- The model line shows the catalog default (`claude-fable-5`); `--learn` is local
  (WDBX metadata), not a live API call — no credentials needed.
- First build may compile the workspace; later `./tools/cargo.sh build -p abi-cli` runs
  are incremental.

## Troubleshooting
| Symptom | Fix |
|---|---|
| `build` FAIL | `./tools/cargo.sh build -p abi-cli`, then `./tools/check.sh`. |
| missing `learn=true` | `complete` grammar drifted — check `crates/abi-cli` complete handler + `--learn`. |
