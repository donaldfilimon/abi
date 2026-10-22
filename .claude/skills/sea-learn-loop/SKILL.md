---
name: sea-learn-loop
description: Build the abi CLI and exercise the SEA (Sparse Evidence Attention) self-learning completion path via `abi complete --learn`. Use when working on ai_learn / complete --learn / evidence recall, or to verify the SEA loop runs and persists. SEA is always linked in the Rust workspace; --sea is a compatibility no-op.
---

# sea-learn-loop — exercise the SEA self-learning completion

Driver: **`.agents/skills/sea-learn-loop/learn.sh`** (paths relative to repo root).
CLI check — evidence is the `RESULT:` line and both passes' metadata lines.

SEA is always available in the Rust workspace. The default run exercises the
real SEA path. The first pass starts with no evidence; the second must recall
the first pass's persisted record after reopening the scratch store. Both must
report successful persistence and adaptation. `--sea` is accepted for
compatibility and does not change the build path.

## Prerequisites
- Pinned stable Rust 1.99.1 via `./tools/cargo.sh` (never bare Homebrew `cargo`).

## Run (agent path)
```bash
.agents/skills/sea-learn-loop/learn.sh                       # default SEA-on path
.agents/skills/sea-learn-loop/learn.sh "my custom prompt"    # custom input
.agents/skills/sea-learn-loop/learn.sh --sea                 # accepted; no-op on build path
```
It builds the locked CLI into the same target directory it executes, creates a
unique scratch store under `target/skill-scratch/` (never the live `~/.abi/`),
overrides inherited persistence settings and runs
`abi complete --learn -- "<input>"` in two separate processes. It requires
`learn=true`, local generation provenance, `persisted=true`, `adapted=true`,
and nonzero second-pass `evidence_count` on the metadata line, not inside the
generated response. Its own scratch directory is removed on exit; other runs'
stores are untouched. Prints
`RESULT: PASS — SEA learning persisted, adapted and recalled across processes.`
(exit 0) or `RESULT: FAIL` (nonzero). Audit-vetoed prompts intentionally fail
this acceptance smoke because they must not learn.

## Gotchas
- **First-pass `evidence_count=0` is expected** in a new scratch store, but a
  zero second-pass count fails the cross-process recall check.
- The requested-model line shows the catalog default (`claude-fable-5`) while
  `provider=local` and `generation_engine=persona-template` establish that
  `--learn` is not a live API call; no credentials are needed.
- Every metadata field rides that one first line, including
  `learn=true evidence_count=N adapted=B self_learn_cycle=... recall_ready=...`.
  `self_learn_cycle` reports what actually happened
  (`persistence+adaptation+recall` only when the learning write committed,
  `adaptation-only` when it was skipped or disabled, `adaptation-only+write-unknown`
  when the outcome is unknown) — never a completed cycle that the adjacent
  `persisted=` and `wdbx_status=` fields contradict. Do not move the learn
  fields onto their own line: the smoke reads only the first line, which is
  the boundary that stops markers repeated in generated response text from
  satisfying a persistence check.
- First build may compile the workspace; later `./tools/cargo.sh build -p abi-cli` runs
  are incremental.
- `--learn` is local persona-template learning, not live-provider or bridge
  inference. Combining it with `--live` or a local-bridge model is rejected;
  this does not train language-model weights.

## Troubleshooting
| Symptom | Fix |
|---|---|
| `build` FAIL | `./tools/cargo.sh build -p abi-cli`, then `./tools/check.sh`. |
| missing `learn=true` | `complete` grammar drifted — check `crates/abi-cli` complete handler + `--learn`. |
