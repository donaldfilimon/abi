---
name: wdbx-bench
description: Build the abi CLI and benchmark the WDBX vector store (in-process insert/search timing), optionally running the abi-wdbx unit-test suite. Use when asked to benchmark WDBX, measure insert/search latency, profile the vector store, or check benchmark output after a storage change.
---

# wdbx-bench — benchmark the WDBX vector store

Driver: **`.agents/skills/wdbx-bench/bench.sh`** (paths relative to repo root).
CLI/timing check — no GUI; evidence is the `RESULT:` line + the latency table.

These are **local, in-memory** measurements — not a published throughput claim
(the CLI says so itself). Numbers vary by machine and are noisy at low counts.

## Prerequisites
- Nightly Rust via `./tools/cargo.sh` (never bare Homebrew `cargo`).

## Run (agent path)
```bash
.agents/skills/wdbx-bench/bench.sh 50        # build CLI, run `abi wdbx benchmark 50`
.agents/skills/wdbx-bench/bench.sh 50 --suite  # also run `./tools/cargo.sh test -p abi-wdbx --lib`
```
It builds the CLI, runs `abi wdbx benchmark <count>`, and asserts the markers
`benchmark (local, in-memory`, `inserts:`, `searches:`. Prints
`RESULT: PASS — WDBX benchmark ran.` (exit 0) or `RESULT: FAIL — N check(s) failed.`

## Gotchas
- **Not a throughput claim.** Per-op time includes acceleration-kernel dispatch;
  the GPU path is the vectorized CPU fallback unless native kernels are linked
  (`abi backends` shows `accelerated=false` on this machine).
- Low counts are high-variance — use ≥50 for a stable-ish p50; the suite
  (`--suite`) runs `./tools/cargo.sh test -p abi-wdbx --lib`.
- `bench.sh <non-number>` → usage, exit 2.

## Troubleshooting
| Symptom | Fix |
|---|---|
| `build` FAIL | Run `./tools/check.sh` to see the real error. |
| missing `inserts:`/`searches:` marker | CLI grammar drifted — check `crates/abi-cli` WDBX handler + `crates/abi-wdbx`. |
