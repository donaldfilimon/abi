---
name: wdbx-bench
description: Build the abi CLI and benchmark the WDBX vector store (in-process insert/search timing), optionally running the full `zig build benchmarks` suite. Use when asked to benchmark WDBX, measure insert/search latency, profile the vector store, or check benchmark output after a storage change.
---

# wdbx-bench — benchmark the WDBX vector store

Driver: **`.claude/skills/wdbx-bench/bench.sh`** (paths relative to repo root).
CLI/timing check — no GUI; evidence is the `RESULT:` line + the latency table.

These are **local, in-memory** measurements — not a published throughput claim
(the CLI says so itself). Numbers vary by machine and are noisy at low counts.

## Prerequisites
- Pinned/master Zig on PATH (see `/zig-newest-skills`). macOS builds via `./build.sh`.

## Run (agent path)
```bash
.claude/skills/wdbx-bench/bench.sh 50        # build CLI, run `abi wdbx benchmark 50`
.claude/skills/wdbx-bench/bench.sh 50 --suite  # also run `zig build benchmarks`
```
It builds the CLI, runs `abi wdbx benchmark <count>`, and asserts the markers
`benchmark (local, in-memory`, `inserts:`, `searches:`. Prints
`RESULT: PASS — WDBX benchmark ran.` (exit 0) or `RESULT: FAIL — N check(s) failed.`

Verified this session: **PASS** — e.g. `inserts: 50 …` / `searches: 50 …` with
p50/p95/p99 lines, on Zig master `0.17.0-dev.1099`.

## Gotchas
- **Not a throughput claim.** Per-op time includes acceleration-kernel dispatch;
  the GPU path is the vectorized CPU fallback unless native kernels are linked
  (`abi backends` shows `accelerated=false` on this machine).
- Low counts are high-variance — use ≥50 for a stable-ish p50; the suite
  (`--suite`) runs the broader `zig build benchmarks` targets.
- `bench.sh <non-number>` → usage, exit 2.

## Troubleshooting
| Symptom | Fix |
|---|---|
| `build` FAIL | Run `/zig-build-doctor` or `./build.sh check` to see the real error. |
| missing `inserts:`/`searches:` marker | CLI grammar drifted — check `src/cli/handlers/wdbx.zig` `benchmark`. |
