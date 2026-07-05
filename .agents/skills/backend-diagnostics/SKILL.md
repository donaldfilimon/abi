---
name: backend-diagnostics
description: Build the abi CLI and report GPU / accelerator / shader / MLIR backend status and the compute-backend matrix (CPU/GPU/NPU/TPU detection + dynamic selection). Use when asked about backend capabilities, why a backend shows accelerated=false, or to capture a hardware/dispatch report.
---

# backend-diagnostics — capture abi's backend capability report

Driver: **`.agents/skills/backend-diagnostics/diag.sh`** (paths relative to repo root).
Read-only CLI capture — evidence is the `RESULT:` line + the backend tables.

## Run (agent path)
```bash
.agents/skills/backend-diagnostics/diag.sh
```
Builds the CLI, then captures `abi backends`, `abi wdbx compute info`, and
`abi wdbx gpu info`; asserts the markers `GPU backend report` and
`compute backends`. Prints `RESULT: PASS` (exit 0) or a FAIL count.

Verified this session: **PASS** on Zig master `0.17.0-dev.1099` — Metal linked,
`accelerated=false` (vectorized CPU fallback active; native dispatch not linked).

## Gotchas
- The backend is **runtime-selected** — there is no `-Dgpu-backend` option.
- `accelerated=false` / `native=false` is the normal local state: frameworks are
  linked but native kernels aren't dispatched until initialized. Not a failure.
- For deeper analysis use the `gpu-backend-analyzer` subagent.

## Troubleshooting
| Symptom | Fix |
|---|---|
| `build` FAIL | `/zig-build-doctor` or `./build.sh check`. |
| missing `GPU backend report` | CLI grammar drift — check `src/cli/handlers/backends.zig`. |
