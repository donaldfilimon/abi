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
`abi wdbx gpu info`; asserts the markers `Compute Backends:` and
`compute backends`. Prints `RESULT: PASS` (exit 0) or a FAIL count.

Current Rust driver: captures all three backend reports and requires the
compute/GPU section markers. Capability rows remain distinct from execution.

## Gotchas
- The backend is **runtime-selected** — there is no `-Dgpu-backend` option.
- A compiled or available backend is not automatically initialized or executed.
  Read the reported state fields rather than inferring acceleration from the OS.
- For deeper analysis use the `gpu-backend-analyzer` subagent.

## Troubleshooting
| Symptom | Fix |
|---|---|
| `build` FAIL | `./tools/check.sh`. |
| missing `Compute Backends:` | CLI grammar drift — check `crates/abi-cli/src/backends.rs`. |
