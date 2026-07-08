---
name: cross-compile-check
description: Run the opt-in cross-target compile smoke for the abi CLI — compile + link for Linux/Windows/macOS cross targets via `zig build cross-smoke`. Use to verify portability after touching platform-sensitive code (OS abstractions, net/IO, build.zig), or before a release.
---

# cross-compile-check — verify abi cross-compiles

Driver: **`.agents/skills/cross-compile-check/cross.sh`** (paths relative to repo root).
Compile-check — evidence is the `RESULT:` line + per-target output. Opt-in and
**slow** (each target is a fresh cross compile); not part of `./build.sh check`.

## Run (agent path)
```bash
.agents/skills/cross-compile-check/cross.sh                      # default target set
.agents/skills/cross-compile-check/cross.sh aarch64-linux-gnu    # specific target(s)
```
Runs `zig build cross-smoke` (default set: `x86_64-linux-gnu`,
`x86_64-windows-gnu`, `aarch64-macos`) and asserts `all targets compiled +
linked`. Prints `RESULT: PASS` (exit 0) or a FAIL count naming the failing
target.

## Gotchas
- ⚠️ **Slow** — three full cross compiles of the feature graph. Run it
  deliberately, not in a tight loop.
- Build-time tools (`gen_plugin_registry`, `check_parity`) stay host-targeted; only
  the CLI is cross-compiled (`tools/cross_smoke.sh`).
- A target failure is a real portability finding (e.g. a platform-specific `std`
  or OS call) — read the output for the moved/unavailable symbol.

## Troubleshooting
| Symptom | Fix |
|---|---|
| one target fails | read its compiler error — usually a platform-gated API; check `src/foundation/` OS abstractions. |
| all fail immediately | wrong `zig` on PATH — `zig version` must be 0.17-dev. |
