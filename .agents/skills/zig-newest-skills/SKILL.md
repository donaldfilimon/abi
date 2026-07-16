---
name: zig-newest-skills
description: "This skill should be used when the user asks to \"test abi on Zig master\", \"build on newest Zig\", \"upgrade to Zig nightly\", \"check forward compatibility with Zig master\", \"validate toolchain drift past .zigversion\", or run zig-master-check / zvm master against the abi tree. Switches the active toolchain to current Zig master via zvm, builds CLI + MCP, and proves the tree still compiles and runs against bleeding-edge std."
---

# zig-newest-skills

Validate **abi** against the **current Zig master** nightly. The repo pin lives
in `.zigversion` (read that file for the live value; do not hardcode pins in
new prose). CI uses the same pin (`.github/workflows/ci.yml` `ZIG_VERSION`).
This skill is the inverse of pin-safety: surface forward `std` API drift the
pin would hide.

**Canonical package:** `.agents/skills/zig-newest-skills/`  
**Claude mirror:** `.claude/skills/zig-newest-skills/` (keep in sync)  
**Driver:** `zig-master-check.sh` (repo-root relative; resolves path from `$0`)

## When to run

| Trigger | Action |
|---------|--------|
| Forward-compat check | Run driver (default) |
| Fresh master tarball | Driver `--update` |
| Deep CLI/MCP exercise | Driver `--smoke` |
| Restore pin after master | Driver `--revert` |
| Build failure only on master | Fix `src/` for moved `std` APIs, or revert pin |

Do **not** use this skill to silently bump `.zigversion` / CI without an
explicit pin-upgrade task and green `./build.sh check` on the new pin.

## Prerequisites

- `zvm` on `PATH` (`zvm --version`; install: https://github.com/tristanisham/zvm)
- macOS or Linux; on macOS prefer `./build.sh` for Metal-linked CLI builds
- Network only when installing/updating master (`zvm install master`)

## Agent path (run first)

From the repo root:

```bash
.agents/skills/zig-newest-skills/zig-master-check.sh
```

Gate order (cheapest first):

1. `zvm use master` (install master if missing)
2. `zig build check-parity` — std-only host tool; isolates toolchain from feature graph
3. `./build.sh cli` → `./build.sh mcp`
4. Run `zig-out/bin/abi help` and `abi backends`

Success ends with:

```text
RESULT: PASS — abi builds + runs on the newest Zig master.
```

(exit 0). Failure: `RESULT: FAIL — master drift broke N gate(s).` (exit N).

### Flags

```bash
.agents/skills/zig-newest-skills/zig-master-check.sh --update
.agents/skills/zig-newest-skills/zig-master-check.sh --smoke
.agents/skills/zig-newest-skills/zig-master-check.sh --revert
```

| Flag | Behavior |
|------|----------|
| `--update` | `zvm install master` before select |
| `--smoke` | After binary gates, run `run-abi/smoke.sh` (CLI + WDBX + MCP stdio) |
| `--revert` | Select `.zigversion` pin and exit (non-interactive install attempt if missing) |

## After a PASS

1. Record master version (`zig version`) vs pin in the report.
2. **Default:** leave master selected for the rest of this session; run
   `--revert` before any pin-gated day-to-day work (or when the user asks).
3. If the user wants a pin bump: update `.zigversion`, CI `ZIG_VERSION`, and
   instruction-file pin strings together; run full `./build.sh check` on the pin.

## After a FAIL

1. Capture the first failing gate and the `std` symbol / compile error.
2. Prefer a minimal `src/` fix over disabling features.
3. Re-run the driver; if unblockable, `--revert` and file a focused fix task.
4. Consult **`references/src-touchpoints.md`** for high-churn Zig 0.17 surfaces.

## Troubleshooting (quick)

| Symptom | Fix |
|---------|-----|
| `zvm not found` | Install zvm; put `~/.zvm/bin` on `PATH` |
| `--revert` cannot fetch pin | Old nightlies not re-served; stay on master or bump pin |
| Re-download master every run | Do not grep ANSI `zvm list`; use on-disk `~/.zvm/<ver>/zig` |
| Fail only under master | Real drift — fix `src/` or pin back |
| `--smoke` skipped | Ensure `.agents/skills/run-abi/smoke.sh` is executable |

## Additional resources

- **`references/gotchas.md`** — zvm/nightly battle scars (ANSI list, non-re-downloadable pins, no macOS `timeout`)
- **`references/src-touchpoints.md`** — abi `src/` areas sensitive to Zig 0.17 / `std` churn
- **Driver:** `zig-master-check.sh` — primary executable; prefer over ad-hoc `zvm` + build sequences

## Claims boundary

- Does **not** claim production multi-host, native ANE, or audited FHE.
- Master PASS is **forward-compat evidence**, not a pin replacement, until
  `.zigversion` + CI are updated deliberately.
- Full `./build.sh check` on master is optional and slower; the driver covers
  the minimum compile+run surface used for drift detection.
