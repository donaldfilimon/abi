---
name: zig-newest-skills
description: Switch the abi repo to the newest Zig master nightly (via zvm), build the CLI + MCP binaries, and verify the tree still compiles and runs against bleeding-edge std. Use when asked to test/build abi on the latest Zig, upgrade to Zig master/nightly, check forward compatibility with newest Zig, or validate toolchain drift past the .zigversion pin.
---

# zig-newest-skills — validate abi against the newest Zig master

The abi repo pins a specific Zig nightly in `.zigversion`
(currently `0.17.0-dev.1398+cb5635714` — read the file for the live value).
This skill does the opposite of pin-safety:
it switches the active toolchain to the **current Zig master** via `zvm`,
rebuilds the real binaries, and proves abi still compiles and runs against
bleeding-edge `std` — surfacing forward API drift the pin would otherwise hide.

Driver: **`.agents/skills/zig-newest-skills/zig-master-check.sh`** (paths below
are relative to the repo root). It's a CLI/toolchain check — no GUI, no
screenshots; evidence is exit codes + the `RESULT:` line.

## Prerequisites

- `zvm` on `PATH` (https://github.com/tristanisham/zvm). Verify: `zvm --version`.
- macOS or Linux. On macOS, builds go through `./build.sh` (Metal link workflow).

## Run (agent path) — FIRST

```bash
# From the repo root. Selects master (installs it if absent), then runs all gates.
.agents/skills/zig-newest-skills/zig-master-check.sh
```

What it does, cheapest gate first: `zvm use master` → `zig build check-parity`
(std-only host gate) → `./build.sh cli` → `./build.sh mcp` → runs the built
`abi help` and `abi backends`. Prints a summary ending in either
`RESULT: PASS — abi builds + runs on the newest Zig master.` (exit 0) or
`RESULT: FAIL — master drift broke N gate(s).` (exit N).

Historical verification on master `0.17.0-dev.1275` (pin was `1252` at the time):
**PASS, 0 failed gates.**

Flags:

```bash
.agents/skills/zig-newest-skills/zig-master-check.sh --update   # re-fetch latest master nightly first
.agents/skills/zig-newest-skills/zig-master-check.sh --smoke    # also run the full run-abi smoke (CLI + WDBX + MCP stdio)
.agents/skills/zig-newest-skills/zig-master-check.sh --revert   # switch back to the .zigversion pin, then exit
```

`--smoke` chains the sibling `run-abi/smoke.sh` harness (13 checks: CLI
subcommands, a WDBX round-trip, and JSON-RPC over the MCP stdio transport).
Historical verification: `pass=13 fail=0 / SMOKE OK`.

## Run (human path)

Same as above — just run the script. There is no separate window or REPL;
it's a build+exercise gate you read the tail of.

## Gotchas (battle scars from this session)

- **Old nightlies are NOT re-downloadable.** zvm only serves the *current*
  master from ziglang.org. e.g. `zvm install 0.17.0-dev.1252+e4b325c19` (a
  since-superseded pin) fails with
  `unsupported Zig version`. So once master replaces the pin's local
  `~/.zvm/<version>` dir, the exact pin is **gone for good** unless you kept
  the tarball. Installing master can therefore permanently retire the
  `.zigversion` pin — `--revert` will then fail with a clear explanation, not
  silently. (If you need reproducibility, bump `.zigversion` to the master you
  validated and treat that as the new pin.)
- **`zvm list` output is ANSI-colored** (`\x1b[32mmaster`). Greping it with a
  word boundary (`grep '\bmaster\b'`) silently fails — the `m` in `[32m`
  abuts `master`. The driver checks `~/.zvm/<ver>/zig` on disk instead; do the
  same in any wrapper, or you'll re-download ~54MB every run.
- **`build.sh` does not enforce the pin.** It just runs whatever `zig` is on
  `PATH` (echoes `Using Zig: …`). So "select master via zvm" is the *only*
  thing that makes the build use master — there's no in-repo guard to fight.
- **`check-parity` builds even when the feature graph wouldn't.** It's a
  std-only, host-target line scanner. Running it first cleanly separates
  "toolchain works" from "feature graph compiles under this std."
- **`timeout(1)` is absent on macOS.** Don't wrap gates in it; the driver
  relies on the caller's timeout instead.
- **As of this session, abi has zero forward drift 1252→1275** — no source
  changes were needed to build on master. If a future master breaks a gate,
  the failing `./build.sh cli|mcp` output names the exact `std` API that moved.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `zvm not found on PATH` | Install zvm; ensure `~/.zvm/bin` is on `PATH`. |
| `--revert` → `unsupported Zig version` / `cannot revert to <pin>` | The pin nightly is no longer fetchable. Stay on master (`zvm use master`) or bump `.zigversion`. See Gotchas. |
| Driver re-downloads master every run | You're on an old copy that greps `zvm list`; use the on-disk `have_ver` check (already in this driver). |
| `./build.sh cli` fails only under master | Real forward drift — read the error for the moved `std` symbol; fix source or pin back. |
