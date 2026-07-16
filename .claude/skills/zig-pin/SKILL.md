---
name: zig-pin
description: Check that the active Zig toolchain matches the repo pin in .zigversion, and if not, print the exact zvm/zigup command to select it. Use before building when a compile fails with API errors that smell like a toolchain mismatch (e.g. std.Io.net.Stream, ArrayListUnmanaged), or any time you want to confirm the right Zig is on PATH.
disable-model-invocation: true
---

# zig-pin — toolchain pin guard

The repo pins Zig to a specific dev build in `.zigversion`. `./build.sh` and
`tools/build.sh` **do not switch or enforce the pin** — they invoke whatever
`zig` is first on `PATH` (just echoing `Using Zig: …`). A mismatched toolchain
fails in confusing ways: e.g. Zig `0.16.0` cannot compile
`src/features/wdbx/{net_line,rest}.zig` or `src/mcp/server.zig`, which use the
0.17 `std.Io.net.Stream.read(io, …)` API. The default `zvm` may also have
drifted to `master`, which is newer than the pin.

This skill compares the pin to the active toolchain and tells you how to fix a
mismatch. It does not change your toolchain for you.

## Driver

Run `.agents/skills/zig-pin/pin.sh` for a one-shot check of active vs. pinned
toolchain. Exit 0 = match, exit 2 = mismatch with fix instructions.

## Steps (manual)

1. **Read the pin.** Read `.zigversion` at the repo root (a single line, e.g.
   `0.17.0-dev.1398+cb5635714`). Call this `PIN`.

2. **Read the active version.** Run `zig version`. Call this `ACTIVE`.
   - If `zig` is not on `PATH`, stop and report that no Zig is installed; point
     the user at a version manager (zvm or zigup).

3. **Compare.**
   - If `ACTIVE` == `PIN` → report `✅ Zig matches the pin (PIN)`. Done.
   - If they differ → report the mismatch clearly:
     `⚠️ active <ACTIVE> ≠ pinned <PIN>` and proceed to step 4.

4. **Print the select command** for whichever manager is present (check with
   `command -v zvm` / `command -v zigup`):
   - **zvm:** `zvm use <PIN>`  (install first if absent: `zvm install <PIN>`)
   - **zigup:** `zigup <PIN>`  (or `zigup fetch <PIN>` then `zigup default <PIN>`)
   - If neither is installed, recommend installing one and note the exact pin
     string to fetch.

5. **Re-verify (optional).** If the user runs the select command, re-run
   `zig version` and confirm it now equals `PIN`.

## Notes

- This is a read/diagnose skill: it never edits `.zigversion` and never
  auto-runs the select command — toolchain switching is the user's call.
- The pin string is the **full** dev-build identifier including the
  `+<hash>` suffix; compare it exactly, since two `0.17.0-dev.*` builds with
  different revision numbers are not interchangeable.
- Reference: the toolchain section at the top of `AGENTS.md` / `CLAUDE.md` /
  `GEMINI.md`.
