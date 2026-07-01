---
name: dashboard-smoke
description: Build the abi CLI and non-interactively smoke the diagnostics dashboard — render `abi dashboard` one-shot (stdin from /dev/null forces the non-interactive fallback), assert exit 0 and all 5 panels (System, Plugins, WDBX Storage, Scheduler, Memory). Use to smoke-test `abi dashboard`/`abi tui`/`abi --tui` in CI or headless without hanging in the interactive TUI. Complements run-tui (which drives the interactive path via tmux).
---

# dashboard-smoke — non-interactive one-shot dashboard render

Driver: **`.claude/skills/dashboard-smoke/dashboard.sh`** (paths relative to repo root).
Builds the CLI, renders the diagnostics dashboard once with a non-TTY stdin, and
asserts exit 0 + all five panels. Evidence is the `RESULT:` line. This is the
headless/CI counterpart to **run-tui**, which drives the *interactive* dashboard
through a tmux pty.

## Run (agent path)
```bash
.claude/skills/dashboard-smoke/dashboard.sh
```
Prints `RESULT: PASS` (exit 0) or `RESULT: FAIL` with the missing panel/assertion.

One-liner by hand:
```bash
./zig-out/bin/abi dashboard < /dev/null 2>&1 | sed $'s/\033\[[0-9;]*m//g'
```

## Gotchas
- ⚠️ **Interactive by default → it HANGS on a TTY.** `abi dashboard`, `abi tui`,
  and `abi --tui` all route to `handleDashboard`, which runs a 1s auto-refresh
  loop (quit on `q`/`Esc`/EOF) when stdin is a terminal. To smoke it, make stdin
  a non-TTY: `< /dev/null` (or pipe input) forces the one-shot render and returns
  immediately.
- **All output is on stderr** — capture `2>&1`. `2>/dev/null` blanks the whole
  frame and makes it look like nothing rendered.
- ⚠️ **Benign `unexpected errno: 19` / `tcgetattr` stack trace on macOS.** With
  `< /dev/null`, `tcgetattr` returns errno 19 (ENODEV, not the ENOTTY the code
  anticipates), so a full stack trace prints *before* the frame. It looks like a
  crash but is not — **exit code is 0** and the dashboard renders right after.
  Assert the panels, not the absence of the trace. (Piping `echo | abi dashboard`
  may sidestep the trace.)
- **No `timeout` needed** (and macOS lacks it) — a non-TTY stdin makes it a clean
  one-shot. `timeout`/`gtimeout` only matters if you insist on a TTY guard.
- **The build is near-silent** — confirm with `ls zig-out/bin/abi`, not stdout.
- Healthy render = title `ABI Diagnostics Dashboard` + 5 panels (System / Plugins
  / WDBX Storage / Scheduler / Memory), exit 0. Fresh-process values are mostly
  zero (empty WDBX, a couple completed scheduler tasks) — that's normal.

## Troubleshooting
| Symptom | Fix |
|---|---|
| hangs / never returns | stdin is a TTY — redirect `< /dev/null` to force the one-shot. |
| empty output | You captured stdout only — the dashboard prints to stderr; use `2>&1`. |
| looks like a crash (errno 19) | Benign macOS fallback trace; exit is 0 and panels render — ignore it. |
| a panel missing | Inspect `src/cli/handlers/dashboard.zig`; use the `tui-navigation-guide` subagent for the render loop. |

Verified this session: **PASS** on Zig master `0.17.0-dev.1099` — one-shot render
of all 5 panels, exit 0, benign errno-19 trace tolerated.
