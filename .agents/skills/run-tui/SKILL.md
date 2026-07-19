---
name: run-tui
description: Build the abi CLI and drive the interactive diagnostics dashboard (`abi dashboard` / `abi tui` / `abi --tui`) through a tmux pty — launch it, screenshot/capture the rendered pane, assert it painted, send the quit key, tear down. Use to run, launch, screenshot, or smoke-test the abi TUI/dashboard headlessly. The one abi surface that needs a real terminal.
---

# run-tui — drive abi's interactive diagnostics dashboard

Driver: **`.agents/skills/run-tui/tui.sh`** (paths relative to repo root).
`abi dashboard`/`tui`/`--tui` are interactive on a real TTY; this driver gives
them a **tmux pty**, captures the rendered pane, asserts the dashboard painted,
sends `q`, and kills the session. Evidence is the `RESULT:` line. Fully local.

## Prerequisites
- **`tmux`** (`brew install tmux`). The driver checks for it and fails fast if absent.

## Run (agent path)
```bash
.agents/skills/run-tui/tui.sh              # drives `abi dashboard` (default)
.agents/skills/run-tui/tui.sh tui          # drive `abi tui` instead
```
Launches the command under `tmux new-session` (200x50 pane), waits for paint,
captures the pane, and asserts the `ABI Diagnostics Dashboard` marker is present
and there's **no** `errno 19`/`tcgetattr`/panic. Sends `q` (the quit key —
`isQuitKey` accepts `q`/`Q`/Esc) and confirms the session tore down. Prints
`RESULT: PASS` (exit 0) or a FAIL count.

To eyeball it yourself: `tmux capture-pane -pt <session>` shows the System pane
(GPU backend, accelerated, native-linked) and the Plugins pane (16 registered).

Historical verification: **PASS** on the pin in `.zigversion` — dashboard box +
System + Plugins panes render under the pty; `q` quits; session cleaned up.
Do not hardcode a Zig nightly in this skill; read `.zigversion` for the live pin.


## Gotchas (battle scars)
- ⚠️ **Do NOT prepend `/opt/homebrew/bin` to PATH.** Homebrew ships a `zig`
  (`/opt/homebrew/bin/zig -> 0.16.0`) that **cannot compile this tree**. Putting
  brew's bin first shadows the zvm 0.17 zig and the build fails with std API
  errors. The driver *appends* brew's bin (for `tmux`) so the zvm zig stays
  first — keep it that way.
- **A pty is mandatory for the interactive path.** Piped stdin or `/dev/null`
  intentionally uses the one-shot fallback; use tmux when you need to prove the
  live loop paints and accepts `q`.
- **Give it time to paint.** The driver sleeps 2.5s before capture; a busy
  machine may need more. A blank pane = captured too early, not a failure to run.
- `Accelerated: yes` / `Native Linked: yes` in the System pane is the dashboard's
  own rendering of backend state; the CLI `backends` report phrases the same
  Metal-linked/CPU-fallback status differently — both are correct.
- For the render loop and screen lifecycle internals, use the
  `tui-navigation-guide` subagent.

## Troubleshooting
| Symptom | Fix |
|---|---|
| `tmux not installed` | `brew install tmux`. |
| `build` FAIL right after adding brew to PATH | brew's `zig` 0.16 shadowed zvm — append, don't prepend (see Gotchas). |
| `dashboard did not paint` | Increase the `sleep`, or confirm the pane size (`-x/-y`) is large enough. |
| `tty error / panic in pane` | Regression in the interactive terminal path; confirm launch went through `tmux new-session`. |
