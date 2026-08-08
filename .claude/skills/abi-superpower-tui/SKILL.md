---
name: abi-superpower-tui
description: TUI/dashboard superpower. Launch interactive agent REPL, diagnostics dashboard panes, and slash commands via real abi CLI paths. Use when working on abi tui / dashboard / agent tui. There is no /abi-superpower-tui binary.
superpower:
  command: "execute"
  parameters:
    - name: "action"
      type: "string"
      enum: ["repl", "dashboard", "pane", "once", "json"]
      description: "TUI action"
    - name: "pane"
      type: "string"
      description: "Dashboard pane: system, plugins, storage, wdbx, scheduler, memory"
    - name: "compact"
      type: "boolean"
      description: "Render only selected pane"
---

# ABI Superpower: TUI

Exposes the interactive TUI and dashboard as a superpower. There is **no**
`/abi-superpower-tui` binary or slash command — map actions to the real CLI
paths below (or the `run-tui` / `dashboard-smoke` drivers).

## Actions

### repl
Launch the agent REPL (slash commands live here):
```bash
./target/debug/abi agent tui
```

### dashboard
Show the diagnostics dashboard (`abi tui` is an alias of `abi dashboard`):
```bash
./target/debug/abi dashboard --pane system
./target/debug/abi dashboard --pane plugins --once --json
./target/debug/abi tui --compact --pane scheduler
```

Interactive pty smoke (tmux driver):
```bash
.agents/skills/run-tui/tui.sh              # drives `abi dashboard`
.agents/skills/run-tui/tui.sh tui          # drives `abi tui`
```

Headless one-shot smoke:
```bash
.agents/skills/dashboard-smoke/dashboard.sh
```

### pane
Select the initial diagnostics pane with `--pane` (system, plugins, storage/wdbx,
scheduler, memory, or 1–5). In the interactive refresh loop, switch panes with
hotkeys / Tab — there is no separate `pane --focus` CLI.

```bash
./target/debug/abi dashboard --list-panes
./target/debug/abi dashboard --pane memory
```

## Slash Commands (in `abi agent tui` REPL)

- `/help` - Show the REPL command surface
- `/model <id>` - Select a known model or a bounded free-form model ID
- `/profile` - Report the adaptive profile/router state
- `/sea [on|off|status|toggle]` - Change session-local SEA preference only
- `/status` - Report session, model, SEA, and live/store boundaries
- `/context` - Report the bounded local context summary
- `/history` - Print in-session prompt history
- `/reset` - Clear in-session history and turn count
- `/features` - Print locally available feature disclosures
- `/clear` - Clear the terminal display
- `/quit` - Exit (`/exit` and `/q` are aliases)

The REPL does not currently implement `/open`, `/diff`, `/commit`, `/learn`,
`/save`, or `/load`. Evidence-augmented completion is the separate
`abi complete --learn` path.

## Implementation

Maps to:
- `crates/abi-cli/src/repl.rs` - REPL commands, session state, and line editor
- `crates/abi-cli/src/dashboard.rs` - Split-pane dashboard (`abi tui` / `abi dashboard`)
- `crates/abi-cli/src/terminal.rs` - Raw-mode guard and incremental key decoder
- `.agents/skills/run-tui/tui.sh` - Interactive pty driver
- `.agents/skills/dashboard-smoke/dashboard.sh` - Headless one-shot smoke

## Runtime Boundary

The Rust `abi-cli` build always includes these surfaces. The dashboard refresh
loop and raw editor require a Unix terminal; `--once`, `--json`, and redirected
REPL input remain deterministic non-interactive paths. Dashboard panes support
bounded keyboard navigation and primary-button SGR mouse selection; capture is
disabled by a drop guard before terminal modes are restored.
