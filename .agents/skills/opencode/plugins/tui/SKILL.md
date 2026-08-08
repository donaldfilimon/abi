---
name: tui
description: TUI/dashboard OpenCode plugin. Launch interactive agent REPL, diagnostics dashboard panes, and slash commands via real abi CLI paths. There is no /abi-superpower-tui binary.
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

# TUI Superpower Plugin

Core TUI capabilities for OpenCode within the ABI framework. There is **no**
`/abi-superpower-tui` binary or slash command — map actions to the real CLI
paths below (or the `run-tui` / `dashboard-smoke` drivers).

## Capabilities

- TUI subsystem integration (dashboard, agent REPL, line editor)
- Plugin framework registration
- Runtime lifecycle management
- Configuration and settings management
- Status monitoring and reporting

## Integration Points

- ABI's TUI subsystem (`abi tui` / `abi dashboard` / `abi agent tui`)
- OpenCode plugin framework integration
- Runtime lifecycle management
- Configuration and settings management

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
- `/model <id>` - Select a model
- `/profile` - Report adaptive profile state
- `/sea [on|off|status|toggle]` - Change session-local preference only
- `/status`, `/context`, `/history`, `/reset`, `/features`, `/clear`, `/quit`

The REPL does not implement `/open`, `/diff`, `/commit`, `/learn`, `/save`, or
`/load`. Actual evidence recall is the separate `abi complete --learn` path.

## Implementation

Maps to:
- `crates/abi-cli/src/repl.rs` - REPL commands and session state
- `crates/abi-cli/src/repl_editor.rs` - bounded line editor and raw input loop
- `crates/abi-cli/src/dashboard.rs` - Split-pane dashboard (`abi tui` / `abi dashboard`)
- `crates/abi-cli/src/terminal.rs` - raw-mode guard and key decoder
- `.agents/skills/run-tui/tui.sh` - Interactive pty driver
- `.agents/skills/dashboard-smoke/dashboard.sh` - Headless one-shot smoke

## Runtime Boundary

The Rust CLI always includes these surfaces. Raw mode requires a Unix TTY;
redirected input and dashboard `--once`/`--json` use deterministic fallbacks.
Dashboard panes support bounded keyboard navigation and primary-button SGR
mouse selection, with capture cleanup on every exit path.
