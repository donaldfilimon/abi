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
./zig-out/bin/abi agent tui
```

### dashboard
Show the diagnostics dashboard (`abi tui` is an alias of `abi dashboard`):
```bash
./zig-out/bin/abi dashboard --pane system
./zig-out/bin/abi dashboard --pane plugins --once --json
./zig-out/bin/abi tui --compact --pane scheduler
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
./zig-out/bin/abi dashboard --list-panes
./zig-out/bin/abi dashboard --pane memory
```

## Slash Commands (in `abi agent tui` REPL)

- `/open <path>` - Load file into context
- `/diff` - Git diff
- `/commit` - Git commit
- `/context` - Show context state
- `/features` - Feature flags
- `/learn` - Toggle SEA learning
- `/save <name>` - Save session
- `/load <name>` - Load session
- `/status` - Agent status
- `/reset` - Clear history

## Implementation

Maps to:
- `src/features/tui/repl.zig` - REPL with line editor (`abi agent tui`)
- `src/features/tui/dashboard.zig` - Split-pane dashboard (`abi tui` / `abi dashboard`)
- `src/features/tui/line_editor.zig` - CSI decode, cursor, history
- `.agents/skills/run-tui/tui.sh` - Interactive pty driver
- `.agents/skills/dashboard-smoke/dashboard.sh` - Headless one-shot smoke

## Feature Gate

Requires `feat-tui=true` (default).
