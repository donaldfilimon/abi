---
name: abi-superpower-tui
description: TUI/dashboard superpower. Launch interactive REPL, dashboard panes, split views, and slash commands.
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

Exposes the interactive TUI and dashboard as a superpower.

## Actions

### repl
Launch agent REPL with slash commands:
```
/abi-superpower-tui repl
```

### dashboard
Show diagnostics dashboard:
```
/abi-superpower-tui dashboard --pane system
/abi-superpower-tui dashboard --pane plugins --once --json
```

### pane
Switch focus between split panes (Tab key in interactive):
```
/abi-superpower-tui pane --focus agent_output
```

## Slash Commands (in REPL)

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
- `src/features/tui/repl.zig` - REPL with line editor
- `src/features/tui/dashboard.zig` - Split-pane dashboard
- `src/features/tui/line_editor.zig` - CSI decode, cursor, history

## Feature Gate

Requires `feat-tui=true` (default).