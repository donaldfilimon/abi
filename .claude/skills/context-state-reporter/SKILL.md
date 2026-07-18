---
name: context-state-reporter
description: Show current agent context state including history, file mentions, and session data. Maps to `/context` slash command in abi agent tui.
---

# Context State Reporter

Reports the current state of the agent REPL context.

## Usage

```
/context
```

## Output Includes

- Turn history (last 10 entries by default)
- Current file mentions in context
- Session name if saved
- Learning mode status
- Active model/profile

## Implementation

Reads `ReplState` from `src/features/tui/repl.zig`:
- `turn_history` ring buffer
- `file_mentions` cache
- `learn_mode` flag
- `current_model` selection

## Skill Integration

Direct mapping to `abi agent tui` REPL `/context` command.