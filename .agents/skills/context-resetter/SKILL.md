---
name: context-resetter
description: Reset agent context - clear history, file mentions, and transient state. Maps to `/reset` slash command in abi agent tui.
---

# Context Resetter

Clears all transient agent context state for a fresh start.

## Usage

```
/reset
```

## Cleared State

- Turn history (all entries)
- File mentions cache
- Input buffer
- Streaming state

## Preserved State

- Model/profile selection
- Learning mode setting
- Session name (if loaded)
- Feature flags

## Implementation

Calls `clearTurnHistory()` and resets `ReplState` fields in `src/features/tui/repl.zig`:
- `turn_history` = empty ring buffer
- `file_mentions` = empty
- `input_buffer` = empty

## Skill Integration

Maps to `abi agent tui` REPL `/reset` command.