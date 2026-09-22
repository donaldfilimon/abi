---
name: context-state
description: Show or reset abi agent context: report history, file mentions and session data (`/context` in abi agent tui), or clear history, file mentions and transient state (`/reset`).
---

# Agent context state

Merged from the former `context-state-reporter` and `context-resetter` skills on 2026-09-21.

## Context State Reporter

*Use when:* Show current agent context state including history, file mentions, and session data. Maps to `/context` slash command in abi agent tui.


Reports the current state of the agent REPL context.

### Usage

```
/context
```

### Output Includes

- Turn history (last 10 entries by default)
- Current file mentions in context
- Session name if saved
- Learning mode status
- Active model/profile

### Implementation

Reads `ReplState` from `crates/abi-cli/src/terminal.rs`:
- `turn_history` ring buffer
- `file_mentions` cache
- `learn_mode` flag
- `current_model` selection

### Skill Integration

Direct mapping to `abi agent tui` REPL `/context` command.

## Context Resetter

*Use when:* Reset agent context - clear history, file mentions, and transient state. Maps to `/reset` slash command in abi agent tui.


Clears all transient agent context state for a fresh start.

### Usage

```
/reset
```

### Cleared State

- Turn history (all entries)
- File mentions cache
- Input buffer
- Streaming state

### Preserved State

- Model/profile selection
- Learning mode setting
- Session name (if loaded)
- Feature flags

### Implementation

Calls `clearTurnHistory()` and resets `ReplState` fields in `crates/abi-cli/src/terminal.rs`:
- `turn_history` = empty ring buffer
- `file_mentions` = empty
- `input_buffer` = empty

### Skill Integration

Maps to `abi agent tui` REPL `/reset` command.
