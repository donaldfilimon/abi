---
name: session-restorer
description: Load previously saved agent session. Maps to `/load <name>` slash command in abi agent tui.
---

# Session Restorer

Restores a previously saved REPL session from named slot.

## Usage

```
/load <name>
```

## Restored State

- Turn history (up to 10 entries, clamped)
- Model/profile selection
- Learning mode state
- File mentions
- Session metadata

## Implementation

Deserializes JSON from `~/.abi/sessions/<name>.json` into `ReplState`:
- Validates schema version
- Clamps history to `MAX_TURN_HISTORY` (10)
- Resets transient state (input buffer, etc.)

## Skill Integration

Maps to `abi agent tui` REPL `/load` command.
Pairs with `session-persister` for `/save`.