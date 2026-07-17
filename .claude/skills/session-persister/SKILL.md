---
name: session-persister
description: Save current agent session to named slot. Maps to `/save <name>` slash command in abi agent tui.
---

# Session Persister

Saves the current REPL session state to a named slot for later restoration.

## Usage

```
/save <name>
```

## Saved State

- Turn history (last 10 entries)
- Current model/profile selection
- Learning mode state
- File mentions in context
- Session metadata (timestamp, name)

## Implementation

Serializes `ReplState` to JSON at `~/.abi/sessions/<name>.json`:
- Clamped on load (max 10 turns, 8KB context budget)
- Overwrites existing slot with same name

## Skill Integration

Maps to `abi agent tui` REPL `/save` command.
Pairs with `session-restorer` for `/load`.