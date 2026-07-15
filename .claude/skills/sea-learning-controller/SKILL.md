---
name: sea-learning-controller
description: Toggle SEA (Sparse Evidence Attention) self-learning mode. Maps to `/learn` slash command in abi agent tui.
---

# SEA Learning Controller

Toggles the SEA adaptive learning loop on/off for the current session.

## Usage

```
/learn [on|off|toggle]
```

## States

- `on` - Enable SEA adaptive completions with evidence recall
- `off` - Use base completion without learning
- `toggle` - Switch current state

## Implementation

Controls `ReplConfig.learn_mode` in `src/features/tui/repl.zig`:
- When on: `complete --learn --stream` uses `completeWithStoreAdaptive`
- Persists `AdaptiveModulator` weights (EMA, alpha=0.3) to WDBX key `modulator:weights`
- 8-signal scorer with task-aware weighting (7 task types)

## Skill Integration

Maps to `abi agent tui` REPL `/learn` command and `abi complete --learn` CLI flag.