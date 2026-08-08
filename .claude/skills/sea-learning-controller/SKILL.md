---
name: sea-learning-controller
description: Run or inspect SEA (Sparse Evidence Attention) learning through the real `abi complete --learn` path. The agent REPL has only a session-local `/sea` preference and no `/learn` command.
---

# SEA Learning Controller

Routes SEA learning to the real CLI and keeps the REPL boundary explicit.

## Usage

```bash
SEA_STORE=$(mktemp -d)
ABI_WDBX_PATH="$SEA_STORE" ./target/debug/abi complete --learn "review evidence"
```

## States

- `abi complete --learn` opens the configured durable store, recalls evidence,
  and updates the adaptive modulator.
- `abi complete` without `--learn` uses the base completion route.
- `/sea on|off|status|toggle` in `abi agent tui` is session-local metadata only;
  it reports `live services=off` and never opens WDBX.

## Implementation

Maps to `crates/abi-cli/src/complete.rs` and `crates/abi-sea/src/learn_loop.rs`.
The durable path persists modulator weights under `modulator:weights`; tests and
smokes must use a scratch store and never the user's live `~/.abi` path.

## Skill Integration

There is no `/learn` REPL command. Use `abi complete --learn` for actual evidence
recall; use `/sea` only to inspect or change the session-local REPL preference.
