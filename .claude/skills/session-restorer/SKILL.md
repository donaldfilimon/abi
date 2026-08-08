---
name: session-restorer
description: Document the unavailable ABI session-restore concept without claiming a live CLI or REPL command. Use when reviewing or planning session persistence.
---

# Session Restorer

ABI does not currently restore named REPL sessions. This skill is planning
guidance only and must not claim that a session was loaded.

## Usage

There is no `/load` command in `abi agent tui` and no equivalent top-level CLI
command.

## Proposed State

- Turn history (up to 10 entries, clamped)
- Model/profile selection
- Learning mode state
- File mentions
- Session metadata

No session schema, deserializer, or `~/.abi/sessions` contract is linked. A
future implementation must validate versions, paths, and bounds before mutating
REPL state.

## Skill Integration

Pairs with `session-persister` as a planning surface only.
