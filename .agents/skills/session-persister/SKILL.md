---
name: session-persister
description: Document the unavailable ABI session-save concept without claiming a live CLI or REPL command. Use when reviewing or planning session persistence.
---

# Session Persister

ABI does not currently persist named REPL sessions. This skill is planning
guidance only and must not claim that a session was saved.

## Usage

There is no `/save` command in `abi agent tui` and no equivalent top-level CLI
command.

## Proposed State

- Turn history (last 10 entries)
- Current model/profile selection
- Learning mode state
- File mentions in context
- Session metadata (timestamp, name)

No `ReplState` serializer or `~/.abi/sessions` contract is linked. Any future
implementation needs an explicit schema, path safety, bounded history, tests,
and an opt-in migration plan before this skill can advertise execution.

## Skill Integration

Pairs with `session-restorer` as a planning surface only.
