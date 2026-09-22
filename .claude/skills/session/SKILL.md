---
name: session
description: Document the unavailable ABI session save and restore concepts without claiming a live CLI or REPL command. Use when reviewing or planning session persistence or restoration.
---

# Session persistence (planning only)

Merged from the former `session-persister` and `session-restorer` skills on 2026-09-21.

## Session Persister

*Use when:* Document the unavailable ABI session-save concept without claiming a live CLI or REPL command. Use when reviewing or planning session persistence.


ABI does not currently persist named REPL sessions. This skill is planning
guidance only and must not claim that a session was saved.

### Usage

There is no `/save` command in `abi agent tui` and no equivalent top-level CLI
command.

### Proposed State

- Turn history (last 10 entries)
- Current model/profile selection
- Learning mode state
- File mentions in context
- Session metadata (timestamp, name)

No `ReplState` serializer or `~/.abi/sessions` contract is linked. Any future
implementation needs an explicit schema, path safety, bounded history, tests,
and an opt-in migration plan before this skill can advertise execution.

### Skill Integration

Pairs with the section below as a planning surface only.

## Session Restorer

*Use when:* Document the unavailable ABI session-restore concept without claiming a live CLI or REPL command. Use when reviewing or planning session persistence.


ABI does not currently restore named REPL sessions. This skill is planning
guidance only and must not claim that a session was loaded.

### Usage

There is no `/load` command in `abi agent tui` and no equivalent top-level CLI
command.

### Proposed State

- Turn history (up to 10 entries, clamped)
- Model/profile selection
- Learning mode state
- File mentions
- Session metadata

No session schema, deserializer, or `~/.abi/sessions` contract is linked. A
future implementation must validate versions, paths, and bounds before mutating
REPL state.

### Skill Integration

Pairs with the section above as a planning surface only.
