---
name: file-context-loader
description: Load file content into agent context via @file mentions. Uses the same resolution logic as abi agent tui and agent plan/multi commands.
---

# File Context Loader

Loads files into the agent context for analysis, planning, or completion tasks.

## Usage

```bash
./target/debug/abi agent plan "review @relative/path.rs"
```

## Implementation

Uses `crates/abi-ai/src/file_context.rs`:
- `resolve_file_mentions()` - resolves `@file` mentions in input
- `validate_mention_path()` - sandboxed to cwd, rejects `..`, absolute paths, and symlink escape
- `ContextBudget` - 8KB default budget per resolution

## Context

- Reads file relative to current working directory
- Injects content with `file:` prefix for model consumption
- Budget enforcement prevents context overflow

## Skill Integration

Maps to existing abi functionality:
- `abi agent plan <input>` - injects bounded mentioned files
- `abi agent multi <input>` - injects bounded mentioned files
- `abi agent tui` free-text prompts - use the same context builder

There is no `/open` command in the Rust REPL.
