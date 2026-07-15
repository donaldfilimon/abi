---
name: file-context-loader
description: Load file content into agent context via @file mentions. Uses the same resolution logic as abi agent tui and agent plan/multi commands.
---

# File Context Loader

Loads files into the agent context for analysis, planning, or completion tasks.

## Usage

```
/open <path>
```

## Implementation

Uses `src/features/ai/file_context.zig`:
- `resolveFileMentions()` - resolves @file mentions in input
- `validateMentionPath()` - sandboxed to cwd, rejects .. / absolute / symlink escape
- `ContextBudget` - 8KB default budget per resolution

## Context

- Reads file relative to current working directory
- Injects content with `file:` prefix for model consumption
- Budget enforcement prevents context overflow

## Skill Integration

Maps to existing abi functionality:
- `abi agent plan <input>` - uses resolveAndInject
- `abi agent multi <input>` - uses resolveAndInject  
- `abi agent tui` - REPL uses resolveFileMentions