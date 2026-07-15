---
name: git-diff-integration
description: Show git diff of working tree or staged changes. Maps to `/diff` slash command in abi agent tui.
---

# Git Diff Integration

Shows git diff for context-aware agent operations.

## Usage

```
/diff [options]
```

## Options

- `--staged` - Show staged changes only
- `--name-only` - Show only filenames
- `<path>` - Limit to specific path

## Implementation

Runs `git diff` via std.process and formats output for agent consumption.

## Skill Integration

Used in `abi agent tui` REPL for `/diff` command to show current changes before committing.