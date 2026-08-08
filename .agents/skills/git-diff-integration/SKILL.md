---
name: git-diff-integration
description: Review Git diffs through the agent workflow. This is not an `abi agent tui` slash command.
---

# Git Diff Integration

Shows git diff for context-aware agent operations.

## Usage

```bash
git diff [options]
```

## Options

- `--staged` - Show staged changes only
- `--name-only` - Show only filenames
- `<path>` - Limit to specific path

## Implementation

The agent invokes read-only Git directly. ABI's Rust REPL does not execute Git.

## Skill Integration

There is no `/diff` command in `abi agent tui`; use this skill or read-only Git
outside the ABI binary.
