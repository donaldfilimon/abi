---
name: git-commit-integration
description: Create reviewed conventional Git commits through the agent workflow. This is not an `abi agent tui` slash command.
---

# Git Commit Integration

Creates properly formatted git commits for the abi project.

## Usage

```bash
git commit -m "type(scope): description"
```

## Features

- Enforces Conventional Commits format
- Validates message format: `type(scope): description`
- Types: feat, fix, refactor, docs, chore, test, perf
- Auto-adds all staged changes

## Implementation

The agent stages reviewed paths explicitly and invokes Git directly. ABI's Rust
REPL does not execute Git.

## Skill Integration

There is no `/commit` command in `abi agent tui`; use this skill outside the ABI
binary after reviewing the exact staged diff.
