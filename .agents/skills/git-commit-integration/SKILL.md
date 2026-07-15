---
name: git-commit-integration
description: Create git commit with conventional commit format. Maps to `/commit` slash command in abi agent tui.
---

# Git Commit Integration

Creates properly formatted git commits for the abi project.

## Usage

```
/commit [message]
```

## Features

- Enforces Conventional Commits format
- Validates message format: `type(scope): description`
- Types: feat, fix, refactor, docs, chore, test, perf
- Auto-adds all staged changes

## Implementation

Uses `git commit` with validation of commit message format.

## Skill Integration

Maps to `abi agent tui` REPL `/commit` command for seamless workflow.