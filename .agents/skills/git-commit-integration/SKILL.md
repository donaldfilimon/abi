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

## Shared checkout with another session's unpushed commits

Trigger: `git status --short --branch` shows `ahead N`, or the tree carries dirty
paths you did not create. On a checkout several agents share, `git switch -c`
from the local tip is the wrong move twice over: the new branch inherits those
unpushed commits and rides them into your PR, and switching rewrites the working
tree under whoever is mid-task there.

Branch from the remote in a throwaway worktree instead:

```bash
git fetch origin
git worktree add ../<repo>-wt-<topic> -b <prefix>/<topic> origin/main
# bring in ONLY your own change, then verify it is the whole diff
git diff --stat origin/main -- <your-path>
git log --oneline origin/main..HEAD      # must list only your commits
git add <your-path> && git commit && git push -u
```

After the merge: `git worktree remove ../<repo>-wt-<topic>`, `git branch -d`,
`git worktree prune`. Leave the shared checkout exactly as you found it — do not
amend, reword, rebase, or `reset` a commit you did not author, even when its
message is malformed.

Verified on `/Users/donald/sources/repos/abi`, whose local `main` was `ahead 1` with another
author's unpushed commit and 11 dirty paths from other sessions: the worktree
branch's diff against `origin/main` was exactly the one intended file
(`AGENTS.md`, +3/-1), the junk commit never entered the PR, and every other
session's path stayed dirty and unstaged.

## Skill Integration

There is no `/commit` command in `abi agent tui`; use this skill outside the ABI
binary after reviewing the exact staged diff.
