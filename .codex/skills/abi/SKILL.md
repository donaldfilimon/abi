---
name: abi
description: Use this skill when working in the ABI Framework repository to apply required planning, consensus, verification, and close-out workflow expectations before and after code changes.
---

# ABI Skill

## When to use
Use this skill for implementation, refactors, bug fixes, or review follow-ups in this repo.

## Workflow
1. Read `AGENTS.md`, then review `tasks/lessons.md` for relevant prevention rules.
2. For non-trivial tasks, create/update a plan in `tasks/todo.md` with checkable items.
3. Run multi-CLI consensus (best effort) before implementation:
   - `/Users/donaldfilimon/.codex/skills/multi-cli-communication-expert/scripts/run_tricli_consensus.sh --mode code --timeout-sec 120 --prompt-file <file> --out-dir <dir>`
4. Implement the smallest elegant change that fixes root cause.
5. Verify behavior with repo-appropriate checks; do not mark done without evidence.
6. Update `tasks/todo.md` review notes with outcomes and residual risk.

## Required quality gates
- Prefer targeted checks first, then broader suite if environment permits.
- If a command is environment-blocked, record the limitation and continue with alternate evidence.
- Keep diffs minimal and avoid unrelated cleanup.

## Finalization checklist
- `git status --short` shows only intended files.
- Commit with a scoped message.
- Prepare PR summary with: change scope, validation commands/results, known limitations.
