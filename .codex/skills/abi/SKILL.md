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
3. Attempt multi-CLI consensus before implementation:
   - If `/Users/donaldfilimon/.codex/skills/multi-cli-communication-expert/scripts/run_tricli_consensus.sh` exists, run it in best effort mode.
   - If it is missing, record it as unavailable and continue; do not block the task.
4. Identify the import boundary before changing module paths:
   - Files compiled inside the `abi` module use relative imports.
   - Separate-root callers created by `build.zig` keep named-module imports. `src/services/tests/mod.zig` and its child tests, CLI roots, and other build-created modules should keep `@import("abi")`.
5. Implement the smallest elegant change that fixes root cause.
6. Verify behavior with repo-appropriate checks; do not mark done without evidence.
7. Update `tasks/todo.md` review notes with outcomes and residual risk.

## Required quality gates
- Prefer targeted checks first, then broader suite if environment permits.
- On Darwin/macOS hosts that need the wrapper path, run `zig fmt --check ...` first, then `./tools/scripts/run_build.sh typecheck --summary all`, then broaden to `test` / `feature-tests` only after the narrower gate passes.
- If a command is environment-blocked, record the limitation and continue with alternate evidence.
- Keep diffs minimal and avoid unrelated cleanup.

## Finalization checklist
- `git status --short` shows only intended files.
- Commit with a scoped message.
- Prepare PR summary with: change scope, validation commands/results, known limitations.
