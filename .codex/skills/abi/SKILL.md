---
name: abi
description: This skill should be used when working in the ABI Framework repository (Zig 0.16, pinned dev.2934+) to apply required planning, verification, and close-out workflow expectations before and after code changes.
---

# ABI Skill

## When to use
Activate for implementation, refactors, bug fixes, or review follow-ups in this repo.

## Workflow
1. Read `AGENTS.md`, then review `tasks/lessons.md` for relevant prevention rules.
2. For non-trivial tasks, create/update a plan in `tasks/todo.md` with checkable items.
3. Identify the import boundary before changing module paths:
   - Files compiled inside the `abi` module use relative imports.
   - Separate-root callers created by `build.zig` keep named-module imports. `src/services/tests/mod.zig` and its child tests, CLI roots, and other build-created modules should keep `@import("abi")`.
4. Implement the smallest elegant change that fixes root cause.
5. Verify behavior with repo-appropriate checks; do not mark done without evidence.
6. Update `tasks/todo.md` review notes with outcomes and residual risk.

## Required quality gates
- Prefer targeted checks first, then broader suite if environment permits.
- Primary build path: `zig build` with pinned Zig (`0.16.0-dev.2934+47d2e5de9`).
- On Darwin/macOS 25+ where stock Zig cannot link, run `zig fmt --check ...` first, then `zig build test --summary all` with a host-built Zig.
- If a command is environment-blocked, record the limitation and continue with alternate evidence.
- Keep diffs minimal and avoid unrelated cleanup.

## Key numbers
- 19 feature directories, 27 feature flags, 58 flag combos, 179 feature test entries
- 34 public modules cataloged, 35 examples, 18 cross-compilation targets
- CLI: ~98 command files, smoke tests (~53 vectors) + exhaustive integration tests

## Finalization checklist
- `git status --short` shows only intended files.
- Commit with a scoped message.
- Prepare PR summary with: change scope, validation commands/results, known limitations.
