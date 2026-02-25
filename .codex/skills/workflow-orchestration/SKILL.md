---
name: workflow-orchestration
description: Enforce deterministic workflow execution for non-trivial engineering work with plan-first task tracking, verification-before-done, explicit re-plan triggers, and correction-driven lessons capture. Use when users request workflow orchestration, autonomous bug fixing with proof, or structured progress tracking through tasks/todo.md and tasks/lessons.md.
---

# Workflow Orchestration

## Overview

Apply this skill to run work as a strict contract: plan first, execute in scope, verify with evidence, and capture learning after corrections.

## Required Sequence

1. Read `tasks/lessons.md` before starting related work.
2. Classify the task:
- Trivial: implement directly with verification.
- Non-trivial (3+ steps, architectural choice, or cross-file behavior): write a checkable plan in `tasks/todo.md` before implementation.
3. Define success criteria and scope in `tasks/todo.md`.
4. Implement only after plan logging is complete.
5. Track progress by marking checklist items as completed as soon as evidence exists.
6. Verify before completion:
- Run relevant tests/checks.
- Capture command-level evidence in `tasks/todo.md`.
- Validate behavior difference when relevant (before/after).
7. Complete the review section in `tasks/todo.md` with residual risk.
8. After any user correction, append a root-cause lesson to `tasks/lessons.md`.

## Re-Plan Contract

Stop and re-plan immediately if any trigger applies:
- New error invalidates the current approach.
- Requirements conflict with observed repository truth.
- A planned step cannot be executed with current constraints.
- Verification disproves the fix.

Use `references/replan-triggers.md` for trigger definitions and required response format.

## Verification Contract

Never mark work complete without evidence. Match verification to change type:
- Code path changes: focused tests plus relevant integration gate.
- Build-system or Zig changes: toolchain check plus failing-step reproduction and rerun.
- API changes: endpoint-level validation.
- UI changes: visual verification.

Use `references/checklists.md` for exact pre-flight, execution, and completion checklists.

## File Contracts

Maintain these files as operational interfaces:
- `tasks/todo.md`: objective, scope, checkable tasks, evidence, and review.
- `tasks/lessons.md`: dated correction entries with root cause and prevention rule.

## Maintenance Contract

Treat repo-local skill as canonical:
- Canonical: `.codex/skills/workflow-orchestration/`
- Mirror: `/Users/donaldfilimon/.codex/skills/workflow-orchestration/`

After skill edits, sync mirror and re-validate both copies.
