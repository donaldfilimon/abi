# Workflow Checklists

## Start Checklist

- Read `tasks/lessons.md` for related patterns.
- Confirm current task objective and out-of-scope items.
- Inspect repository state (`git status --short`) before edits.
- Identify smallest valid verification loop for the task.

## Plan Checklist (Non-Trivial Work)

- Write objective and scope in `tasks/todo.md`.
- Write checkable implementation steps.
- Write explicit verification steps and expected outcomes.
- Mark assumptions that could invalidate the approach.

## Execution Checklist

- Work one checklist item at a time.
- Mark item complete only when evidence exists.
- Keep implementation scoped to required files.
- Re-plan immediately when blocked instead of continuing with workaround drift.

## Verification Checklist

- Capture exact commands executed.
- Capture whether each command passed or failed.
- If behavior changed, document intended before/after delta.
- If tests are skipped, state why and document residual risk.

## Completion Checklist

- Ensure all checklist items are complete.
- Fill `Review` section in `tasks/todo.md`:
- Task.
- Scope delivered.
- Verification evidence summary.
- Risks and follow-ups.
- If a correction happened, append a lesson entry in `tasks/lessons.md`.
