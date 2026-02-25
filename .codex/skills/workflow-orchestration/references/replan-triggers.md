# Re-Plan Triggers

## Trigger Definitions

- `invalid-assumption`: A core assumption is disproven by repository or command output.
- `scope-break`: Required fix extends beyond planned scope or introduces new interfaces.
- `blocked-step`: Planned command or step cannot run in current environment.
- `verification-fail`: Verification fails after implementation.
- `conflict-detected`: Newly discovered changes conflict with current edits.

## Required Response

When any trigger fires, do all items below before continuing:

1. Stop implementation actions.
2. Record the trigger in `tasks/todo.md`.
3. Replace or amend checklist items for the new path.
4. Update verification steps for the new path.
5. Continue only after plan and verification are coherent again.

## Re-Plan Notes Format

Use this format in `tasks/todo.md` under verification or review notes:

- Trigger:
- Impact:
- Plan change:
- Verification change:
