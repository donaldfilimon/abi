# Lessons Learned

## 2026-02-25 - Workflow orchestration guardrail miss
- Root cause: Started repository edits without first creating a checkable task plan file for a non-trivial multi-file change.
- Prevention rule: Before any non-trivial implementation in this repo, create or update `tasks/todo.md` with objective, scope, verification criteria, and checklist, then proceed to code edits.
