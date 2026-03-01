# Lessons Learned

## 2026-02-28 - Markdown reset baseline
- Root cause: Workflow contract expected markdown files that were removed during global markdown purge.
- Prevention rule: Preserve required workflow markdown interfaces (`tasks/todo.md`, `tasks/lessons.md`) when performing markdown reset operations.

## 2026-03-01 - Use `apply_patch` directly for file edits
- Root cause: Attempted to execute patching through shell command flow instead of the dedicated patch tool.
- Prevention rule: For source edits, call `apply_patch` directly; reserve shell commands for read-only inspection or non-editing operations.
