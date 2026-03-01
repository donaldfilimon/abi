# Lessons Learned

## 2026-02-28 - Markdown reset baseline
- Root cause: Workflow contract expected markdown files that were removed during global markdown purge.
- Prevention rule: Preserve required workflow markdown interfaces (`tasks/todo.md`, `tasks/lessons.md`) when performing markdown reset operations.
