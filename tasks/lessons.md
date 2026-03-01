# Lessons Learned

## 2026-02-28 - Markdown reset baseline
- Root cause: Workflow contract expected markdown files that were removed during global markdown purge.
- Prevention rule: Preserve required workflow markdown interfaces (`tasks/todo.md`, `tasks/lessons.md`) when performing markdown reset operations.

## 2026-03-01 - Use `apply_patch` directly for file edits
- Root cause: Attempted to execute patching through shell command flow instead of the dedicated patch tool.
- Prevention rule: For source edits, call `apply_patch` directly; reserve shell commands for read-only inspection or non-editing operations.

## 2026-03-01 - Zig 0.16 ZON parsing and `fromSliceAlloc`
- Root cause: `std.zon.parse.fromSliceAlloc` returns the parsed struct `T` directly, not a wrapper with a `.value` field like previous JSON parsers. It also uses the provided allocator for all nested slices, which can lead to memory leaks if not managed correctly.
- Prevention rule: When using `std.zon.parse.fromSliceAlloc` for complex configurations, wrap the call with an `std.heap.ArenaAllocator` to easily manage and clean up the nested allocations, and assign the result directly to your data variable. Also, avoid `std.ArrayList.init` and prefer `std.ArrayListUnmanaged(T).empty` to comply with Zig 0.16 patterns.

## 2026-03-01 - Generated registry and ZON data parsing regressions
- Root cause: Tooling still assumed direct `@import(...)` command wiring and used ad-hoc regex conversion for `.zon` data in the browser, which broke once command metadata moved to generated snapshot wiring and nested ZON structures were introduced.
- Prevention rule: For docs/CLI metadata extraction, resolve generated registry artifacts explicitly (not only direct imports); for `.zon` web consumption, use a deterministic parser for the generated subset instead of regex-based structural rewrites.


## 2026-03-01 - Do not invoke `apply_patch` via `exec_command`
- Root cause: Used `exec_command` to run the `apply_patch` shell helper after prior guidance preferred direct patch tooling.
- Prevention rule: Never run `apply_patch` through `exec_command`; use direct file-write/edit mechanisms allowed by the environment and reserve `exec_command` for inspection/verification commands.
