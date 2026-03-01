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
