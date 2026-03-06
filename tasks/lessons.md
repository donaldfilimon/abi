# Lessons Learned

## 2026-03-06 - Aggressive-5 reprioritization must be canonical-first
- Root cause: Plan state drift appears when tasks/docs are edited before canonical roadmap catalog values are updated.
- Prevention rule: In one wave, update `roadmap_catalog.zig`, regenerate roadmap/plans data and markdown artifacts, then update `tasks/todo.md` and `tasks/lessons.md` before running close-out gates.

## 2026-03-06 - Pin + planning sync must move together
- Root cause: Zig pin and planning artifacts can drift when version updates are applied without the full contract set.
- Prevention rule: When repinning Zig, update `.zigversion`, `build.zig.zon`, `tools/scripts/baseline.zig`, `README.md`, and planning/generated artifacts in one wave.

## 2026-03-01 - Zig 0.16 ZON parsing ownership
- Root cause: `std.zon.parse.fromSliceAlloc` allocations were treated like wrapper-owned values instead of direct struct-owned slices.
- Prevention rule: Use arena-backed parsing for complex ZON inputs and deinit the arena at scope end.

## 2026-03-01 - Registry/docs extraction coupling
- Root cause: Tooling assumed direct imports and regex-based ZON rewrites after metadata moved to generated registry snapshots.
- Prevention rule: Resolve generated registry artifacts explicitly and keep deterministic parser paths for generated ZON.

## 2026-03-01 - Tool boundary discipline
- Root cause: Patch flow was attempted through generic shell execution instead of dedicated patch tooling.
- Prevention rule: Use dedicated patch/edit tools for file mutations and reserve shell for non-mutating inspection or command execution.

## 2026-03-06 - Workflow contract must be applied before implementation
- Root cause: Mandatory workflow rules were applied only after implementation work had already started, which created avoidable drift in consensus, task tracking, and review discipline.
- Prevention rule: For any non-trivial ABI task, review `tasks/lessons.md`, run the required multi-CLI consensus with a real prompt packet, and refresh `tasks/todo.md` before making repo-tracked edits.
