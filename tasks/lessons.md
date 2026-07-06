# Lessons — ABI Framework

Session-start checklist and conventions for agents working on this repo.

## Session-Start Checklist

1. Read this file (`tasks/lessons.md`) at session start.
2. Read `tasks/todo.md` for current work items and priorities.
3. Run `./build.sh check` to verify baseline state before making changes.
4. Identify which modules you are touching; confirm mod/stub pairs if changing public APIs.
5. Update `tasks/todo.md` as you begin and complete work items.

## Key Conventions

### Naming
- Functions/variables: `camelCase`
- Types/structs: `PascalCase`
- Constants: `SCREAMING_SNAKE_CASE`
- Enum variants: `snake_case`

### Import Rules
- **Within `src/`**: Relative imports ONLY, except the MCP executable + handler module graph (`src/mcp/main.zig` plus the `handlers.zig` group: `handlers.zig`, `ai_tools.zig`, `connector_tools.zig`, `plugin_tools.zig`, `state.zig`) that intentionally imports the public `abi` module for tool dispatch.
- **From outside `src/`**: Use `@import("abi")` and `@import("build_options")`.
- Always include `.zig` extension on path imports.

### Zig 0.17 Patterns
- Entry point: `pub fn main(init: std.process.Init) !void`
- `ArrayListUnmanaged(T).empty` for initialization (not `.init(allocator)`)
- `std.mem.trimEnd` (not `trimRight`)
- `std.mem.splitScalar` / `std.mem.splitAny` / `std.mem.splitSequence` (not `split`)
- Use `foundation.time.unixMs()` for timestamps (not `std.time.milliTimestamp`)
- `std.mem.Allocator` passed explicitly; no global allocator

### Error Handling
- Silent empty catch blocks are forbidden in data access, inference, and persistence paths.
- Errors must be logged or propagated.
- Use error unions (`!T`) for runtime failures.
- `@panic` only for unrecoverable invariants in CLI/tests.
- `unreachable` only for provably impossible branches.

### Mod/Stub Pattern
- Each feature has `mod.zig` (real impl) and `stub.zig` (no-op when disabled).
- Changing public APIs requires updating **both** files.
- Run `zig build check-parity` after any public API change.

### Testing
- Tests are inline `test {}` blocks — no separate `test/` directory.
- Each module must include `std.testing.refAllDecls(@This())` in its test block.
- Run single test: `zig build test -Dtest-filter="<pattern>"` (on macOS `./build.sh test -Dtest-filter="<pattern>"`). The post-`--` form `-- --test-filter` is **not** wired up and is silently ignored.

## Build/Test Workflow

```bash
# Baseline check
./build.sh check

# Build specific targets
./build.sh cli        # Build CLI binary
./build.sh mcp        # Build MCP server

# Run all module and connector tests
zig build test --summary all

# Run bundled plugin coverage
zig build test-plugins --summary all

# Run single test (the post-`--` form is silently ignored; use the build option)
zig build test -Dtest-filter="pattern"

# Lint and format
zig build lint        # Check formatting
zig build fix         # Auto-format

# Parity check (after API changes)
zig build check-parity

# Full validation
./build.sh full-check
```

## Common Pitfalls to Avoid

1. **Circular imports**: Avoid `@import("abi")` from within `src/` except the MCP executable + handler module graph (`src/mcp/main.zig` + the `handlers.zig` group); use relative paths elsewhere.
2. **Missing `.zig` extension**: All path imports must include the extension.
3. **Empty catch blocks**: Always log or propagate errors; never silently swallow them.
4. **Wrong ArrayList init**: Use `ArrayListUnmanaged(T).empty`, not `.init(allocator)`.
5. **Deprecated APIs**: `std.mem.split` → `splitScalar`/`splitAny`/`splitSequence`; `trimRight` → `trimEnd`.
6. **Timestamp API**: Use `foundation.time.unixMs()`, not `std.time.milliTimestamp`.
7. **Mod/stub mismatch**: Always update both when changing public APIs.
8. **macOS build entrypoint**: Prefer `./build.sh ...` on macOS — the documented Darwin workflow sets up Metal linking and project conventions; raw `zig build` works with a compatible local toolchain but bypasses the wrapper.
9. **Missing refAllDecls**: Every module test block must include `std.testing.refAllDecls(@This())`.
10. **Feature flag imports**: Use `build_options.feat_*` for conditional compilation, not runtime checks.

## MemoryTracker Wiring (learned this session)

- **Balance transient tracking; never track escaping buffers.** Track owned-and-freed scratch (the HNSW search arena, completion metadata/key) as a `trackAllocNoTag`/`trackFreeNoTag` pair so cumulative totals reflect cost without a false leak. An allocation that escapes to the caller (search `results`, the completion response) must NOT be tracked at the alloc site — its free happens elsewhere, so tracking only the alloc unbalances the tracker.
- **Reach a tracker parity-safely via a method, not a signature change.** `check_parity` compares only top-level `pub const`/`pub fn` *names* between `mod.zig`/`stub.zig`; struct methods are invisible to it. Adding `Store.getTracker()` (mod returns the tracker, stub returns `null`) let the AI path reach a tracker with no signature/parity churn — but the stub still needs the method or the `-Dfeat-wdbx=false` matrix won't compile.
- **Isolate transient from persistent tracking in tests.** Persistent allocations (vector inserts) track-alloc but never free until `deinit`, so `getTotalFreed() > 0` after an op cleanly proves a *balanced transient* pair fired — a stronger assertion than `getTotalAllocated() > 0`, which persistent allocs satisfy anyway. Added `getTotalAllocated`/`getTotalFreed` getters for exactly this.
