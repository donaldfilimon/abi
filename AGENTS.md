# Repository Guidelines

## Project Structure
- `src/` – library code (`core/`, `compute/`, `features/`, `framework/`, `shared/`).
- Public API is `src/abi.zig`.
- CLI entry point: `tools/cli/main.zig` (fallback to `src/main.zig`).
- Tests live in `src/tests/` and inline `test "…"` blocks.
- Examples, benchmarks, and docs are organised in parallel directories.
- Build files: `build.zig` and `build.zig.zon`.

## Build, Test, and Development Commands
```bash
# Core
zig build                      # Build all modules
zig build run -- --help        # Run CLI help
zig build test                 # Run entire test suite
zig build test --summary all   # Detailed test output

# Targeted tests (single file / filter by name)
zig test src/compute/runtime/engine.zig
zig test --test-filter "engine init"
zig test src/abi.zig --test-filter "version"

# Formatting and checks
zig fmt .                      # Reformat
zig fmt --check .              # Verify formatting

# Benchmarks / WASM
zig build benchmark
zig build benchmarks
zig build wasm

# Feature flags
zig build -Denable-gpu=false -Denable-network=true
```

## Coding Style & Naming Conventions
- **Formatting:** 4 spaces, no tabs, max 100 chars/line, one blank line between functions.
- **Types:** PascalCase (`Engine`, `TaskConfig`).
- **Functions / Variables:** snake_case (`create_engine`, `task_id`).
- **Constants:** UPPER_SNAKE_CASE, e.g., `MAX_TASKS`, `DEFAULT_MAX_TASKS`.
- **Struct Fields:** `allocator` comes first, followed by config/state, collections, resources, flags.
- **Re‑exports:** Public APIs should re‑export internal names via `pub usingnamespace`. Example:
```zig
pub const abi = @import("abi.zig");
```
- **Imports:** Keep explicit; avoid `usingnamespace` unless re‑exporting.
- **Cleanup:** Prefer `defer` and `errdefer` for resources.

## Zig 0.16‑Specific Conventions
### Memory Management
```zig
// Good – use unmanaged for struct fields
pub const BenchmarkSuite = struct {
    allocator: std.mem.Allocator,
    results: std.ArrayListUnmanaged(BenchmarkResult),
};
```
### Modern Format Specifiers
```zig
std.debug.print("{t}: {t}\n", .{status, status});
std.debug.print("Size: {B}\n", .{size});
std.debug.print("Duration: {D}\n", .{dur});
```
### I/O API Changes
```zig
var srv = std.http.Server.init(&reader, &writer);
```
### std.Io.Threaded Example
```zig
pub const Client = struct {
    io: std.Io.Threaded,
    client: std.http.Client,
    pub fn init(alloc: std.mem.Allocator) !Client {
        var io = std.Io.Threaded.init(alloc, .{});
        return .{ .io = io, .client = .{ .allocator = alloc, .io = io.io() } };
    }
    pub fn deinit(self: *Client) void {
        self.io.deinit();
    }
};
```
## Error Handling
Keep errors scoped: use specific `error{}` sets; document with `@return`; clean up with `errdefer`.

## Testing Guidelines
Tests in `src/tests/` and inline blocks. Use `error.SkipZigTest` to gate hardware‑specific tests.
```zig
test "gpu‑feature" {
    if (!enable_gpu) return error.SkipZigTest;
    // test body
}
```

## Commit & PR Guidelines
Use `<type>: <summary>` format. Keep summaries ≤ 72 chars. Focus commits; update docs when APIs change.

## Architecture References
System overview: `docs/intro.md`; API surface: `API_REFERENCE.md`.
Migration guide: `docs/migration/zig-0.16-migration.md`.

## Configuration Notes
Feature flags: `-Denable-*` (e.g., `enable-gpu`). GPU backends: `-Dgpu-*`. Credentials via env vars.
