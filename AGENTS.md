# Repository Guidelines

## Project Structure & Module Organization

- `src/` holds the library; key areas include `src/compute/`, `src/features/`,
  `src/framework/`, `src/shared/`, and `src/core/`.
- Public API entrypoints live in `src/abi.zig` and `src/root.zig`; the CLI
  entrypoint is `tools/cli/main.zig` (fallback: `src/main.zig`).
- `tests/` contains unit, integration, and property tests (see `tests/mod.zig`).
- `examples/`, `benchmarks/`, and `docs/` provide demos, performance runs, and
  documentation.
- Build metadata is in `build.zig` and `build.zig.zon`.

## Build, Test, and Development Commands

Zig 0.16.x is required.

```bash
# Core build commands
zig build                                 # Build all modules
zig build run -- --help                   # Run the CLI
zig build test                            # Run the full test suite
zig build test --summary all              # Detailed test output

# Single-file and filtered tests
zig test src/compute/runtime/engine.zig   # Single-file tests
zig test --test-filter "engine init"      # Filtered test names

# Benchmarks and formatting
zig build benchmark                       # Run benchmarks
zig fmt .                                 # Format code
zig fmt --check .                         # Format check

# Feature flags
zig build -Denable-gpu=false -Denable-network=true  # Disable GPU, enable network
zig build -Denable-gpu=true -Denable-database=true  # Enable GPU and database
```

## Coding Style & Naming Conventions

- 4 spaces, no tabs, max 100 chars, one blank line between functions.
- `//!` module docs, `///` public API docs with `@param`/`@return`.
- Types: PascalCase; functions/variables: snake_case; constants: UPPER_SNAKE_CASE.
- Allocator is the first field/arg when needed; prefer `std.ArrayListUnmanaged`
  for struct fields.
- Use explicit imports only; never `usingnamespace`. Prefer `defer`/`errdefer`
  for cleanup.

## Zig 0.16-Specific Conventions

### Memory Management

**Prefer `std.ArrayListUnmanaged` over `std.ArrayList` for struct fields:**

```zig
// Good - unmanaged for struct fields
pub const BenchmarkSuite = struct {
    allocator: std.mem.Allocator,
    results: std.ArrayListUnmanaged(BenchmarkResult),
};

// Good - explicit allocator in methods
try list.append(allocator, item);
list.deinit(allocator);

// Avoid for struct fields
results: std.ArrayList(BenchmarkResult),
```

### Modern Format Specifiers

Use Zig 0.16 format specifiers instead of manual conversions:

```zig
// Good - modern specifiers
std.debug.print("Status: {t}\n", .{status});           // {t} for enums/errors
std.debug.print("Size: {B}\n", .{size});               // {B} for bytes
std.debug.print("Duration: {D}\n", .{duration});       // {D} for durations
std.debug.print("Data: {b64}\n", .{data});             // {b64} for base64

// Avoid - legacy patterns
std.debug.print("Status: {s}\n", .{@tagName(status)});
```

### Format Specifier Migration Guidelines

**Use {t} for enums and errors:**
```zig
// Good - use {t} directly
std.debug.print("Backend: {t}\n", .{backend});
std.debug.print("Error: {t}\n", .{err});

// Avoid - manual conversions
std.debug.print("Backend: {s}\n", .{@tagName(backend)});
std.debug.print("Error: {s}\n", .{@errorName(err)});
```

**Exception: JSON serialization requires strings:**
```zig
// Acceptable - JSON needs strings
try obj.put("level", json.Value{ .string = @tagName(level) });
try obj.put("error", json.Value{ .string = @errorName(err) });
```

**Use {B} for byte sizes (expects raw bytes, not pre-divided):**
```zig
// Good - use raw byte values with {B}
try writer.print("Total Memory: {B}\n", .{info.total_memory});
try writer.print("Cache Size: {B}\n", .{info.l2_cache_size});

// Avoid - manual division
try writer.print("Memory: {d} MB\n", .{info.total_memory / (1024 * 1024)});
```

**Use {D} for durations (expects nanoseconds, not milliseconds):**
```zig
// Good - use nanosecond durations with {D}
const duration_ns = end_time - start_time;
std.debug.print("Duration: {D}\n", .{duration_ns});

// Note: If you have milliseconds, convert or use {d}ms
const duration_ms = @divTrunc(duration_ns, std.time.ns_per_ms);
std.debug.print("Duration: {d}ms\n", .{duration_ms});
```

**Use {b64} for base64 encoding:**
```zig
// Good - encode byte slices directly
std.debug.print("Auth: {b64}\n", .{auth_bytes});
```

### I/O API Changes (Zig 0.16)

```zig
// HTTP Server - direct reader/writer, no .interface
var server: std.http.Server = .init(
    &stream.reader(io, &recv_buffer),
    &stream.writer(io, &send_buffer),
);

// Streaming response - use std.Io.Reader
pub const StreamingResponse = struct {
    reader: std.Io.Reader,
    response: HttpResponse,
};

// File.Reader still uses .interface for delimiter methods
const line_opt = reader.interface.takeDelimiter('\n') catch |err| {
    return err;
};
```

## Error Handling

- Use specific error sets instead of `anyerror` where possible.
- Document when errors can occur with `@return` tags.
- Use `errdefer` for cleanup on error paths.

```zig
// Good - specific error set
const FileError = error{
    FileNotFound,
    PermissionDenied,
    ReadError,
} || std.fs.File.OpenError;

// Good - errdefer cleanup
var buffer = try allocator.alloc(u8, size);
errdefer allocator.free(buffer);
```

### Error Set Guidelines

**Prefer specific error sets over anyerror:**
```zig
// Good - specific error set
const TaskError = error{
    Timeout,
    Cancelled,
    TaskFailed,
} || std.mem.Allocator.Error;
```

**Keep anyerror for truly generic contexts:**
```zig
// Acceptable - function pointer types need flexibility
execute: *const fn (std.mem.Allocator, *anyopaque) anyerror!ResultBlob,
handler: *const fn ([]const u8, *Context) anyerror!ToolResult,

// Acceptable - generic error logging
pub fn log(self: ErrorContext, level: std.log.Level, err: anyerror) void {
    // ... logging logic
}
```

## Testing Guidelines

- Tests live in `tests/` and inline `test "..."` blocks in modules.
- Name tests descriptively, and add coverage for new features or note why not.
- Use feature flags to gate hardware-specific tests (e.g., `-Denable-gpu=true`).

## Commit & Pull Request Guidelines

- History favors short, imperative subjects; doc-only commits often use `docs:`.
- Required format: `<type>: <imperative summary>` with `feat`, `fix`, `docs`,
  `refactor`, `test`, `chore`, or `build`; keep summaries <= 72 chars.
- Keep commits focused and update docs when public APIs change.
- PRs should explain intent, link related issues if any, and list commands run
  (e.g., `zig build`, `zig build test`, `zig fmt .`).

## Architecture References

- System overview: `docs/intro.md`.
- API surface: `API_REFERENCE.md`.
- Zig 0.16 migration: `docs/migration/zig-0.16-migration.md`.

## Configuration Notes

- Feature flags use `-Denable-*` and GPU backends use `-Dgpu-*` (see `README.md`).
- Connector credentials are provided via environment variables listed in
  `README.md`.
