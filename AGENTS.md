# Repository Guidelines

## Project Structure

- `src/` holds the library: `core/`, `compute/`, `features/`, `framework/`, `shared/`
- Public API: `src/abi.zig`; CLI: `tools/cli/main.zig` (fallback: `src/main.zig`)
- Tests: `src/tests/` (integration, property tests) and inline `test "..."` blocks
- Examples, benchmarks, and docs in corresponding directories
- Build: `build.zig` and `build.zig.zon`

## Build, Test, and Development Commands

**Required:** Zig 0.16.x

```bash
# Core commands
zig build                      # Build all modules
zig build run -- --help        # Run CLI
zig build test                 # Run test suite
zig build test --summary all   # Detailed test output

# Single file tests (IMPORTANT - use this for focused testing)
zig test src/compute/runtime/engine.zig      # Run specific file tests
zig test --test-filter "engine init"         # Filter by test name
zig test src/abi.zig --test-filter "version" # Filter in specific file

# Formatting and checks
zig fmt .                      # Format code
zig fmt --check .              # Check formatting (no linter beyond this)

# Benchmarks and WASM
zig build benchmark            # Run legacy benchmarks
zig build benchmarks           # Run comprehensive benchmarks
zig build wasm                 # Build WASM bindings

# Feature flags
zig build -Denable-gpu=false -Denable-network=true
```

## Coding Style & Naming Conventions

- **Formatting:** 4 spaces, no tabs, max 100 chars/line, one blank line between functions
- **Types:** PascalCase (`Engine`, `TaskConfig`)
- **Functions/Variables:** snake_case (`createEngine`, `task_id`)
- **Constants:** UPPER_SNAKE_CASE (`MAX_TASKS`, `CacheLineBytes`)
- **Documentation:** `//!` module docs, `///` function docs with `@param`/`@return`
- **Imports:** Explicit only; never `usingnamespace`
- **Cleanup:** Prefer `defer`/`errdefer`
- **Allocator:** First field/argument when needed; use `std.ArrayListUnmanaged` for struct fields

## Zig 0.16-Specific Conventions

### Memory Management

```zig
// Good - unmanaged for struct fields
pub const BenchmarkSuite = struct {
    allocator: std.mem.Allocator,
    results: std.ArrayListUnmanaged(BenchmarkResult),
};

// Usage
try list.append(allocator, item);
list.deinit(allocator);

// Avoid - managed for struct fields
results: std.ArrayList(BenchmarkResult),
```

### Modern Format Specifiers

```zig
// Good - use modern specifiers
std.debug.print("Status: {t}\n", .{status});       // {t} for enums/errors
std.debug.print("Size: {B}\n", .{size});           // {B} for bytes (raw)
std.debug.print("Duration: {D}\n", .{duration});   // {D} for nanoseconds
std.debug.print("Data: {b64}\n", .{data});         // {b64} for base64

// Avoid - manual conversions
std.debug.print("Status: {s}\n", .{@tagName(status)});
```

**Exceptions for JSON:** JSON requires strings, so `@tagName()` and `@errorName()` are acceptable.

### I/O API Changes

```zig
// HTTP Server - direct reader/writer
var connection_reader = stream.reader(io, &recv_buffer);
var connection_writer = stream.writer(io, &send_buffer);
var server: std.http.Server = .init(
    &connection_reader,     // Direct reference (no .interface)
    &connection_writer,    // Direct reference (no .interface)
);

// Streaming - use std.Io.Reader
pub const StreamingResponse = struct {
    reader: std.Io.Reader,
    response: HttpResponse,
};

// File.Reader uses .interface for delimiter methods only
const line_opt = reader.interface.takeDelimiter('\n') catch |err| {
    return err;
};
```

### std.Io.Threaded Usage

```zig
// Async runtime pattern for HTTP clients
pub const HttpClient = struct {
    allocator: std.mem.Allocator,
    io_backend: std.Io.Threaded,
    client: std.http.Client,

    pub fn init(allocator: std.mem.Allocator) !HttpClient {
        var io_backend = std.Io.Threaded.init(allocator, .{
            .environ = std.process.Environ.empty,
        });
        return .{
            .allocator = allocator,
            .io_backend = io_backend,
            .client = std.http.Client{
                .allocator = allocator,
                .io = io_backend.io(),
            },
        };
    }

    pub fn deinit(self: *HttpClient) void {
        self.io_backend.deinit();
    }
};
```

## Error Handling

- Use specific error sets instead of `anyerror` where possible
- Document errors with `@return` tags
- Use `errdefer` for cleanup on error paths

```zig
// Good - specific error set
const TaskError = error{
    Timeout,
    Cancelled,
    TaskFailed,
} || std.mem.Allocator.Error;

// Good - errdefer cleanup
var buffer = try allocator.alloc(u8, size);
errdefer allocator.free(buffer);
```

**Keep `anyerror` for:**

- Function pointer types needing flexibility
- Generic error logging contexts

## Testing Guidelines

- Tests in `src/tests/` and inline `test "..."` blocks
- Name tests descriptively; add coverage for new features or note why not
- Gate hardware-specific tests with feature flags (e.g., `-Denable-gpu=true`)

```zig
// Feature-gated test pattern
fn testGPUOperations(allocator: std.mem.Allocator) !void {
    if (!abi.gpu.moduleEnabled()) {
        std.debug.print("GPU module not enabled, skipping\n", .{});
        return;
    }
    // ... test code
}
```

## Commit & PR Guidelines

- Format: `<type>: <imperative summary>` with `feat`, `fix`, `docs`, `refactor`, `test`, `chore`, `build`
- Keep summaries <= 72 chars
- Keep commits focused; update docs when public APIs change
- PRs should explain intent, link issues, and list commands run (e.g., `zig build`, `zig build test`, `zig fmt .`)

## Architecture References

- System overview: `docs/intro.md`
- API surface: `API_REFERENCE.md`
- Zig 0.16 migration: `docs/migration/zig-0.16-migration.md`

## Configuration Notes

- Feature flags: `-Denable-*` (e.g., `enable-gpu`, `enable-ai`)
- GPU backends: `-Dgpu-*` (e.g., `gpu-cuda`, `gpu-vulkan`)
- Connector credentials via environment variables (see `README.md`)
