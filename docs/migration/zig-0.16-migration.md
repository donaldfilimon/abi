# Zig 0.16 Migration Guide
> **Codebase Status:** Synced with repository as of 2026-01-30.

> **Status:** Complete ✅
> **Developer Guide**: See [CONTRIBUTING.md](../../CONTRIBUTING.md) for coding patterns and [CLAUDE.md](../../CLAUDE.md) for comprehensive guidance.
>
> **Last Updated:** January 16, 2026
> **Zig Version:** 0.16.x

## Overview

This guide documents the completed migration of the ABI Framework to Zig 0.16.x. The migration includes:

- **std.Io unified API** - Full adoption of the new I/O interface
- **std.Io.Threaded** - Synchronous file operations
- **std.Io.Dir.cwd()** - Replaces deprecated `std.fs.cwd()`
- **std.time.Timer** - High-precision timing
- **std.Io.Clock.Duration** - Sleep operations
- **std.ArrayListUnmanaged** - Explicit allocator passing
- **Format specifiers** - `{t}`, `{B}`, `{D}` for modern formatting

## Changes Made

### 1. Reader Type Migration

**File**: `src/shared/utils/http/async_http.zig`

**Change**: Replaced `std.io.AnyReader` with `std.Io.Reader`

```zig
// OLD (Zig 0.15)
pub const StreamingResponse = struct {
    reader: std.io.AnyReader,
    response: HttpResponse,
    // ...
};

// NEW (Zig 0.16)
pub const StreamingResponse = struct {
    reader: std.Io.Reader,
    response: HttpResponse,
    // ...
};
```

**Impact**: Streaming HTTP responses now use the new unified reader interface.

### 2. HTTP Server Initialization

**File**: `src/features/database/http.zig`

**Status**: ✅ CORRECT - Uses `.interface` access for `std.http.Server`

**Pattern**: The `std.http.Server.init()` function expects `*std.Io.Reader` and `*std.Io.Writer`, but `std.Io.net.Stream.reader()` returns `std.Io.net.Stream.Reader`. The `.interface` field provides the `std.Io.Reader` type that the server expects.

```zig
// CORRECT (Zig 0.16 pattern)
var connection_reader = stream.reader(io, &recv_buffer);
var connection_writer = stream.writer(io, &send_buffer);
var server: std.http.Server = .init(
    &connection_reader.interface,  // ✅ .interface provides *Io.Reader
    &connection_writer.interface,  // ✅ .interface provides *Io.Writer
);
```

**Rationale**: The `std.Io.net.Stream.Reader` type wraps `std.Io.Reader` in its `.interface` field. Since `std.http.Server.init()` expects `*Io.Reader` (not `*Io.net.Stream.Reader`), the `.interface` access is required.

### 3. File Reader Delimiter Methods

**File**: `src/cli.zig`

**Change**: Kept `.interface` access for `std.Io.File.Reader` delimiter methods

```zig
// File.Reader .interface access is still valid for delimiter methods
const line_opt = reader.interface.takeDelimiter('\n') catch |err| {
    // ...
};
```

**Rationale**: The `std.Io.File.Reader` type provides specialized delimiter methods through its `interface` field. This is intentional and correct usage in Zig 0.16.

### 4. Format Specifiers for Errors and Enums

**Files**: `build.zig`, `src/features/ai/explore/results.zig`

**Change**: Use `{t}` format specifier instead of `@errorName()` or `@tagName()` in format strings

```zig
// OLD (Zig 0.15 pattern)
std.log.err("Error: {s}", .{@errorName(err)});
std.debug.print("State: {s}", .{@tagName(state)});

// NEW (Zig 0.16 pattern)
std.log.err("Error: {t}", .{err});
std.debug.print("State: {t}", .{state});
```

**Rationale**: The `{t}` format specifier directly handles error and enum types, producing human-readable output without manual conversion. This is cleaner and more idiomatic.

**Note**: `@errorName()` is still valid when you need the error name as a `[]const u8` string (e.g., for storing in a struct field), but should not be used with format specifiers.

### 5. Synchronous File I/O with std.Io.Threaded

**Pattern**: For synchronous file operations outside async contexts, use `std.Io.Threaded`:

```zig
// Create I/O backend for synchronous file operations
var io_backend = std.Io.Threaded.init(allocator, .{
    .environ = std.process.Environ.empty,
});
defer io_backend.deinit();
const io = io_backend.io();

// Read file
const content = std.Io.Dir.cwd().readFileAlloc(io, path, allocator, .limited(10 * 1024 * 1024)) catch |err| {
    return err;
};
defer allocator.free(content);

// Write file
var file = std.Io.Dir.cwd().createFile(io, path, .{ .truncate = true }) catch return error.Failed;
defer file.close(io);
var writer = file.writer(io);
try writer.writeAll(content);
```

**Note**: `std.fs.cwd()` does not exist in Zig 0.16. Use `std.Io.Dir.cwd()` (passing an `std.Io` context) instead.

### 6. Timing and Measurement

**Change**: Use `std.time.Timer` for high-precision timing (not `std.time.nanoTimestamp()`):

```zig
// OLD (deprecated)
const start = std.time.nanoTimestamp();
// ... work ...
const elapsed = std.time.nanoTimestamp() - start;

// NEW (Zig 0.16)
var timer = std.time.Timer.start() catch return error.TimerFailed;
// ... work ...
const elapsed_ns = timer.read();
```

### 7. Sleep API

**Change**: Use `std.Io`-based sleep instead of `std.time.sleep()`:

```zig
// OLD (deprecated)
std.time.sleep(nanoseconds);

// NEW (Zig 0.16) - use the time utilities module
const time_utils = @import("src/shared/utils/time.zig");
time_utils.sleepMs(100);   // Sleep 100 milliseconds
time_utils.sleepSeconds(1); // Sleep 1 second
time_utils.sleepNs(50_000); // Sleep 50 microseconds

// Direct Io usage (when you have an Io context)
const duration = std.Io.Clock.Duration{
    .clock = .awake,
    .raw = .fromNanoseconds(nanoseconds),
};
std.Io.Clock.Duration.sleep(duration, io) catch {};
```

### 8. Aligned Memory Allocation

**Change**: For aligned allocations, use `std.mem.Alignment`:

```zig
// NEW (Zig 0.16)
const page_size = 4096;
const data = try allocator.alignedAlloc(u8, comptime std.mem.Alignment.fromByteUnits(page_size), size);
defer allocator.free(data);
```

### 9. Memory Management

**Change**: Prefer `std.ArrayListUnmanaged` over `std.ArrayList`:

```zig
// OLD
var list = std.ArrayList(u8).init(allocator);
try list.append(item);
list.deinit();

// NEW (Zig 0.16 - explicit allocator passing)
var list = std.ArrayListUnmanaged(u8).empty;
try list.append(allocator, item);
list.deinit(allocator);
```

**Benefits**:
- Explicit allocator passing improves clarity
- Better control over memory ownership
- Reduces hidden dependencies
- Modern Zig 0.16 idiom

## API Compatibility Notes

### Reader Type Hierarchy

Zig 0.16 introduces a unified `std.Io.Reader` type:
- Base type: `std.Io.Reader` - Generic reader interface
- File reader: `std.Io.File.Reader` - File-specific reader with `.interface` for delimiter methods
- Net reader: `std.Io.net.Stream.Reader` - Network stream reader

### HTTP Server Initialization

The `std.http.Server.init()` function signature:
```zig
pub fn init(in: *Reader, out: *Writer) Server
```

Where:
- `Reader` is `*std.Io.Reader` (not `*std.Io.net.Stream.Reader`)
- `Writer` is `*std.Io.Writer`

When using `std.Io.net.Stream.reader()` and `std.Io.net.Stream.writer()`, access their `.interface` field to get the correct type for `std.http.Server.init()`.

## Testing

All existing tests pass with the new API:
```bash
zig build test --summary all  # All tests pass
zig build benchmark                   # Benchmarks run successfully
```

## Build Configuration

The CI configuration has been updated to use Zig 0.16.x instead of 0.17.0.

### CI Changes
- Updated `.github/workflows/ci.yml` to use `version: 0.16`

## Breaking Changes Summary

| Component | Change | Impact |
|-----------|--------|--------|
| HTTP Client | `std.io.AnyReader` → `std.Io.Reader` | Low - Streaming interface updated |
| HTTP Server | Requires `.interface` access for stream reader/writer | Low - Use `.interface` to get `*Io.Reader` |
| File I/O | `std.fs.cwd()` → `std.Io.Dir.cwd()` | Medium - Requires `std.Io` context |
| Synchronous I/O | Requires `std.Io.Threaded` backend | Medium - New initialization pattern |
| Sleep API | `std.time.sleep()` → `std.Io.Clock.Duration.sleep()` | Low - Use time utilities module |
| Timing | `std.time.nanoTimestamp()` → `std.time.Timer` | Low - Better API |
| Format Specifiers | Use `{t}` instead of `@errorName()/@tagName()` | Low - Improved formatting |
| ArrayListUnmanaged | Preferred over `std.ArrayList` | Low - Explicit allocator passing |

## Migration Checklist

- [x] Update CI to use Zig 0.16.x
- [x] Replace `std.io.AnyReader` with `std.Io.Reader`
- [x] Verify HTTP Server uses `.interface` correctly
- [x] Migrate `std.fs.cwd()` to `std.Io.Dir.cwd()`
- [x] Implement `std.Io.Threaded` for synchronous I/O
- [x] Replace `std.time.sleep()` with `std.Io.Clock.Duration.sleep()`
- [x] Replace `std.time.nanoTimestamp()` with `std.time.Timer`
- [x] Migrate `std.ArrayList` to `std.ArrayListUnmanaged`
- [x] Use `{t}` format specifier for error/enum values
- [x] Update documentation
- [x] Test all feature flag combinations
- [x] Run benchmarks
- [ ] ~~Consolidate HTTP modules~~ (Deferred - Current code works, consolidation is optional)

## Next Steps

1. Monitor Zig 0.16.x release announcements for any additional breaking changes
2. Consider consolidating HTTP modules in a future refactor (optional)
3. Keep `build.zig.zon` minimum Zig version aligned to the latest 0.16.x point release

## References

- [Zig main branch](https://github.com/ziglang/zig/tree/master)
- [Zig Standard Library Documentation](https://ziglang.org/documentation/master/)

---

## See Also

- [Documentation Index](../docs-index.md) - Full documentation
- [Framework](../framework.md) - Configuration and lifecycle
- [Troubleshooting](../troubleshooting.md) - Migration issues
*See [../../TODO.md](../../TODO.md) (including the [Claude‑Code Massive TODO](../../TODO.md#claude-code-massive-todo)) and [../../ROADMAP.md](../../ROADMAP.md) for the Llama‑CPP parity task list and upcoming milestones.*
