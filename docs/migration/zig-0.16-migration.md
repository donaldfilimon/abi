# Zig 0.16 Migration Guide

## Overview

This guide documents the migration of the ABI Framework to Zig 0.16.x. The migration focuses on adopting the new `std.Io` API and removing deprecated interfaces.

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
| File I/O | `.interface` access required for many operations | Low - Use `.interface` for delimiter methods and write operations |
| Format Specifiers | Use `{t}` instead of `@errorName()/@tagName()` | Low - Improved formatting |

## Migration Checklist

- [x] Update CI to use Zig 0.16.x
- [x] Replace `std.io.AnyReader` with `std.Io.Reader`
- [x] Verify HTTP Server uses `.interface` correctly
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
- [ABI Framework Documentation](../index.md)

## Contacts

src/shared/contacts.zig provides a centralized list of maintainer contacts extracted from the repository markdown files. Import this module wherever contact information is needed.

