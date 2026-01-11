# Zig 0.16-dev Migration Guide

## Overview

This guide documents the migration of the ABI Framework to Zig 0.16.0-dev. The migration focuses on adopting the new `std.Io` API and removing deprecated interfaces.

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

**Status**: ✅ COMPLETED - Direct stream reader/writer usage for `std.http.Server`

**Change**: Direct stream reader/writer usage for `std.http.Server`

```zig
// INCORRECT (deprecated Zig 0.15 pattern)
var connection_reader = stream.reader(io, &recv_buffer);
var connection_writer = stream.writer(io, &send_buffer);
var server: std.http.Server = .init(
    &connection_reader.interface,  // ❌ Deprecated .interface access
    &connection_writer.interface,
);

// CORRECT (Zig 0.16 pattern)
var connection_reader = stream.reader(io, &recv_buffer);
var connection_writer = stream.writer(io, &send_buffer);
var server: std.http.Server = .init(
    &connection_reader,  // ✅ Direct reference
    &connection_writer,  // ✅ Direct reference
);
```

**Impact**: HTTP server initialization now uses direct reader/writer references instead of the `.interface` access pattern. This aligns with the new `std.Io` API design.

**Note**: This issue has been fixed in the current codebase. The migration is complete.

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
- `Reader` is `*std.Io.Reader` (not `*std.Io.File.Reader`)
- `Writer` is `*std.Io.Writer`

When using `std.Io.net.Stream.reader()` and `std.Io.net.Stream.writer()`, pass them directly without `.interface` access.

## Testing

All existing tests pass with the new API:
```bash
zig build test --summary all  # All tests pass
zig build benchmark                   # Benchmarks run successfully
```

## Build Configuration

The CI configuration has been updated to use Zig 0.16.0-dev instead of 0.17.0.

### CI Changes
- Updated `.github/workflows/ci.yml` to use `version: 0.16.0`

## Breaking Changes Summary

| Component | Change | Impact |
|-----------|--------|--------|
| HTTP Client | `std.io.AnyReader` → `std.Io.Reader` | Low - Streaming interface updated |
| HTTP Server | Removed `.interface` access | Low - Direct reader/writer usage |
| File I/O | No changes required | None - File.Reader still uses `.interface` for delimiters |

## Migration Checklist

- [x] Update CI to use Zig 0.16.0-dev
- [x] Replace `std.io.AnyReader` with `std.Io.Reader`
- [x] Fix HTTP Server initialization (remove .interface access)
- [x] Update documentation
- [x] Test all feature flag combinations
- [x] Run benchmarks
- [ ] ~~Consolidate HTTP modules~~ (Deferred - Current code works, consolidation is optional)

## Next Steps

1. Monitor Zig 0.16 release announcements for any additional breaking changes
2. Consider consolidating HTTP modules in a future refactor (optional)
3. Update minimum Zig version in `build.zig.zon` when 0.16 is officially released

## References

- [Zig 0.16-dev Branch](https://github.com/ziglang/zig/tree/master)
- [Zig Standard Library Documentation](https://ziglang.org/documentation/master/)
- [ABI Framework Documentation](../index.md)

## Contacts

src/shared/contacts.zig provides a centralized list of maintainer contacts extracted from the repository markdown files. Import this module wherever contact information is needed.

