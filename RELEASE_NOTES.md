# Release Notes for ABI v0.3.0-dev (Unreleased)

**Target Release:** Q1 2026
**Zig Version:** 0.16.x

## Overview

ABI v0.3.0 builds upon the Zig 0.16 migration completed in v0.2.2, adding comprehensive GPU backend support, AI features, and distributed systems infrastructure. This release represents a major milestone in the framework's evolution.

## What's New

### Complete Zig 0.16 API Migration
- **std.Io API** - Full migration to the unified I/O interface
  - `std.Io.Threaded` for synchronous file operations
  - `std.Io.Dir.cwd()` replaces deprecated `std.fs.cwd()`
  - `std.Io.Clock.Duration` for sleep operations
- **HTTP Server Pattern** - Updated initialization with `.interface` access
  ```zig
  var connection_reader = stream.reader(io, &recv_buffer);
  var connection_writer = stream.writer(io, &send_buffer);
  var server: std.http.Server = .init(
      &connection_reader.interface,
      &connection_writer.interface,
  );
  ```
- **ArrayListUnmanaged Migration** - All `std.ArrayList` usage migrated to `std.ArrayListUnmanaged`
  - Better control over memory ownership
  - Explicit allocator passing throughout codebase
  - 13+ files modernized
- **Format Specifiers** - Adopted modern Zig 0.16 format specifiers
  - `{t}` for enum and error values (replaces `@tagName()`)
  - `{B}` for byte sizes, `{D}` for durations
- **Timing API** - `std.time.Timer` replaces deprecated `std.time.nanoTimestamp()`

### Documentation
- **Module Documentation** - Added `//!` documentation to core modules
  - `src/compute/mod.zig` - Compute runtime, memory, concurrency, GPU, SIMD
  - `src/core/mod.zig` - Platform detection, versioning, aligned buffers
  - `src/shared/observability/mod.zig` - Metrics collection primitives
- **Enhanced Guides** - Updated CONTRIBUTING.md with Zig 0.16 conventions
  - ArrayListUnmanaged preference with examples
  - Format specifier guidelines
  - Error handling best practices

### Testing
- **Integration Test Suite** - New end-to-end tests
  - Database operations (insert, search, statistics)
  - Vector operations (dot product, L2 norm, cosine similarity)
  - 4/4 integration tests now passing

### Benchmark Suite
- **Benchmark Framework** - New `benchmarks/mod.zig` module
  - `BenchmarkResult` struct with comprehensive metrics
  - `BenchmarkConfig` with warm-up and iteration control
  - Support for text, JSON, and CSV output formats
- **Benchmark Runner** - `benchmarks/run.zig` with 4 benchmarks
  - Empty Task (harness overhead)
  - Simple Compute (sum 1-1000)
  - Memory Allocation (1KB test)
  - Vector Dot Product (4D operations)
- **Documentation** - `benchmarks/README.md` with running instructions

## Security Improvements

### Critical Vulnerability Fix (CWE-22)
- **Path Traversal Prevention** in database backup/restore
  - Added `isSafeBackupPath()` validation
  - Added `normalizeBackupPath()` to restrict to `backups/` directory
  - Rejects path traversal sequences (`..`), absolute paths, Windows drive letters
  - Created `backups/` directory automatically
  - Returns `PathValidationError` on invalid input

### Memory Safety
- **Atomic Database Swap** in restore operation
  - Added `errdefer` for proper cleanup on failure
  - Prevents memory corruption when restore fails
  - Makes restore operation (fully succeed or fully fail)

## Breaking Changes

### API Changes
- **Timeout Semantics** (from v0.2.1):
  - **Old:** `timeout_ms=0` returned `ResultNotFound` after one check
  - **New:** `timeout_ms=0` immediately returns `EngineError.Timeout`
  - **Migration:** Replace `timeout_ms=0` with `timeout_ms=1000` for one-second timeout

### Database Security
- **Backup/Restore Filenames** (new in v0.2.1):
  - **Restriction:** Filenames must be relative and safe
  - **Validation:** No path traversal, absolute paths, or drive letters
  - **Location:** Operations restricted to `backups/` directory
  - **Migration:** Use simple filenames only, no directory components

## Bug Fixes

- Fixed duplicate `Histogram` definition in `src/shared/observability/mod.zig`
- Resolved compilation issues in test files
- Proper cleanup with `errdefer` patterns throughout codebase

## Performance

### Benchmarks Baseline
- Establishes performance baseline for future optimization
- Four benchmark categories: compute, memory, vector, harness overhead
- Results format: iterations, duration (ns), ops/sec, avg, min, max

### Code Quality
- All `std.ArrayList` migrated to `std.ArrayListUnmanaged` (better memory control)
- No `usingnamespace` usage found (explicit imports preferred)
- No deprecated API usage (Zig 0.16 compliant)

## Developer Experience

### CI/CD
- **GitHub Actions** - `.github/workflows/ci.yml`
  - Build and test on Ubuntu
  - Zig 0.16.x toolchain in CI
  - Feature flag combinations
  - Security scanning (unsafe patterns, path traversal)
  - Automatic formatting checks

### Issue Templates
- **Bug Report** - Structured template with reproduction steps
- **Feature Request** - Template with problem statement and proposed solution
- **RFC** - Template for major changes requiring discussion

### Public Roadmap
- **ROADMAP.md** - Version milestones and long-term goals
  - Version 0.3.0 (Q1 2026): GPU backends, async I/O
  - Version 0.4.0 (Q2 2026): Performance, DX, documentation, testing
  - Version 0.5.0 (Q3 2026): Distributed systems, HA, ecosystem

### Project Status
- **PROJECT_STATUS.md** - Comprehensive project state report
  - All commits and changes documented
  - Test results and quality metrics
  - Deployment readiness checklist
  - Next steps and recommendations

## Migration Guide

### For Zig 0.16 Compliance
If you're upgrading from v0.2.0 or earlier:

1. **std.Io API Migration**:
   ```zig
   // Old - std.fs
   const file = try std.fs.cwd().openFile(path, .{});

   // New - std.Io with backend
   var io_backend = std.Io.Threaded.init(allocator, .{
       .environ = std.process.Environ.empty,
   });
   defer io_backend.deinit();
   const io = io_backend.io();
   const content = std.Io.Dir.cwd().readFileAlloc(io, path, allocator, .limited(max_size)) catch |err| {
       return err;
   };
   ```

2. **ArrayList Migration** (if using custom code):
   ```zig
   // Old
   var list = std.ArrayList(u8).init(allocator);
   try list.append(item);

   // New
   var list = std.ArrayListUnmanaged(u8).empty;
   try list.append(allocator, item);
   ```

3. **Format Specifiers**:
   ```zig
   // Old
   std.debug.print("Status: {s}\n", .{@tagName(status)});
   std.log.err("Error: {s}", .{@errorName(err)});

   // New
   std.debug.print("Status: {t}\n", .{status});
   std.log.err("Error: {t}", .{err});
   ```

4. **Timing API**:
   ```zig
   // Old
   const start = std.time.nanoTimestamp();

   // New
   var timer = std.time.Timer.start() catch return error.TimerFailed;
   const elapsed_ns = timer.read();
   ```

5. **Sleep API**:
   ```zig
   // Old
   std.time.sleep(nanoseconds);

   // New - use time utilities module
   const time_utils = @import("src/shared/utils/time.zig");
   time_utils.sleepMs(100);   // Sleep 100 milliseconds
   ```

### For Security Changes
If you're using backup/restore:

```zig
// Old (still works in v0.2.0)
try db.backup("data/backup.db");
try db.restore("../malicious.db");

// New (v0.2.1+)
try db.backup("data/backup.db");  // OK: relative path
try db.restore("safe.db");  // OK: simple filename
try db.restore("../data.db");  // ERROR: path traversal
try db.restore("C:/data.db");  // ERROR: absolute path
```

### For Timeout Semantics
```zig
// Old (v0.2.0)
const result = try runTask(engine, u64, task, 0);

// New (v0.2.1+)
const result = try runTask(engine, u64, task, 1000);  // 1-second timeout
```

## Known Issues

None at time of release.

## Testing

All tests passing:
- Unit tests: 4/4 passed
- Integration tests: 4/4 passed
- Build: Successful
- Formatting: Clean (no violations)

## Compatibility

- **Zig Version:** 0.16.x (required)
- **Minimum Zig:** 0.16.0
- **Maximum Zig:** 0.16.x (0.17+ not yet supported)
- **Platforms:** Linux, macOS, Windows

## Acknowledgments

This release includes contributions from the ABI team and community feedback.

## Upgrade Instructions

1. Update Zig to 0.16.x or later
2. Pull latest changes from main branch
3. Review CHANGELOG.md for breaking changes
4. Update code to use ArrayListUnmanaged where needed
5. Review backup/restore path restrictions
6. Update timeout values in custom code
7. Run `zig build test` to verify

## Documentation

- [CHANGELOG.md](CHANGELOG.md) - Complete change history
- [README.md](README.md) - Quick start guide
- [CONTRIBUTING.md](CONTRIBUTING.md) - Development guidelines
- [API_REFERENCE.md](API_REFERENCE.md) - API documentation
- [SECURITY.md](SECURITY.md) - Security policy
- [ROADMAP.md](ROADMAP.md) - Future plans
- [examples/](examples/) - Example programs
- [benchmarks/](benchmarks/) - Benchmark suite

## Next Release (v0.4.0)

Planned for Q2 2026:
- SIMD optimizations (AVX-512, NEON, WASM SIMD)
- Memory management improvements (arena allocator, memory pools)
- Comprehensive API documentation (auto-generated)
- Performance benchmark suite with regression detection
- Tooling (debugger integration, performance profiler)

See [ROADMAP.md](ROADMAP.md) for detailed roadmap.

## Contacts

src/shared/contacts.zig provides a centralized list of maintainer contacts extracted from the repository markdown files. Import this module wherever contact information is needed.
See [TODO.md](TODO.md) for the list of pending implementations.
... (no change) ...
[Main Workspace](MAIN_WORKSPACE.md)
