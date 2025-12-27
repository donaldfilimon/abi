# Changelog

## 0.2.2 - 2025-12-27
### Zig 0.16 Modernization
- Migrated all `std.ArrayList` to `std.ArrayListUnmanaged`
  - Passes allocator explicitly to all methods
  - Better control over memory ownership
  - Updated 13 files across the codebase
- Adopted new format specifier `{t}` for enum and error formatting
  - Replaced `@tagName()` calls with modern `{t}` specifier
  - Cleaner, more idiomatic Zig 0.16 code
  - Updated 4 files: demo.zig, cli.zig, logging/mod.zig
- Verified codebase compliance with Zig 0.16 deprecations
  - No `usingnamespace` usage found
  - No deprecated I/O API usage found
  - No deprecated File API usage found
  - No format method signature issues found

### Code Quality
- Improved type safety with explicit allocator passing
- Enhanced code readability with modern format specifiers
- All tests passing (3/3) with no regressions

## 0.2.1 - 2025-12-27
### Security Fixes
- **CRITICAL**: Fixed path traversal vulnerability in database backup/restore endpoints
  - Added path validation to restrict operations to `backups/` directory
  - Rejects path traversal sequences (`..`), absolute paths, and Windows drive letters
  - CVE-NOT-ASSIGNED (reported 2025-12-27)
  - See SECURITY.md for details

### Bug Fixes
- Fixed memory safety issue in database restore operation
  - Added `errdefer` to ensure proper cleanup on restore failure
  - Prevents memory corruption when database swap fails during restore
  - Makes restore operation atomic (fully succeed or fully fail)

### API Breaking Changes
- Changed `timeout_ms=0` semantics in compute engine
  - Old behavior: `timeout_ms=0` returned `ResultNotFound` after one check
  - New behavior: `timeout_ms=0` immediately returns `EngineError.Timeout`
  - Updated default timeout from 0ms to 1000ms throughout codebase
  - Migration guide: Replace any `timeout_ms=0` with `timeout_ms=1000` for one-second timeout

### Documentation
- Added path validation examples and best practices
- Updated timeout semantics documentation with migration guide
- Added security advisory documentation

## 0.2.0 - 2025-12-24
### High-Performance Compute Runtime
- Implemented work-stealing scheduler with worker thread pool
- Lock-free data structures: Chase-Lev deque, ShardedMap
- Memory management with stable allocator + worker arenas
- Result caching with metadata (worker ID, timestamps, execution duration)
- Support for GPU workloads with feature gating

### GPU Support
- GPU execution path in compute engine with CPU fallback
- GPU workload vtable extending CPU workload ABI
- GPU memory management (GPUBuffer, GPUMemoryPool, AsyncTransfer)
- Backend stubs for CUDA, Vulkan, Metal, WebGPU

### Network Distributed Compute
- Network engine with node registry and discovery
- Task/result serialization (binary format)
- Node management with capability tracking

### Profiling & Metrics
- Thread-safe metrics collector with per-worker statistics
- Performance histograms with configurable buckets
- Execution time tracking and summary reporting

### Benchmarking
- Benchmark framework with timing statistics
- Pre-built workloads: Matrix multiplication, memory allocation, Fibonacci, hashing
- Throughput and latency metrics

### Build System
- Feature flags: `-Denable-gpu`, `-Denable-network`, `-Denable-profiling`
- Conditional compilation for optional modules
- Updated for Zig 0.16 APIs (cmpxchgStrong, std.time.Timer, spinLoopHint)

### Testing
- 10+ integration tests for GPU, network, profiling features
- All tests feature-gated with `error.SkipZigTest`
- 3/3 tests passing in main suite

## 0.1.0 - 2025-12-24
- Streamlined CLI to a minimal, production-ready entrypoint.
- Removed deprecated benchmarks, examples, and legacy test suites.
- Consolidated feature exports and build options for Zig 0.16.
- Updated documentation and project structure.
