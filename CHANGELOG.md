# Changelog

## 0.3.0 - 2025-12-27

### Major Features

**GPU Backend Implementation**
- Added complete GPU backend support with CUDA, Vulkan, Metal, and WebGPU
- Implemented kernel compilation and execution framework (`src/compute/gpu/kernels.zig`)
- Created backend-specific implementations:
  - CUDA backend (`src/compute/gpu/backends/cuda.zig`) with simulation layer for graceful fallback
  - Vulkan backend (`src/compute/gpu/backends/vulkan.zig`) with shader module support
  - Metal backend (`src/compute/gpu/backends/metal.zig`) with compute pipeline
  - WebGPU backend (`src/compute/gpu/backends/webgpu.zig`) with compute pipeline
- Added default kernels for each backend: vector_add, matmul, reduce_sum
- Implemented stream synchronization primitives
- Added device memory management with tracking
- Support for GPU workload hints and preferences

**Async I/O Support**
- Implemented full async HTTP client using `std.http.Client` (`src/shared/utils/http/async_http.zig`)
- Added streaming response support with `StreamingResponse` type
- Methods: `fetch()`, `fetchJson()`, `get()`, `post()`, `postJson()`, `fetchStreaming()`
- Bearer token authentication support
- Redirect following with configurable limits
- Request/response timeout management

**Connector Enhancements**
- Completed OpenAI connector with full JSON decoding (`src/features/connectors/openai.zig`)
- Completed Ollama connector with generate and chat APIs
- Completed HuggingFace connector with inference API
- All connectors support environment variable configuration
- Added streaming support to OpenAI connector
- Proper error handling (rate limits, model loading, API failures)
- JSON response parsing for all services

**JSON Utilities**
- Created comprehensive JSON encoding/decoding utilities (`src/shared/utils/json/mod.zig`)
- Functions: `parseString()`, `parseNumber()`, `parseInt()`, `parseUint()`, `parseBool()`
- Field helpers: `parseStringField()`, `parseOptionalStringField()`, `parseNumberField()`, etc.
- String escaping: `escapeString()` for JSON output
- Type-safe JSON parsing with proper error handling

**NUMA and CPU Affinity**
- Implemented NUMA topology detection (`src/compute/runtime/numa.zig`)
- Platform-specific detection: Linux (/sys/devices/system/cpu), Windows, generic fallback
- CPU topology structure with NUMA nodes
- `AffinityMask` struct with bit-level CPU selection
- Thread affinity control: `setThreadAffinity()`, `setThreadAffinityMask()`
- Integration with compute runtime engine
- NUMA-aware task scheduling support

**Property-Based Testing**
- Created property testing framework (`tests/property_tests.zig`)
- `PropertyTest` struct with test case tracking
- `PropertyTestConfig` with configurable max cases, size, seed
- Assertion helpers: `assertEq()`, `assertLessThan()`, `assertGreaterThan()`, `assertContains()`, `assertLength()`
- Random input generation using `std.Random.DefaultPrng`
- Test result reporting with pass/failure statistics

**Distributed Scheduling**
- Implemented task scheduler (`src/features/network/scheduler.zig`)
- Load balancing strategies: round_robin, least_loaded, random, affinity_based
- `TaskScheduler` struct with node management
- Priority queue system: low, normal, high, critical
- Node health tracking with CPU count and active tasks
- Task state management: pending, scheduled, running, completed, failed, cancelled
- Exported from network module for integration

**High Availability Mechanisms**
- Implemented health check system (`src/features/network/ha.zig`)
- `HealthCheck` struct with node health tracking
- Health states: healthy, unhealthy, degraded, unknown
- Cluster state management: forming, stable, unstable, partitioned
- Automatic failover support
- Leader election mechanism
- Configurable health check intervals and timeouts
- Multi-node cluster coordination

**C API and Language Bindings**
- Created C-compatible API (`bindings/c_api.zig`)
- Error codes and log levels
- Memory management functions
- CPU and NUMA information access
- Python bindings stub (`bindings/python/abi.py`) with ctypes
- `AbiFramework` class with context management
- JavaScript/WebAssembly bindings (`bindings/wasm/abi_wasm.zig`)
- Emscripten-compatible exports for browser and Node.js
- Memory management: malloc, free, realloc, memset, memcpy
- String utilities: strlen, strcmp, strcpy
- Vector operations: add, dot, L2 norm, cosine similarity

### API Changes

### Breaking Changes

None. All changes are backward compatible.

### Bug Fixes

None.

### Code Quality

- Improved error handling across all modules
- Enhanced memory safety with explicit allocator passing
- Better resource cleanup with proper deinit patterns
- Comprehensive test coverage for all new features

### Documentation

Updated all feature documentation with new APIs and examples.

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
