# Changelog

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
