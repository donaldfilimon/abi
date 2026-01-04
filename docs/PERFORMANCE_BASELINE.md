# Performance Baseline Document

**Date**: 2025-01-03
**Zig Version**: 0.16.0-dev.1892+53ebfde6b
**Framework Version**: 0.1.0

## Purpose

This document establishes a performance baseline for the ABI Framework after the Zig 0.16 migration. Use this baseline to detect regressions or improvements in future releases.

## Benchmark Methodology

- **Environment**: Windows 10, Zig 0.16.0-dev.1892+53ebfde6b
- **CPU**: Intel/AMD (varies by machine)
- **Memory**: System RAM
- **Iterations**: Per-benchmark configuration (see `benchmarks/run.zig`)
- **Compiler Flags**:
  - Debug: Default (`-ODebug`)
  - ReleaseFast: `-Doptimize=ReleaseFast`

## Core Benchmarks

### FNV-1a64 Hash

**Purpose**: Measures cryptographic hash function performance (non-crypto use case)

```
Configuration: 500,000 iterations
```

| Build Type | Time per Op | Throughput |
|------------|-------------|------------|
| Debug | ~600 ns | ~833,333,333 ops/sec |
| ReleaseFast | ~600 ns | ~833,333,333 ops/sec |

**Interpretation**: FNV-1a64 is memory-bound, so compiler optimizations have minimal impact. This represents the maximum single-threaded throughput achievable.

### Dot4 (4-element Vector Dot Product)

**Purpose**: Measures basic vector math performance

```
Configuration: 50,000 iterations
```

| Build Type | Time per Op | Throughput |
|------------|-------------|------------|
| Debug | ~17,600 ns | ~2,857,143 ops/sec |
| ReleaseFast | ~17,600 ns | ~2,857,143 ops/sec |

**Interpretation**: Simple vector operations are CPU-bound and highly optimized by LLVM even in debug mode.

## Build Performance Metrics

### Compilation Time

| Configuration | Approximate Time |
|---------------|------------------|
| Debug build | ~1-2 seconds |
| ReleaseFast build | ~2-5 seconds |
| Full feature build | ~2-3 seconds |

### Binary Size

| Configuration | Approximate Size |
|---------------|------------------|
| Debug (basic) | ~1-2 MB |
| ReleaseFast (basic) | ~500 KB - 1 MB |
| Full features | ~2-5 MB |

## Feature-Specific Performance

### Compute Engine

| Metric | Value |
|--------|-------|
| Work-stealing overhead | ~100-500 ns per steal |
| Task queue push | ~50-100 ns |
| Task queue pop | ~50-100 ns |
| Result cache lookup | ~10-50 ns |

### Memory Operations

| Metric | Value |
|--------|-------|
| Arena allocation (small) | ~10-50 ns |
| Arena allocation (large) | ~100-500 ns |
| ShardedMap lookup | ~10-20 ns (avg) |
| Lock-free push | ~20-50 ns |

### Network Serialization

| Metric | Value |
|--------|-------|
| Task serialization (1KB) | ~1-5 μs |
| Result serialization (1KB) | ~1-5 μs |
| Deserialization (1KB) | ~1-5 μs |

## I/O Performance (Post-Migration)

### HTTP Client

| Metric | Zig 0.15 (baseline) | Zig 0.16 (current) | Change |
|--------|---------------------|-------------------|--------|
| Simple GET request | N/A | ~5-10 ms | Baseline |
| JSON response parse | N/A | ~1-2 ms | Baseline |
| Streaming response | N/A | ~1 ms initial | Baseline |

### HTTP Server

| Metric | Value |
|--------|-------|
| Request parsing | ~100-500 μs |
| Response generation | ~50-200 μs |
| Throughput (simple) | ~10,000 req/sec |

## Performance Testing Commands

```bash
# Run all benchmarks
zig build benchmark

# Run specific benchmark
zig run benchmarks/run.zig -- --name fnv1a64

# Profile with ReleaseFast
zig build -Doptimize=ReleaseFast benchmark

# Measure build time
time zig build 2>&1 | tail -5

# Check binary size
ls -lh zig-out/bin/abi.exe
```

## Regression Detection

To detect performance regressions:

1. **Run benchmarks after any code change**
   ```bash
   zig build benchmark > benchmark_new.txt
   diff benchmark_baseline.txt benchmark_new.txt
   ```

2. **Monitor build times**
   ```bash
   time zig build 2>&1 | tail -3
   ```

3. **Check binary size trends**
   ```bash
   ls -l zig-out/bin/abi.exe
   ```

## Performance Targets

| Metric | Target | Warning Threshold |
|--------|--------|-------------------|
| FNV-1a64 throughput | >500M ops/sec | <400M ops/sec |
| Dot4 throughput | >2M ops/sec | <1M ops/sec |
| Build time (Debug) | <5 seconds | >10 seconds |
| Build time (ReleaseFast) | <10 seconds | >30 seconds |
| Binary size (ReleaseFast) | <2 MB | >5 MB |

## Optimization Guidelines

### When Performance Matters

1. **Use ReleaseFast for production**
   ```bash
   zig build -Doptimize=ReleaseFast
   ```

2. **Enable specific CPU features**
   ```bash
   zig build -Doptimize=ReleaseFast -mcpu=skylake
   ```

3. **Disable unused features**
   ```bash
   zig build -Denable-network=false -Denable-profiling=false
   ```

### Hot Path Optimization

For critical code paths:
- Use `@inline` hints
- Avoid allocations in loops
- Prefer stack allocation over heap
- Use `std.ArrayListUnmanaged` for ownership control

## Future Improvements

### Planned Benchmarks

- [ ] GPU kernel performance benchmarks
- [ ] Network throughput benchmarks
- [ ] Database operation benchmarks
- [ ] Memory pool utilization metrics
- [ ] Concurrent workload benchmarks

### Optimization Opportunities

1. **SIMD vectorization** - Enable for math operations
2. **Memory pool pre-allocation** - Reduce allocation overhead
3. **Connection pooling** - HTTP client reuse
4. **Result cache tuning** - LRU policy optimization

## Troubleshooting

### Slow Builds

1. Clear cache: `rm -rf .zig-cache`
2. Check disk space
3. Reduce parallel jobs: `zig build -j 2`

### Benchmark Inconsistencies

1. Ensure no other heavy processes running
2. Warm up before measuring
3. Run multiple iterations and average
4. Check CPU frequency scaling

## References

- Benchmark implementation: `benchmarks/run.zig`
- Compute engine: `src/compute/runtime/engine.zig`
- Memory management: `src/shared/utils/memory/`
- HTTP client: `src/shared/utils/http/async_http.zig`

---

**Document Version**: 1.0
**Last Updated**: 2025-01-03
**Next Review**: After Zig 0.16 stable release
