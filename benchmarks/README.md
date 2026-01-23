---
title: "README"
tags: []
---
# ABI Benchmark Suite
> **Codebase Status:** Synced with repository as of 2026-01-22.

Comprehensive performance benchmarks for the ABI framework, measuring throughput, latency, and resource utilization across all major subsystems.

## Quick Start

```bash
# Run all benchmark suites
zig build benchmarks

# Run specific suite
zig build benchmarks -- --suite=simd

# Quick mode (reduced iterations)
zig build benchmarks -- --quick

# Verbose output
zig build benchmarks -- --verbose
```

---

## Available Suites

| Suite | Purpose | Key Metrics |
|-------|---------|-------------|
| **simd** | Vector operations | ops/sec, throughput (GB/s) |
| **memory** | Allocator patterns | allocs/sec, fragmentation % |
| **concurrency** | Lock-free structures | ops/sec, contention ratio |
| **database** | WDBX operations | insert/search latency (μs) |
| **network** | HTTP/JSON parsing | req/sec, parse time (ns) |
| **crypto** | Hash/encrypt ops | MB/sec, cycles/byte |
| **ai** | GEMM/attention | GFLOPS, memory bandwidth |
| **quick** | Fast verification | subset of all suites |

---

## Suite Details

### SIMD Suite (`simd.zig`)

Tests vectorized operations using SIMD intrinsics:
- Dot product (single/batch)
- Matrix multiplication
- L2 norm computation
- Cosine similarity
- Distance calculations (Euclidean, Manhattan)

```bash
zig build benchmarks -- --suite=simd
```

### Memory Suite (`memory.zig`)

Measures allocator performance:
- General purpose allocator throughput
- Arena allocator patterns
- Pool allocator efficiency
- Fragmentation under stress
- Memory pressure handling

```bash
zig build benchmarks -- --suite=memory
```

### Concurrency Suite (`concurrency.zig`)

Tests lock-free data structures:
- Lock-free queue throughput
- Work-stealing deque performance
- Atomic counter operations
- MPMC queue contention
- Thread pool scaling

```bash
zig build benchmarks -- --suite=concurrency
```

### Database Suite (`database.zig`)

WDBX vector database benchmarks:
- Vector insertion (single/batch)
- Linear search performance
- HNSW approximate search
- Concurrent search operations
- Cache-aligned memory access
- Memory prefetching effectiveness

```bash
zig build benchmarks -- --suite=database
```

### Network Suite (`network.zig`)

Network protocol benchmarks:
- HTTP header parsing
- JSON encoding/decoding
- WebSocket frame processing
- Request routing overhead

```bash
zig build benchmarks -- --suite=network
```

### Crypto Suite (`crypto.zig`)

Cryptographic operation benchmarks:
- SHA-256/SHA-512 hashing
- AES-256 encryption
- HMAC computation
- Key derivation (PBKDF2, Argon2)
- Random number generation

```bash
zig build benchmarks -- --suite=crypto
```

### AI Suite (`ai.zig`)

Machine learning operation benchmarks:
- GEMM (General Matrix Multiply)
- Attention mechanism
- Activation functions (ReLU, GELU, SiLU)
- Softmax computation
- Layer normalization

```bash
zig build benchmarks -- --suite=ai
```

---

## Competitive Benchmarks

Compare ABI performance against industry-standard implementations:

```bash
# Run competitive benchmarks
zig build bench-competitive

# With custom dataset size
zig build run-competitive -- --vectors=100000 --dims=768
```

### Available Comparisons

| Comparison | Target | Metrics |
|------------|--------|---------|
| **FAISS** | Vector similarity search | QPS, recall@k |
| **Vector DBs** | Milvus, Pinecone | Insert/search latency |
| **LLM Inference** | llama.cpp | Tokens/sec, memory usage |

Results are output as JSON for easy integration with CI/CD pipelines.

---

## Running Benchmarks

### Command Line Options

```
zig build benchmarks -- [OPTIONS]

OPTIONS:
  --suite=<name>    Run specific suite (simd, memory, concurrency, database, network, crypto, ai)
  --quick           Run with reduced iterations
  --verbose         Show detailed output
  --json            Output results as JSON
  --iterations=<n>  Override default iteration count
```

### Examples

```bash
# All suites with verbose output
zig build benchmarks -- --verbose

# Database benchmarks with more iterations
zig build benchmarks -- --suite=database --iterations=10000

# Quick verification run
zig build benchmarks -- --quick

# JSON output for CI integration
zig build benchmarks -- --json > benchmark_results.json
```

---

## Understanding Results

### Throughput Metrics

- **ops/sec**: Operations per second (higher is better)
- **MB/sec** or **GB/sec**: Data throughput (higher is better)
- **GFLOPS**: Billion floating-point operations per second

### Latency Metrics

- **μs** (microseconds): 1/1,000,000 second
- **ns** (nanoseconds): 1/1,000,000,000 second
- **p50/p99**: Percentile latencies

### Memory Metrics

- **RSS**: Resident Set Size (actual memory usage)
- **fragmentation %**: Wasted memory due to allocation patterns
- **allocs/sec**: Allocation rate

---

## Performance Baseline

The framework maintains a performance baseline in `docs/PERFORMANCE_BASELINE.md`. To update after significant changes:

```bash
# Generate new baseline
zig build benchmarks -- --json > docs/baseline_new.json

# Compare with existing
diff docs/PERFORMANCE_BASELINE.md docs/baseline_new.json
```

---

## Adding New Benchmarks

New benchmarks should follow this pattern:

```zig
const BenchmarkSuite = @import("mod.zig").BenchmarkSuite;

pub fn run(allocator: std.mem.Allocator) !void {
    var suite = BenchmarkSuite.init(allocator, "My Suite");
    defer suite.deinit();

    suite.benchmark("operation_name", struct {
        fn bench() void {
            // Operation to benchmark
        }
    }.bench, .{});

    suite.report();
}
```

---

## Troubleshooting

### Inconsistent Results

- Disable CPU frequency scaling: `sudo cpupower frequency-set -g performance`
- Close background applications
- Run multiple iterations and average
- Use `--quick` for initial verification

### High Variance

- Increase iteration count with `--iterations=N`
- Check for thermal throttling
- Ensure consistent memory pressure

### Build Failures

```bash
# Ensure all dependencies are available
zig build benchmarks -Denable-database=true -Denable-gpu=true
```

---

## See Also

- [docs/PERFORMANCE_BASELINE.md](../docs/PERFORMANCE_BASELINE.md) - Reference performance metrics
- [docs/gpu.md](../docs/gpu.md) - GPU-specific benchmarking
- [CLAUDE.md](../CLAUDE.md) - Development guidelines

