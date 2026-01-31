---
title: "Benchmark Suite"
tags: [benchmarks, performance, testing]
---
# ABI Benchmark Suite
> **Codebase Status:** Synced with repository as of 2026-01-31.

<p align="center">
  <img src="https://img.shields.io/badge/Benchmarks-Comprehensive-blue?style=for-the-badge" alt="Benchmarks"/>
  <img src="https://img.shields.io/badge/WDBX-6.1M_ops%2Fsec-success?style=for-the-badge" alt="WDBX"/>
  <img src="https://img.shields.io/badge/LLM-2.8M_tokens%2Fsec-green?style=for-the-badge" alt="LLM"/>
</p>

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

## Directory Layout

| Path | Purpose |
| --- | --- |
| `benchmarks/` | Suite entry points (`main.zig`, `run.zig`, `mod.zig`) |
| `benchmarks/core/` | Core utilities (config, vectors, distance) |
| `benchmarks/domain/` | Feature-domain suites (ai, database, gpu) |
| `benchmarks/infrastructure/` | Infra suites (concurrency, crypto, memory, simd, network) |
| `benchmarks/system/` | System/integration baselines (framework, CI, baselines) |
| `benchmarks/competitive/` | Competitive comparisons (FAISS, vector DBs, LLMs) |
| `benchmarks/baselines/` | Regression baselines and comparisons |
| `benchmarks/run_competitive.zig` | CLI entry point for competitive runs |
<<<<<<< HEAD
| `benchmarks/system/industry_standard.zig` | Industry-standard baseline harness |
=======
>>>>>>> origin/cursor/ai-module-source-organization-0282

---

## Available Suites

| Suite | Purpose | Key Metrics |
|-------|---------|-------------|
<<<<<<< HEAD
| **simd** | Vector operations | ops/sec, throughput (GB/s) |
| **memory** | Allocator patterns | allocs/sec, fragmentation % |
| **concurrency** | Lock-free structures | ops/sec, contention ratio |
| **database** | WDBX operations | insert/search latency (μs) |
| **network** | HTTP/JSON parsing | req/sec, parse time (ns) |
| **crypto** | Hash/encrypt ops | MB/sec, cycles/byte |
| **ai** | GEMM/attention | GFLOPS, memory bandwidth |
| **gpu** | Kernel and backend checks | throughput, availability |
=======
| **simd** | Vector operations (`infrastructure/simd.zig`) | ops/sec, throughput (GB/s) |
| **memory** | Allocator patterns (`infrastructure/memory.zig`) | allocs/sec, fragmentation % |
| **concurrency** | Lock-free structures (`infrastructure/concurrency.zig`) | ops/sec, contention ratio |
| **database** | WDBX ops (`domain/database/*.zig`) | insert/search latency (μs) |
| **network** | HTTP/JSON parsing (`infrastructure/network.zig`) | req/sec, parse time (ns) |
| **crypto** | Hash/encrypt ops (`infrastructure/crypto.zig`) | MB/sec, cycles/byte |
| **ai** | Kernels + LLM metrics (`domain/ai/*.zig`) | GFLOPS, tokens/sec |
| **gpu** | Backend kernels (`domain/gpu/*.zig`) | ops/sec, transfer MB/s |
>>>>>>> origin/cursor/ai-module-source-organization-0282
| **quick** | Fast verification | subset of all suites |

---

## Suite Details

### SIMD Suite (`infrastructure/simd.zig`)

Tests vectorized operations using SIMD intrinsics:
- Dot product (single/batch)
- Matrix multiplication
- L2 norm computation
- Cosine similarity
- Distance calculations (Euclidean, Manhattan)

```bash
zig build benchmarks -- --suite=simd
```

### Memory Suite (`infrastructure/memory.zig`)

Measures allocator performance:
- General purpose allocator throughput
- Arena allocator patterns
- Pool allocator efficiency
- Fragmentation under stress
- Memory pressure handling

```bash
zig build benchmarks -- --suite=memory
```

### Concurrency Suite (`infrastructure/concurrency.zig`)

Tests lock-free data structures:
- Lock-free queue throughput
- Work-stealing deque performance
- Atomic counter operations
- MPMC queue contention
- Thread pool scaling

```bash
zig build benchmarks -- --suite=concurrency
```

### Database Suite (`domain/database/operations.zig`)

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

### Network Suite (`infrastructure/network.zig`)

Network protocol benchmarks:
- HTTP header parsing
- JSON encoding/decoding
- WebSocket frame processing
- Request routing overhead

```bash
zig build benchmarks -- --suite=network
```

### Crypto Suite (`infrastructure/crypto.zig`)

Cryptographic operation benchmarks:
- SHA-256/SHA-512 hashing
- AES-256 encryption
- HMAC computation
- Key derivation (PBKDF2, Argon2)
- Random number generation

```bash
zig build benchmarks -- --suite=crypto
```

### AI Suite (`domain/ai/kernels.zig`, `domain/ai/llm_metrics.zig`)

Machine learning operation benchmarks:
- GEMM (General Matrix Multiply)
- Attention mechanism
- Activation functions (ReLU, GELU, SiLU)
- Softmax computation
- Layer normalization

```bash
zig build benchmarks -- --suite=ai
```

<<<<<<< HEAD
### GPU Suite (`gpu.zig`)

GPU-related benchmarks:
- Backend availability checks
- Kernel throughput comparisons
- GPU vs CPU baselines
=======
### GPU Suite (`domain/gpu/backends.zig`, `domain/gpu/kernels.zig`)

GPU backend and kernel benchmarks:
- Backend comparison (throughput + latency)
- Kernel execution (matmul, elementwise, reductions)
- GPU vs CPU comparisons
>>>>>>> origin/cursor/ai-module-source-organization-0282

```bash
zig build benchmarks -- --suite=gpu
```

---

## Competitive Benchmarks

Compare ABI performance against industry-standard implementations:

```bash
# Run competitive benchmarks
zig build bench-competitive

# With custom dataset size
zig build bench-competitive -- --vectors=100000 --dims=768
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
  --suite=<name>    Run specific suite (simd, memory, concurrency, database, network, crypto, ai, gpu)
<<<<<<< HEAD
  --quick           Run quick subset for CI
=======
  --quick           Run with reduced iterations
>>>>>>> origin/cursor/ai-module-source-organization-0282
  --verbose         Show detailed output
  --json            Output results as JSON to stdout
  --output=<file>   Output results as JSON to a file
```

### Examples

```bash
# All suites with verbose output
zig build benchmarks -- --verbose

# Database benchmarks only
zig build benchmarks -- --suite=database

# Quick verification run
zig build benchmarks -- --quick

# JSON output for CI integration
zig build benchmarks -- --json > benchmark_results.json

# Write JSON output to a file
zig build benchmarks -- --output=benchmark_results.json
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

Baselines are stored under `benchmarks/baselines/`. See
`benchmarks/baselines/README.md` for format and storage conventions.

```bash
# Generate new baseline results
zig build benchmarks -- --output=benchmarks/baselines/branches/local.json

# Compare using the baseline comparator utilities
zig test benchmarks/system/baseline_comparator.zig
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

- [benchmarks/baselines/README.md](baselines/README.md) - Baseline format and workflow
- [GPU Docs](../docs/content/gpu.html) - GPU-specific benchmarking
- [CLAUDE.md](../CLAUDE.md) - Development guidelines
