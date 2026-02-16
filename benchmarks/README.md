---
title: "Benchmark Suite"
tags: [benchmarks, performance, testing]
---
# ABI Benchmark Suite

Comprehensive performance benchmarks for ABI across core runtime paths, infrastructure services, domain workloads, and competitive comparisons.

## Quick Start

```bash
# Run all benchmark suites
zig build benchmarks

# Run all benchmark suites plus competitive benchmarks
zig build bench-all

# Run a specific suite
zig build benchmarks -- --suite=simd

# Run a reduced CI-safe subset
zig build benchmarks -- --quick

# Emit JSON report and verbose output
zig build benchmarks -- --verbose --json --output=benchmark_results.json

# Run competitive benchmarks
zig build bench-competitive

# Emit competitive benchmark JSON
zig build bench-competitive -- --json
```

---

## Directory Layout

| Path | Purpose |
| --- | --- |
| `benchmarks/` | Suite entry points (`main.zig`, `run.zig`, `mod.zig`) |
| `benchmarks/core/` | Shared benchmark config + vector utilities |
| `benchmarks/domain/` | Domain suites (AI, database, GPU, services) |
| `benchmarks/domain/services/` | Service-level benchmarks (cache, search, messaging, gateway, storage) |
| `benchmarks/infrastructure/` | Infrastructure suites (SIMD, memory, concurrency, crypto, network, v2 modules) |
| `benchmarks/system/` | Framework/system tooling, baselines, and comparator |
| `benchmarks/competitive/` | Industry comparisons (FAISS, vector DBs, LLMs) |
| `benchmarks/baselines/` | Baseline JSON storage (main/branches/releases) |

---

## Available Suites

| Suite | Purpose | Key Metrics |
|-------|---------|-------------|
| **all** | Run all suites | aggregate ops/sec, p50/p90/p99 |
| **simd** | Vectorized compute | ops/sec, throughput (GB/s) |
| **memory** | Allocator patterns | allocs/sec, fragmentation % |
| **concurrency** | Lock-free and parallel primitives | throughput, contention ratio |
| **database** | WDBX/HNSW workloads | insert/search latency (μs), recall |
| **network** | HTTP/JSON/network primitives | req/sec, parse latency (ns) |
| **crypto** | Hashing/encryption/KDF | MB/sec, cycles/byte |
| **ai** | AI/ML operations | GFLOPS, memory bandwidth |
| **gpu** | GPU kernels | kernel time (ns), throughput |
| **v2** | SIMD/Math v2 primitives | primitive throughput |
| **services** | Cache/search/gateway/messaging/storage | request throughput, hit rate |
| **quick** | CI-friendly subset | minimized runtime metrics |

---

## Suite Details

### SIMD Suite (`infrastructure/simd.zig`)

Vectorized primitive performance for small/large payloads:
- Dot product (scalar vs SIMD comparisons)
- Matrix multiplication and accumulation
- L2 norm and cosine similarity
- Distance kernels (Euclidean/Manhattan)

```bash
zig build benchmarks -- --suite=simd
```

### Memory Suite (`infrastructure/memory.zig`)

Allocation and allocator behavior under different pressure patterns:
- General purpose allocator throughput
- Arena allocator hot/cold cycles
- Pool allocator usage behavior
- Fragmentation growth and release overhead

```bash
zig build benchmarks -- --suite=memory
```

### Concurrency Suite (`infrastructure/concurrency.zig`)

Parallel runtime and synchronization throughput:
- Atomic counter throughput under contention
- Lock-free queue and MPSC/MPMC queue patterns
- Work-stealing deque behavior
- Thread-pool scaling

```bash
zig build benchmarks -- --suite=concurrency
```

### Database Suite (`domain/database/`)

Vector database and ANN workload performance:
- Batch and incremental inserts
- Linear search and HNSW retrieval
- Query throughput and p99 latency
- Cache-aligned layout and prefetching impact

```bash
zig build benchmarks -- --suite=database
```

### Network Suite (`infrastructure/network/mod.zig`)

HTTP and serialization path benchmarking:
- Header and query parsing
- JSON encoding/decoding
- URL handling and framing paths
- Request dispatching overhead

```bash
zig build benchmarks -- --suite=network
```

### Crypto Suite (`infrastructure/crypto.zig`)

Cryptographic primitives and RNG paths:
- SHA-256 / SHA-512 hashing
- AES operations
- HMAC
- PBKDF2 / key derivation workloads
- Random number generation

```bash
zig build benchmarks -- --suite=crypto
```

### AI Suite (`domain/ai/`)

AI/ML microbenchmarks used in model and inference paths:
- GEMM and attention kernels
- Activation families (`ReLU`, `GELU`, `SiLU`)
- Softmax and layer norm
- Streaming/throughput-oriented paths

```bash
zig build benchmarks -- --suite=ai
```

### GPU Suite (`domain/gpu/`)

Compute on GPU vs CPU paths:
- Vector and matrix kernels
- Reduction kernels
- Backend and memory-transfer comparisons

```bash
zig build benchmarks -- --suite=gpu
```

### v2 Modules (`infrastructure/v2_modules.zig`)

Modernized infrastructure primitives:
- SIMD activations and primitive matrix operations
- SwissMap and container-like behaviors
- Core primitive throughput and memory impact

```bash
zig build benchmarks -- --suite=v2
```

### Services Suite (`domain/services/`)

Service-level benchmark coverage:
- Cache lookup and storage behavior
- Search index request path
- Gateway dispatch and serialization overhead
- Messaging/storage workflow latency

```bash
zig build benchmarks -- --suite=services
```

### Quick Suite (`--quick`)

Reduces workload to a deterministic, low-duration subset suitable for CI. Use this for fast regressions, then run full suites locally before merging.

```bash
zig build benchmarks -- --quick
```

---

## Competitive Benchmarks

Competitive runs compare ABI against external reference implementations.

```bash
# Run all competitive comparisons
zig build bench-competitive

# Emit JSON for downstream processing
zig build bench-competitive -- --json
```

### Available Comparisons

| Comparison | Target | Metrics |
|------------|--------|---------|
| **FAISS** | Vector similarity search | QPS, recall@k |
| **Vector DBs** | Milvus / Pinecone-style workloads | Insert/search latency |
| **LLM Inference** | llama.cpp | Tokens/sec, memory usage |

Results are emitted in Markdown by default and in JSON when `--json` is provided.

---

## Running Benchmarks

### Command Line Options

```bash
zig build benchmarks -- [OPTIONS]

OPTIONS:
  --suite=<name>    Run specific suite (all, simd, memory, concurrency, database, network, crypto, ai, gpu, v2, services, quick)
  --quick, -q       Use reduced CI workload
  --verbose, -v     Show detailed output
  --json            Output results as JSON to stdout
  --output=<file>   Write JSON report to a file
  --help, -h        Print runner usage
```

`bench-all` runs both `benchmarks` and `bench-competitive` in a single command for a full performance sweep.

### Examples

```bash
# All suites with verbose output
zig build benchmarks -- --verbose

# Run only database suite and emit JSON
zig build benchmarks -- --suite=database --json

# CI-friendly quick suite
zig build benchmarks -- --quick

# Write JSON output for CI
zig build benchmarks -- --output=benchmark_results.json

# Run competitive and dump machine-readable output
zig build bench-competitive -- --json > competitive_results.json
```

---

## Output and Metrics

- **ops/sec**: Operations per second
- **p50/p90/p95/p99**: Latency percentiles
- **MB/sec / GB/sec**: Data throughput
- **GFLOPS**: Floating-point throughput
- **recall@k**: Search quality metric
- **allocs/sec**: Allocation throughput
- **fragmentation %**: Allocator memory fragmentation

---

## Performance Baselines

Baseline reports are stored under `benchmarks/baselines/` (see `benchmarks/baselines/README.md`).
After significant changes, generate a fresh JSON report and commit it to the relevant baseline directory:

```bash
# Generate a new baseline file
zig build benchmarks -- --output=benchmarks/baselines/branches/my_branch.json
```

---

## Adding New Benchmarks

Add benchmarks to the appropriate suite file and use `BenchmarkSuite` for scheduling and reporting.

```zig
const std = @import("std");
const BenchmarkSuite = @import("mod.zig").BenchmarkSuite;

fn matmulBench(allocator: std.mem.Allocator) !void {
    const a = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const b = [_]f32{ 5.0, 6.0, 7.0, 8.0 };

    var out = [_]f32{ 0.0, 0.0, 0.0, 0.0 };
    _ = allocator;

    for (0..1_000) |_| {
        out[0] = a[0] * b[0] + a[1] * b[1];
        std.mem.doNotOptimizeAway(&out);
    }
}

pub fn run(allocator: std.mem.Allocator) !void {
    var suite = BenchmarkSuite.init(allocator);
    defer suite.deinit();

    try suite.runBenchmark("Matrix Multiply Warmup", matmulBench, .{allocator});

    suite.printSummary();
}
```

For richer suites, register multiple `runBenchmark` calls in one file and import that file from the relevant domain/infrastructure module.

---

## Troubleshooting

### Inconsistent Results

- Disable CPU frequency scaling where possible (e.g., `performance` governor)
- Close background applications and browser/IDE workers
- Run at least two passes and compare variance
- Start with `--quick` during setup

### High Variance

- Increase fixture sizes or sample count inside suite configuration
- Check for thermal throttling and memory pressure
- Ensure fixed thread affinity/power settings where available

### Build Failures

```bash
# Enable optional systems explicitly for benchmark paths that require them
zig build benchmarks -Denable-database=true -Denable-gpu=true
```

---

## See Also

- [benchmarks/baselines/README.md](baselines/README.md) - Baseline format and CI flow
- [tools/benchmark-dashboard/README.md](../tools/benchmark-dashboard/README.md) - Dashboarding benchmark output
- [docs/content/gpu.html](../docs/content/gpu.html) - GPU benchmarking guide
- [CLAUDE.md](../CLAUDE.md) - Development guidelines
