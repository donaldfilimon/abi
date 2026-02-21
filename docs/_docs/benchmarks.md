---
title: "Benchmarks"
description: "Built-in performance benchmark suite"
section: "Operations"
order: 5
---

# Benchmarks

The benchmarks module provides a built-in performance measurement framework
covering SIMD, memory, concurrency, database, network, cryptography, AI/ML
inference, GPU operations, and service-level benchmarks. Results can be
compared against stored baselines for regression detection.

- **Build flag:** `-Denable-benchmarks=true` (default: enabled)
- **Namespace:** `abi.benchmarks`
- **Source:** `src/features/benchmarks/` (feature module) and `benchmarks/` (suite root)

## Overview

The benchmark system is split into two parts:

1. **Feature module** (`src/features/benchmarks/`) -- Provides a `Context` for Framework integration and the `isEnabled()` gate. Configured via `BenchmarksConfig` with warmup iterations, sample iterations, and JSON export options.

2. **Benchmark suite** (`benchmarks/`) -- The actual benchmark implementations organized into four layers:
   - **Infrastructure** -- SIMD, memory allocators, concurrency primitives, network (HTTP, WebSocket, connection pool, URL parsing), and cryptography
   - **Domain** -- Database (HNSW, ANN, vector operations), AI (LLM metrics, kernel benchmarks, streaming, FPGA), GPU (backends, CPU baselines, vector ops, matmul, memory ops), and services (cache, gateway, messaging, search, storage)
   - **Competitive** -- Comparisons against FAISS, other vector databases, and LLM serving frameworks
   - **System** -- Baseline storage, comparison framework, and regression detection

## Running Benchmarks

### Build Commands

```bash
# Run the main benchmark suite
zig build benchmarks

# Run all benchmark suites (comprehensive)
zig build bench-all

# Run benchmarks with a specific suite filter
zig build benchmarks -- --suite=simd
zig build benchmarks -- --suite=memory
zig build benchmarks -- --suite=concurrency
zig build benchmarks -- --suite=database
zig build benchmarks -- --suite=network
zig build benchmarks -- --suite=crypto
zig build benchmarks -- --suite=ai
zig build benchmarks -- --suite=gpu
zig build benchmarks -- --suite=v2

# Show help
zig build benchmarks -- --help
```

### CLI Commands

The CLI provides benchmark access through nested subcommands:

```bash
# Run benchmarks via CLI
zig build run -- bench                    # Run default suite
zig build run -- bench list               # List available suites
zig build run -- bench micro hash         # Run micro-benchmark: hashing
zig build run -- bench micro alloc        # Run micro-benchmark: allocation
```

## API Reference

### Core Types

| Type | Description |
|------|-------------|
| `Config` (alias for `BenchmarksConfig`) | Configuration for warmup, samples, and export |
| `Context` | Framework integration context |
| `BenchmarksError` | Error set: `FeatureDisabled`, `OutOfMemory`, `InvalidConfig`, `BenchmarkFailed` |

### Configuration

```zig
const config = abi.benchmarks.Config{
    .warmup_iterations = 3,       // Warmup runs before timing
    .sample_iterations = 10,      // Timed sample runs
    .export_json = false,         // Export results as JSON
    .output_path = null,          // JSON output file path
};

var ctx = try abi.benchmarks.Context.init(allocator, config);
defer ctx.deinit();
```

## Benchmark Suite Structure

```
benchmarks/
  main.zig                          Entry point for `zig build benchmarks`
  run.zig                           Suite runner
  run_competitive.zig               Competitive comparison runner
  mod.zig                           Top-level module
  core/
    config.zig                      Benchmark configuration
    distance.zig                    Distance function benchmarks
    vectors.zig                     Vector operation benchmarks
    mod.zig
  infrastructure/
    simd.zig                        SIMD instruction benchmarks
    memory.zig                      Allocator benchmarks
    concurrency.zig                 Thread pool, channel benchmarks
    crypto.zig                      Hashing, encryption benchmarks
    v2_modules.zig                  v2 primitive benchmarks
    gpu_backends.zig                GPU backend benchmarks
    network/
      http.zig                      HTTP benchmarks
      websocket.zig                 WebSocket benchmarks
      pool.zig                      Connection pool benchmarks
      url.zig                       URL parsing benchmarks
  domain/
    ai/
      kernels.zig                   AI kernel benchmarks
      llm_metrics.zig               LLM latency/throughput
      streaming.zig                 Streaming inference benchmarks
      fpga_kernels.zig              FPGA kernel benchmarks
    database/
      hnsw.zig                      HNSW index benchmarks
      ann_benchmarks.zig            ANN recall/latency
      operations.zig                CRUD operation benchmarks
    gpu/
      backends.zig                  GPU backend comparison
      gpu_vs_cpu.zig                GPU vs CPU comparison
      kernels/                      Kernel-level benchmarks
    services/
      cache.zig                     Cache hit/miss benchmarks
      gateway.zig                   API gateway routing benchmarks
      messaging.zig                 Message queue benchmarks
      search.zig                    Full-text search benchmarks
      storage.zig                   Storage I/O benchmarks
  competitive/
    faiss_comparison.zig            vs FAISS vector search
    llm_comparison.zig              vs other LLM frameworks
    vector_db_comparison.zig        vs other vector databases
  system/
    framework.zig                   Benchmark framework utilities
    baseline_store.zig              Baseline result storage
    baseline_comparator.zig         Regression detection
```

## Profiler Integration

The benchmarks module integrates with the observability profiler for detailed
performance analysis:

```zig
const profiler = abi.shared.utils.profiler;

var p = profiler.Profiler.init(allocator);
defer p.deinit();

p.start("my_operation");
// ... code to benchmark ...
p.stop("my_operation");

const report = p.report();
```

## Baseline Comparison

The system framework supports storing and comparing benchmark results against
baselines to detect performance regressions:

```bash
# Run benchmarks and store as baseline
zig build benchmarks -- --save-baseline

# Run benchmarks and compare against stored baseline
zig build benchmarks -- --compare-baseline
```

## Disabling at Build Time

```bash
zig build -Denable-benchmarks=false
```

When disabled, `abi.benchmarks.isEnabled()` returns `false` and the `Context`
uses a stub that returns `error.FeatureDisabled`. The benchmark CLI commands
remain registered but report that the feature is disabled.

## Related

- [Observability](observability.html) -- Profiler and metrics used by benchmarks
- [Analytics](analytics.html) -- Event tracking for benchmark result reporting
