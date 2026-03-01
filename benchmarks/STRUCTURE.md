# Benchmarks Directory Structure

```
benchmarks/
├── README.md                     # Overview and usage
├── STRUCTURE.md                  # Directory structure overview
├── main.zig                      # Main benchmark runner (zig build benchmarks)
├── mod.zig                       # Root benchmark module (exports suites)
├── run.zig                       # Legacy CLI wrapper
├── run_competitive.zig           # Competitive benchmark runner
├── baselines/                    # Baseline JSON storage
│   ├── README.md
│   ├── main/                     # Main branch baselines
│   ├── branches/                 # Feature branch baselines
│   └── releases/                 # Release snapshots
│
├── baselines/                    # Baseline storage + CI regression data
├── competitive/                  # Industry comparison benchmarks
│   ├── mod.zig
│   ├── faiss_comparison.zig
│   ├── llm_comparison.zig
│   └── vector_db_comparison.zig
│
├── core/                         # Core framework benchmarks/utilities
│   ├── mod.zig
│   ├── config.zig
│   ├── distance.zig
│   └── vectors.zig
│
├── domain/                       # Feature-specific benchmarks
│   ├── ai/
│   │   ├── mod.zig
│   │   ├── fpga_kernels.zig
│   │   ├── kernels.zig
│   │   ├── llm_metrics.zig
│   │   └── streaming.zig
│   ├── database/
│   │   ├── mod.zig
│   │   ├── ann_benchmarks.zig
│   │   ├── hnsw.zig
│   │   └── operations.zig
│   └── gpu/
│       ├── mod.zig
│       ├── backends.zig
│       ├── gpu_vs_cpu.zig
│       └── kernels.zig
│
├── infrastructure/               # System-level benchmarks
│   ├── mod.zig
│   ├── concurrency.zig
│   ├── crypto.zig
│   ├── gpu_backends.zig
│   ├── memory.zig
│   ├── network.zig
│   └── simd.zig
│
└── system/                       # Framework + baseline tooling
    ├── mod.zig
    ├── baseline_store.zig
    ├── baseline_comparator.zig
    └── framework.zig
```

## Purpose of Each Directory

### `competitive/` - Industry Comparisons
Compare ABI Framework against industry-standard implementations:
- **FAISS**: Vector similarity search libraries
- **LLM frameworks**: Inference speed comparisons
- **Vector databases**: QPS and latency comparisons

### `baselines/` - Baseline Storage
Stores benchmark snapshots for releases, branches, and regression tracking.

### `core/` - Framework Core Performance
Measure fundamental framework operations:
- Configuration loading speed
- Vector math operations
- Distance/similarity calculations

### `domain/` - Feature-Specific Benchmarks
Domain-specific performance testing:
- **AI/ML**: Streaming, kernels, LLM metrics, FPGA kernels
- **Database**: ANN algorithms, HNSW, CRUD operations
- **GPU**: Kernel performance, backend comparisons, GPU vs CPU
We use `domain/*` instead of root-level category folders for consistency.

### `infrastructure/` - System Infrastructure
Infrastructure component performance:
- Concurrency and parallelism
- Cryptographic operations
- Memory management strategies
- Network parsing/JSON throughput
- SIMD/vectorization effectiveness
- Network parsing and request handling
- GPU backend enumeration and setup costs

### `system/` - Framework & Baseline Tooling
System-level testing and regression tracking:
- Baseline storage/comparison for regressions
- Framework startup/shutdown
- Baseline storage and comparison tooling

### `baselines/` - Baseline Storage
Baseline JSON reports organized by main/branches/releases.

## Benchmark Types

1. **Throughput Benchmarks**: Ops/sec, QPS, tokens/sec
2. **Latency Benchmarks**: μs/op, p50/p95/p99 latencies
3. **Memory Benchmarks**: Allocation patterns, fragmentation
4. **Accuracy Benchmarks**: Precision/recall, error rates (competitive)
5. **Scalability Benchmarks**: Performance at different scales

## Adding New Benchmarks

### 1. Determine Category
- Core framework operation → `core/`
- AI/ML operation → `domain/ai/`
- Database operation → `domain/database/`
- GPU operation → `domain/gpu/`
- System/infrastructure operation → `infrastructure/` or `system/`

### 2. Create Benchmark File
```zig
// benchmarks/domain/ai/new_benchmark.zig
const std = @import("std");
const BenchmarkSuite = @import("../../mod.zig").BenchmarkSuite;

pub const suite = BenchmarkSuite.init("New AI Benchmark", .{
    .description = "Description of what this benchmark measures",
    .category = .ai,
});

pub fn run(allocator: std.mem.Allocator) !void {
    var bench = suite.create(allocator);
    defer bench.deinit();
    
    bench.test("operation_name", struct {
        fn run(_: *@This()) !void {
            // Benchmark operation
        }
    }.run, .{});
    
    try bench.execute();
}
```

### 3. Register in Domain mod.zig
```zig
// benchmarks/domain/ai/mod.zig
pub const suites = .{
    @import("kernels.zig").suite,
    @import("llm_metrics.zig").suite,
    @import("new_benchmark.zig").suite,  // Add new benchmark
};
```

### 4. Run Verification
```bash
# Run all benchmarks
zig build benchmarks

# Run all AI benchmarks
zig build benchmarks -- --suite=ai

# Full suite verification
zig build test --summary all
```

## Best Practices

### 1. Realistic Data
- Use production-like data sizes
- Include edge cases and realistic distributions
- Avoid synthetic data where possible

### 2. Fair Comparisons
- Compare against similar configurations
- Document any differences in setup
- Include statistical significance analysis

### 3. Resource Management
- Clean up all resources after each iteration
- Use no-op iterations for warm-up
- Measure memory usage and cleanup

### 4. Documentation
- Document what the benchmark measures
- Include expected performance ranges
- Note any assumptions or limitations

## Running Benchmarks

### All Benchmarks
```bash
zig build benchmarks
```

### Specific Category
```bash
zig build benchmarks -- --suite=simd
zig build benchmarks -- --suite=ai
zig build benchmarks -- --suite=database
```

### Competitive Benchmarks
```bash
zig build bench-competitive
```

### Individual Benchmark
```bash
zig build benchmarks -- --suite=ai
```

## Output Formats

### Human-readable
```bash
zig build benchmarks  # Default format
```

### JSON
```bash
zig build benchmarks -- --output=results.json
```

### JSON to stdout
```bash
zig build benchmarks -- --json
```

### CI Integration
```bash
# With version tagging
zig build benchmarks -- --output=results.json
```

## Performance Baselines

Baseline files are stored in `benchmarks/baselines/` (see `benchmarks/baselines/README.md`).
Use the baseline store and comparator APIs in `benchmarks/system/` to compare
current runs against saved baselines.