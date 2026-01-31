# Benchmarks Directory Structure

```
benchmarks/
├── README.md                     # This file
├── STRUCTURE.md                  # Directory structure overview
├── main.zig                      # Main benchmark runner
├── mod.zig                       # Root benchmark module (exports all suites)
├── run.zig                       # CLI wrapper
├── run_competitive.zig           # Competitive benchmark runner
├── baselines/                    # Baseline snapshots and helpers
│
├── competitive/                  # Industry comparison benchmarks
│   ├── mod.zig                   # Competitive benchmark module
│   ├── faiss_comparison.zig      # vs Facebook FAISS
│   ├── llm_comparison.zig        # vs Other LLM frameworks
│   └── vector_db_comparison.zig  # vs Vector databases
│
├── core/                         # Core framework benchmarks
│   ├── mod.zig                   # Core benchmark module
│   ├── config.zig                # Configuration loading
│   ├── distance.zig              # Distance calculations
│   └── vectors.zig               # Vector operations
│
├── domain/                       # Domain-specific benchmarks
│   ├── ai/                       # AI/ML benchmarks
│   │   ├── mod.zig
│   │   ├── fpga_kernels.zig      # FPGA kernel benchmarks
│   │   ├── kernels.zig           # Compute kernels
│   │   ├── llm_metrics.zig       # LLM metrics (tokens/sec, etc.)
│   │   └── streaming.zig         # Streaming benchmarks
│   │
│   ├── database/                 # Database benchmarks
│   │   ├── mod.zig
│   │   ├── ann_benchmarks.zig    # ANN algorithms
│   │   ├── hnsw.zig              # HNSW performance
│   │   └── operations.zig        # CRUD operations
│   │
│   └── gpu/                      # GPU benchmarks
│       ├── mod.zig
│       ├── backends.zig          # Backend comparisons
│       ├── gpu_vs_cpu.zig        # GPU vs CPU comparisons
│       └── kernels.zig           # Kernel performance
│
├── infrastructure/               # Infrastructure benchmarks
│   ├── mod.zig
│   ├── concurrency.zig           # Concurrency patterns
│   ├── crypto.zig                # Cryptographic operations
│   ├── gpu_backends.zig          # Backend availability checks
│   ├── memory.zig                # Memory management
│   ├── network.zig               # Network stack benchmarks
│   └── simd.zig                  # SIMD/vectorization
│
└── system/                       # System/integration benchmarks
    ├── mod.zig
    ├── baseline_comparator.zig   # Baseline comparisons
    ├── baseline_store.zig        # Baseline persistence
    ├── ci_integration.zig        # CI/CD integration tests
    ├── framework.zig             # Framework initialization
    └── industry_standard.zig     # Industry standard compliance
```

## Purpose of Each Directory

### `competitive/` - Industry Comparisons
Compare ABI Framework against industry-standard implementations:
- **FAISS**: Vector similarity search libraries
- **LLM frameworks**: Inference speed comparisons
- **Vector databases**: QPS and latency comparisons

### `core/` - Framework Core Performance
Measure fundamental framework operations:
- Configuration loading speed
- Vector math operations
- Distance/similarity calculations
- Memory allocation patterns

### `domain/` - Feature-Specific Benchmarks
Domain-specific performance testing:
- **AI/ML**: compute kernels, LLM metrics, streaming
- **Database**: ANN algorithms, HNSW, CRUD operations
- **GPU**: backend comparisons, GPU vs CPU
We use `domain/ai/` and `domain/gpu/` instead of top-level `ai/`/`gpu/` for consistency.

### `infrastructure/` - System Infrastructure
Infrastructure component performance:
- Concurrency and parallelism
- Cryptographic operations
- GPU backend availability
- Memory management strategies
- Network stack benchmarks
- SIMD/vectorization effectiveness

### `system/` - Integration & Compliance
System-level and compliance testing:
- CI/CD integration performance
- Framework startup/shutdown
- Industry standard compliance
- Baseline storage/comparison

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
- System operation → `infrastructure/` or `system/`

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
# Test the new benchmark
zig run benchmarks/domain/ai/new_benchmark.zig

# Run all AI benchmarks
zig run benchmarks/domain/ai/mod.zig

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
zig run benchmarks/core/mod.zig
zig run benchmarks/domain/ai/mod.zig
zig run benchmarks/domain/database/mod.zig
zig run benchmarks/domain/gpu/mod.zig
```

### Competitive Benchmarks
```bash
zig build bench-competitive
```

### Individual Benchmark
```bash
zig run benchmarks/domain/ai/kernels.zig
```

## Output Formats

### Human-readable
```bash
zig build benchmarks  # Default format
```

### JSON
```bash
zig build benchmarks -- --format=json > results.json
```

### CSV
```bash
zig build benchmarks -- --format=csv > results.csv
```

### CI Integration
```bash
# With version tagging
zig build benchmarks -- --format=json --tag=git-$(git rev-parse --short HEAD)
```

## Performance Baselines

Baseline files are stored in `benchmarks/baselines/`:
- `baselines/latest.json` - Most recent results
- `baselines/release-*.json` - Release snapshots
- `baselines/regression-*.json` - Regression test results

Compare against baseline:
```bash
node scripts/compare-baseline.js benchmarks/baselines/latest.json current_results.json
```