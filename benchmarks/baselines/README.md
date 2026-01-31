# Benchmark Baselines
> **Last reviewed:** 2026-01-31

This directory stores benchmark baseline results for regression detection.

## Directory Structure

```
baselines/
├── main/                    # Main branch baselines (default)
│   ├── vector_dot_128.json
│   ├── database_insert.json
│   └── ...
├── releases/                # Release tag baselines
│   ├── v1.0.0/
│   │   └── ...
│   └── v1.1.0/
│       └── ...
└── branches/                # Feature branch baselines
    ├── feature_simd/
    │   └── ...
    └── fix_memory_leak/
        └── ...
```

## JSON Format

Each baseline file contains a single benchmark result:

```json
{
  "category": "simd",
  "git_branch": "main",
  "git_commit": "abc123def456",
  "memory_bytes": 4096,
  "metric": "ops_per_sec",
  "name": "vector_dot_128",
  "p99_ns": 750,
  "sample_count": 1000,
  "std_dev": 1500.25,
  "timestamp": 1706000000,
  "unit": "ops/s",
  "value": 1500000.5
}
```

## Usage

### Saving Baselines

```zig
const system = @import("benchmarks/system/mod.zig");

var store = system.BaselineStore.init(allocator, "benchmarks/baselines");
defer store.deinit();

try store.saveBaseline(.{
    .name = "my_benchmark",
    .metric = "ops_per_sec",
    .value = 1500000.0,
    .unit = "ops/s",
    .timestamp = std.time.timestamp(),
    .git_branch = "main",
    .git_commit = "abc123",
});
```

### Comparing Against Baselines

```zig
const report = try system.compareAll(&store, current_results, allocator);
defer report.deinit(allocator);

if (report.hasRegressions()) {
    std.debug.print("Regressions detected!\n", .{});
    try report.format(std.io.getStdErr().writer());
}
```

## CI Integration

Baselines are automatically compared in CI:

1. Feature branches compare against `main/` baselines
2. Main branch updates baselines after successful merge
3. Release tags create snapshots in `releases/`

## Configuration

Comparison thresholds can be configured:

```zig
const config = system.ComparisonConfig{
    .regression_threshold = 5.0,  // 5% slower = regression
    .improvement_threshold = 5.0, // 5% faster = improvement
    .fallback_to_main = true,     // Use main if no branch baseline
};
```
