# Benchmark Suite

This directory contains benchmark tests for measuring ABI framework performance.

## Running Benchmarks

```bash
zig build-exe benchmarks/run.zig
```

## Benchmark Categories

### Empty Task (No-op)
Measures the overhead of the benchmark harness itself.

### Simple Compute
Performs a simple sum operation (1 + 2 + ... + 1000) to test compute engine throughput.

### Memory Allocation
Allocates and frees a 1KB buffer to test memory management performance.

### Vector Operations
Performs 4D dot product calculation to test SIMD operations and vector math.

## Interpreting Results

Each benchmark outputs:
- `iterations`: Number of iterations measured
- `duration_ns`: Total duration in nanoseconds
- `ops_per_sec`: Operations per second
- `avg_ns`: Average time per operation
- `min_ns`: Minimum time for any iteration
- `max_ns`: Maximum time for any iteration

## Adding New Benchmarks

To add a new benchmark:

1. Create a benchmark function that takes `std.mem.Allocator` and returns a value or `!void`
2. Add a call in `benchmarks/run.zig` `main()` function:
   ```zig
   _ = try benchmark.executeBenchmark(allocator, "Your Benchmark Name", yourBenchFunc);
   ```

## Notes

- Benchmarks warm up for 50,000 iterations before measuring
- Measurements run for 500,000 iterations
- Results include warm-up time in reported duration
- Use consistent naming conventions
- Free any allocated memory to avoid leaks in benchmark runs
