# Quickstart

## Requirements
- Zig 0.16.x

## Build and run the CLI
```bash
zig build
zig build run -- --help
```

## Run tests and benchmarks
```bash
zig build test
zig build benchmark
```

## Use the library
```zig
const std = @import("std");
const abi = @import("abi");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();

    var framework = try abi.createDefaultFramework(gpa.allocator());
    defer framework.deinit();

    var engine = try abi.compute.createDefaultEngine(gpa.allocator());
    defer engine.deinit();

    const task_id = try engine.submit_task(u64, exampleTask);
    const result = try engine.wait_for_result(u64, task_id, 0);
    std.debug.print("Result: {d}\n", .{result});
}

fn exampleTask(_: std.mem.Allocator) !u64 {
    return 42;
}
```

## Module map
Primary modules live in the modern layout:
- `src/core/mod.zig` for hardware helpers and cache-aligned buffers
- `src/compute/runtime/engine.zig` for the runtime engine
- `src/compute/runtime/workload.zig` for sample CPU workloads
- `src/compute/concurrency/lockfree.zig` for lock-free data structures
- `src/compute/memory/mod.zig` for pool and scratch allocators

Use `src/demo.zig` for a small end-to-end program and
`src/compute/runtime/benchmark_demo.zig` for the benchmark runner.
