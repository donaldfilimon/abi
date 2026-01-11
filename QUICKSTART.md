# Quickstart

> For comprehensive guides, see [docs/intro.md](docs/intro.md).

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

### Basic Framework Usage (Zig 0.16)

```zig
const std = @import("std");
const abi = @import("abi");

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;

    var framework = try abi.init(allocator, abi.FrameworkOptions{});
    defer abi.shutdown(&framework);

    std.debug.print("ABI v{s} initialized\n", .{abi.version()});
}
```

### Compute Engine Example

```zig
const std = @import("std");
const abi = @import("abi");

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;

    var framework = try abi.init(allocator, abi.FrameworkOptions{});
    defer abi.shutdown(&framework);

    var engine = try abi.compute.createDefaultEngine(allocator);
    defer engine.deinit();

    // Option 1: Submit and wait separately
    const task_id = try abi.compute.submitTask(&engine, u64, exampleTask);
    const result1 = try abi.compute.waitForResult(&engine, u64, task_id, 1000);
    std.debug.print("Result 1: {d}\n", .{result1});

    // Option 2: Submit and wait in one call
    const result2 = try abi.compute.runTask(&engine, u64, exampleTask, 1000);
    std.debug.print("Result 2: {d}\n", .{result2});

    // Option 3: Using runWorkload (alias for runTask)
    const result3 = try abi.compute.runWorkload(&engine, u64, exampleTask, 1000);
    std.debug.print("Result 3: {d}\n", .{result3});
}

fn exampleTask(_: std.mem.Allocator) !u64 {
    return 42;
}
```

### AI Agent Example

```zig
const std = @import("std");
const abi = @import("abi");

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;

    var framework = try abi.init(allocator, abi.FrameworkOptions{
        .enable_ai = true,
    });
    defer abi.shutdown(&framework);

    var agent = try abi.ai.Agent.init(allocator, .{
        .name = "assistant",
        .temperature = 0.7,
    });
    defer agent.deinit();

    const response = try agent.chat("Hello!", allocator);
    defer allocator.free(response);
    std.debug.print("Agent: {s}\n", .{response});
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
