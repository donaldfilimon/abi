# Quickstart

> For comprehensive guides, see [docs/intro.md](docs/intro.md).

## Requirements

- Zig 0.16.x

## Build and Run the CLI

```bash
zig build
zig build run -- --help
zig build run -- --version
```

## Run Tests and Benchmarks

```bash
zig build test                    # Run all tests
zig build test --summary all      # Run tests with detailed output
zig build benchmarks              # Run performance benchmarks
```

## Use the Library

### Basic Initialization

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

| Module | Description |
|--------|-------------|
| `src/abi.zig` | Public API entry point |
| `src/core/` | Hardware helpers and cache-aligned buffers |
| `src/compute/runtime/engine.zig` | Runtime engine and scheduler |
| `src/compute/concurrency/` | Lock-free data structures |
| `src/compute/memory/` | Pool and scratch allocators |
| `src/features/ai/` | AI features (LLM, embeddings, RAG) |
| `src/features/database/` | WDBX vector database |
| `src/features/gpu/` | GPU backend implementations |

## Next Steps

- Read the [Introduction](docs/intro.md) for architecture overview
- See [API Reference](API_REFERENCE.md) for API documentation
- Check [examples/](examples/) for more code samples
- Review [CONTRIBUTING.md](CONTRIBUTING.md) for development guidelines

## Contacts

src/shared/contacts.zig provides a centralized list of maintainer contacts extracted from the repository markdown files. Import this module wherever contact information is needed.

