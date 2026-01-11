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

```zig
const std = @import("std");
const abi = @import("abi");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();

    var framework = try abi.init(gpa.allocator(), abi.FrameworkOptions{});
    defer abi.shutdown(&framework);

    std.debug.print("ABI version: {s}\n", .{abi.version()});
}
```

### Compute Engine

```zig
const std = @import("std");
const abi = @import("abi");

fn computeTask(_: std.mem.Allocator) !u32 {
    return 42;
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();

    var engine = try abi.compute.createDefaultEngine(gpa.allocator());
    defer engine.deinit();

    const result = try abi.compute.runTask(&engine, u32, computeTask, 1000);
    std.debug.print("Result: {d}\n", .{result});
}
```

## Module Map

Primary modules:

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

