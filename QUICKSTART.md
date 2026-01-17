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
zig test src/runtime/engine/engine.zig     # Test single file (new path)
zig test src/tests/mod.zig --test-filter "pattern"  # Filter tests
zig build test -Denable-gpu=true -Denable-network=true  # Test with features
zig build benchmarks              # Run performance benchmarks
```

## CLI Commands

```bash
# General
zig build run -- --help           # Show all commands
zig build run -- --version        # Show version info
zig build run -- system-info       # System and framework status

# Database
zig build run -- db stats
zig build run -- db add --id 1 --embed "text to embed"
zig build run -- db backup --path backup.db
zig build run -- db restore --path backup.db

# AI & Agents
zig build run -- agent                          # Interactive agent
zig build run -- agent --message "Hello"          # Single message
zig build run -- agent --persona coder           # Use specific persona

# GPU
zig build run -- gpu backends                  # List backends
zig build run -- gpu devices                   # List devices
zig build run -- gpu summary                   # GPU summary

# Training
zig build run -- train run --epochs 10          # Run training
zig build run -- train info                   # Show config
zig build run -- train resume ./model.ckpt     # Resume from checkpoint

# Explore codebase
zig build run -- explore "fn init" --level quick
zig build run -- explore "pub fn" --level thorough
```

## Use the Library

### Basic Framework Usage (Zig 0.16)

**New unified configuration** (recommended):
```zig
const std = @import("std");
const abi = @import("abi");

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;

    // Unified Config with builder pattern
    const config = abi.Config.init()
        .withAI(true)
        .withGPU(true)
        .withDatabase(true);

    var framework = try abi.Framework.init(allocator, config);
    defer framework.deinit();

    std.debug.print("ABI v{s} initialized\n", .{abi.version()});

    // Access feature modules through the framework
    if (framework.ai()) |ai| {
        _ = ai; // Use AI features
    }
}
```

**Backward-compatible initialization**:
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

Flat domain structure (new modular architecture):

| Module | Description | Documentation |
|--------|-------------|----------------|
| `src/abi.zig` | Public API entry point | [API Reference](API_REFERENCE.md) |
| `src/config.zig` | Unified configuration system | [API Reference](API_REFERENCE.md) |
| `src/framework.zig` | Framework orchestration | [Framework Guide](docs/framework.md) |
| `src/runtime/` | Always-on infrastructure (scheduler, memory, concurrency) | [Compute Guide](docs/compute.md) |
| `src/gpu/` | GPU backends and unified API | [GPU Guide](docs/gpu.md) |
| `src/ai/` | AI module entry point | [AI Guide](docs/ai.md) |
| `src/ai/llm/` | Local LLM inference | [AI Guide](docs/ai.md) |
| `src/ai/embeddings/` | Vector embeddings | [AI Guide](docs/ai.md) |
| `src/ai/agents/` | AI agent runtime | [AI Guide](docs/ai.md) |
| `src/ai/training/` | Training pipelines | [AI Guide](docs/ai.md) |
| `src/database/` | WDBX vector database | [Database Guide](docs/database.md) |
| `src/network/` | Distributed compute and Raft | [Network Guide](docs/network.md) |
| `src/observability/` | Metrics, tracing, profiling | [Monitoring Guide](docs/monitoring.md) |
| `src/web/` | HTTP helpers and web utilities | - |
| `src/shared/` | Shared utilities and platform helpers | - |

## Next Steps

- Read the [Introduction](docs/intro.md) for architecture overview
- See [API Reference](API_REFERENCE.md) for API documentation
- Check [examples/](examples/) for more code samples
- Review [CONTRIBUTING.md](CONTRIBUTING.md) for development guidelines

## See Also

- [TODO.md](TODO.md) - Pending implementations
- [ROADMAP.md](ROADMAP.md) - Upcoming milestones
