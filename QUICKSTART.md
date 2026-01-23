---
title: "QUICKSTART"
tags: []
---
# Quickstart
> **Codebase Status:** Synced with repository as of 2026-01-22.

<p align="center">
  <img src="https://img.shields.io/badge/Zig-0.16-F7A41D?style=for-the-badge&logo=zig&logoColor=white" alt="Zig 0.16"/>
  <img src="https://img.shields.io/badge/Difficulty-Beginner-green?style=for-the-badge" alt="Difficulty"/>
  <img src="https://img.shields.io/badge/Time-5_minutes-blue?style=for-the-badge" alt="Time"/>
</p>

> **Getting Started Fast** — This guide gets you running in under 5 minutes.
> For comprehensive guides, see [docs/intro.md](docs/intro.md).

---

## Prerequisites

| Requirement | Version | Status |
|-------------|---------|--------|
| Zig | 0.16.x | ![Required](https://img.shields.io/badge/-Required-red) |
| Git | Any | ![Required](https://img.shields.io/badge/-Required-red) |
| GPU Drivers | Latest | ![Optional](https://img.shields.io/badge/-Optional-yellow) |

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

    var engine = try abi.runtime.createEngine(allocator, .{});
    defer engine.deinit();

    // Option 1: Submit and wait separately
    const task_id = try abi.runtime.submitTask(&engine, u64, exampleTask);
    const result1 = try abi.runtime.waitForResult(&engine, u64, task_id, 1000);
    std.debug.print("Result 1: {d}\n", .{result1});

    // Option 2: Submit and wait in one call
    const result2 = try abi.runtime.runTask(&engine, u64, exampleTask, 1000);
    std.debug.print("Result 2: {d}\n", .{result2});

    // Option 3: Using runWorkload (alias for runTask)
    const result3 = try abi.runtime.runWorkload(&engine, u64, exampleTask, 1000);
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

## Module Map

Flat domain structure with modular architecture:

### Core Modules

| Module | Description | Status | Docs |
|--------|-------------|--------|------|
| `src/abi.zig` | Public API entry point | ![Ready](https://img.shields.io/badge/-Ready-success) | [API Reference](API_REFERENCE.md) |
| `src/config.zig` | Unified configuration system | ![Ready](https://img.shields.io/badge/-Ready-success) | [API Reference](API_REFERENCE.md) |
| `src/framework.zig` | Framework orchestration | ![Ready](https://img.shields.io/badge/-Ready-success) | [Framework Guide](docs/framework.md) |
| `src/runtime/` | Scheduler, memory, concurrency | ![Ready](https://img.shields.io/badge/-Ready-success) | [Compute Guide](docs/compute.md) |

### Feature Modules

| Module | Description | Status | Docs |
|--------|-------------|--------|------|
| `src/gpu/` | GPU backends and unified API | ![Ready](https://img.shields.io/badge/-Ready-success) | [GPU Guide](docs/gpu.md) |
| `src/ai/` | AI module entry point | ![Ready](https://img.shields.io/badge/-Ready-success) | [AI Guide](docs/ai.md) |
| `src/ai/llm/` | Local LLM inference | ![Ready](https://img.shields.io/badge/-Ready-success) | [AI Guide](docs/ai.md) |
| `src/ai/embeddings/` | Vector embeddings | ![Ready](https://img.shields.io/badge/-Ready-success) | [AI Guide](docs/ai.md) |
| `src/ai/agents/` | AI agent runtime | ![Ready](https://img.shields.io/badge/-Ready-success) | [AI Guide](docs/ai.md) |
| `src/ai/training/` | Training pipelines | ![Ready](https://img.shields.io/badge/-Ready-success) | [AI Guide](docs/ai.md) |
| `src/database/` | WDBX vector database | ![Ready](https://img.shields.io/badge/-Ready-success) | [Database Guide](docs/database.md) |
| `src/network/` | Distributed compute and Raft | ![Ready](https://img.shields.io/badge/-Ready-success) | [Network Guide](docs/network.md) |
| `src/observability/` | Metrics, tracing, profiling | ![Ready](https://img.shields.io/badge/-Ready-success) | [Monitoring Guide](docs/monitoring.md) |
| `src/web/` | HTTP helpers and web utilities | ![Ready](https://img.shields.io/badge/-Ready-success) | - |
| `src/shared/` | Shared utilities and helpers | ![Ready](https://img.shields.io/badge/-Ready-success) | - |

## Next Steps

<table>
<tr>
<td width="50%">

### Learn More
- [Introduction](docs/intro.md) — Architecture overview
- [API Reference](API_REFERENCE.md) — API documentation
- [Examples](examples/) — Code samples

</td>
<td width="50%">

### Get Involved
- [Contributing](CONTRIBUTING.md) — Development guidelines
- [Roadmap](ROADMAP.md) — Upcoming milestones
- [TODO](TODO.md) — Pending implementations (see [Claude‑Code Massive TODO](TODO.md#claude-code-massive-todo))

</td>
</tr>
</table>

---

<p align="center">
  <a href="README.md">← Back to README</a> •
  <a href="docs/intro.md">Full Documentation →</a>
</p>

