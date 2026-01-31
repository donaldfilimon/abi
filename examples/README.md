---
title: "Examples"
tags: [examples, tutorials, getting-started]
---
# ABI Framework Examples
> **Codebase Status:** Synced with repository as of 2026-01-31.

<p align="center">
  <img src="https://img.shields.io/badge/Examples-15-blue?style=for-the-badge" alt="15 Examples"/>
  <img src="https://img.shields.io/badge/Zig-0.16-F7A41D?style=for-the-badge&logo=zig&logoColor=white" alt="Zig"/>
  <img src="https://img.shields.io/badge/Learning-Path-success?style=for-the-badge" alt="Learning Path"/>
</p>

This directory contains example programs demonstrating various features of the ABI framework.

## Examples

| Example | Description | Run |
| --- | --- | --- |
| `hello.zig` | Basic framework initialization and version check | `zig build run-hello` |
| `database.zig` | WDBX insert/search/statistics | `zig build run-database` |
| `agent.zig` | AI agent chat with history tracking | `zig build -Denable-ai=true run-agent` |
| `compute.zig` | Compute engine + SIMD operations | `zig build run-compute` |
| `concurrency.zig` | Lock-free concurrency primitives | `zig build run-concurrency` |
| `gpu.zig` | GPU acceleration and SIMD operations | `zig build -Denable-gpu=true run-gpu` |
| `network.zig` | Cluster setup and node management | `zig build -Denable-network=true run-network` |
| `observability.zig` | Metrics and profiling demo | `zig build -Denable-profiling=true run-observability` |
| `discord.zig` | Discord bot integration | `zig build run-discord` |
| `llm.zig` | Local GGUF inference | `zig build run-llm -- path/to/model.gguf` |
| `orchestration.zig` | Multi-model routing and fallback | `zig build -Denable-ai=true run-orchestration` |
| `training.zig` | Training pipeline with checkpoints | `zig build -Denable-ai=true run-training` |
| `training/train_demo.zig` | Synthetic LLM training demo | `zig build -Denable-ai=true run-train-demo` |
| `train_ava.zig` | Train the Ava assistant model | `zig build run-train-ava -- path/to/gpt-oss.gguf --dataset-path train.jsonl` |
| `ha.zig` | HA backup, PITR, failover | `zig build -Denable-database=true run-ha` |

**Notes:**
- Examples may require feature flags (`-Denable-ai`, `-Denable-gpu`, `-Denable-network`, `-Denable-profiling`).
- `discord.zig` requires `DISCORD_BOT_TOKEN` in the environment.

## Building Examples

All examples are integrated into the main build system:

```bash
# Build all examples
zig build examples

# Run a specific example (see table above for commands)
zig build run-hello
zig build run-database
zig build run-concurrency
zig build -Denable-gpu=true run-gpu
zig build -Denable-network=true run-network
zig build -Denable-profiling=true run-observability
zig build run-llm -- path/to/model.gguf
```

## Running Benchmarks

The comprehensive benchmark suite tests all framework features:

```bash
# Run all benchmarks
zig build benchmarks
```

## Learning Path

1. **Start with `hello.zig`** - Learn basic framework initialization
2. **Try `database.zig`** - Understand vector storage and search
3. **Explore `compute.zig`** - Learn about task execution + SIMD
4. **Check `concurrency.zig`** - Learn lock-free primitives
5. **Review `gpu.zig`** - Understand GPU acceleration
6. **Study `network.zig`** - Learn distributed computing
7. **Check `observability.zig`** - Learn metrics and profiling
8. **Explore `agent.zig`** - See AI integration
9. **Try `llm.zig`** - Local LLM inference
10. **Explore `orchestration.zig`** - Multi-model routing and fallback
11. **Explore `training.zig`** - Model training and checkpointing
12. **Run `training/train_demo.zig`** - Synthetic training demo
13. **Train `train_ava.zig`** - Train the Ava assistant from gpt-oss
14. **Study `ha.zig`** - High availability features
15. **Check `discord.zig`** - Discord bot integration

## Common Patterns

All examples follow these Zig 0.16 best practices:

1. **Main Signature (Zig 0.16):**

   ```zig
   pub fn main() !void {
       var gpa = std.heap.GeneralPurposeAllocator(.{}){};
       defer _ = gpa.deinit();
       const allocator = gpa.allocator();
       // ... your code
   }
   ```

2. **Framework Initialization:**

   ```zig
   var framework = try abi.init(allocator, abi.FrameworkOptions{});
   defer abi.shutdown(&framework);
   ```

3. **Error Handling:**

   ```zig
   const result = doWork() catch |err| {
       std.debug.print("Failed: {t}\n", .{err});
       return err;
   };
   std.mem.doNotOptimizeAway(result);
   ```

4. **Cleanup with defer:**

   ```zig
   const data = try allocateData();
   defer allocator.free(data);
   ```

5. **Format Specifiers (Zig 0.16):**

   ```zig
   std.debug.print("Status: {t}\n", .{status});  // {t} for enums
   std.debug.print("Count: {d}\n", .{count});    // {d} for integers
   ```

## Need Help?

See [docs/README.md](../docs/README.md) for documentation sources and
[API_REFERENCE.md](../API_REFERENCE.md) for the public API summary.

## See Also

- [API Reference](../API_REFERENCE.md) - Detailed API information
- [Documentation Site](https://donaldfilimon.github.io/abi/) - Published docs
