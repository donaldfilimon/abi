---
title: "Examples"
tags: [examples, tutorials, getting-started]
---
# ABI Framework Examples
> **Codebase Status:** Synced with repository as of 2026-02-25.

<p align="center">
  <img src="https://img.shields.io/badge/Examples-40-blue?style=for-the-badge" alt="40 Examples"/>
  <img src="https://img.shields.io/badge/Zig-0.16-F7A41D?style=for-the-badge&logo=zig&logoColor=white" alt="Zig"/>
  <img src="https://img.shields.io/badge/Learning-Path-success?style=for-the-badge" alt="Learning Path"/>
</p>

This directory contains example programs demonstrating various features of the ABI framework.

## Examples

### hello.zig

Basic framework initialization and version check.

**Run:**

```bash
zig build run-hello
```

### database.zig

Vector database operations including insert, search, and statistics.

**Run:**

```bash
zig build run-database
```

### agent.zig

AI agent usage with conversational chat interface. Demonstrates the `Agent.chat()` method for processing user input with history tracking.

**Features:**
- Agent initialization with configuration
- Using `chat()` method for conversational interface
- Proper memory management with defer

**Run:**

```bash
zig build run-agent
```

### compute.zig

Compute engine task execution and result handling.

**Run:**

```bash
zig build run-compute
```

### concurrency.zig

Lock-free concurrency primitives (MPMC queue, Chase-Lev deque).

**Run:**

```bash
zig build run-concurrency
```

### gpu.zig

GPU acceleration and SIMD operations.

**Run:**

```bash
zig build run-gpu
```

### network.zig

Network cluster setup and node management.

**Run:**

```bash
zig build run-network
```

### observability.zig

Metrics and tracing primitives (counters, gauges, histograms).

**Run:**

```bash
zig build run-observability
```

### discord.zig

Discord bot integration with bot info, guild listing, and gateway information.

**Prerequisites:**
- Set `DISCORD_BOT_TOKEN` environment variable with your bot token

**Run:**

```bash
zig build run-discord
```

### training.zig

Model training with optimizers, checkpointing, and metrics.

**Features:**
- Training configuration (epochs, batch size, learning rate)
- AdamW optimizer with weight decay
- Checkpoint saving and resuming
- Loss history tracking

**Run:**

```bash
zig build run-training
```

### training/train_demo.zig

Focused LLM training demo with smaller defaults.

**Run:**

```bash
zig build run-train-demo
```

### llm.zig

Local LLM inference with GGUF models.

**Features:**
- GGUF model loading
- BPE/SentencePiece tokenization
- Text generation with sampling (temperature, top-k, top-p)
- Streaming output

**Run:**

```bash
zig build run-llm -- path/to/model.gguf
```

### orchestration.zig

Multi-model orchestration (routing, fallback, ensemble).

**Run:**

```bash
zig build run-orchestration
```

### train_ava.zig

Train the Ava assistant model based on gpt-oss.

**Features:**
- Fine-tuning from gpt-oss compatible GGUF models
- LoRA support for efficient training
- JSONL and text dataset formats
- Checkpointing and GGUF export
- GPU acceleration with CPU fallback

**Run:**

```bash
# Basic training
zig build run-train-ava -- path/to/gpt-oss.gguf --dataset-path train.jsonl

# With custom configuration
zig build run-train-ava -- gpt2.gguf -d data.jsonl --epochs 5 --lr 2e-5

# Show help
zig build run-train-ava -- --help
```

### ha.zig

High Availability features for production deployments.

**Features:**
- Multi-region replication setup
- Backup orchestration
- Point-in-time recovery (PITR)
- Automatic failover

**Run:**

```bash
zig build run-ha
```

## Building Examples

All examples are integrated into the main build system:

```bash
# Build all examples
zig build examples

# Run a specific example
zig build run-hello
zig build run-database
zig build run-compute
zig build run-concurrency
zig build run-gpu
zig build run-network
zig build run-discord
zig build run-training
zig build run-train-demo
zig build run-llm
zig build run-orchestration
zig build run-train-ava
zig build run-observability
zig build run-ha
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
3. **Explore `compute.zig`** - Learn about task execution
4. **Review `concurrency.zig`** - Lock-free primitives and queues
5. **Check `agent.zig`** - See AI integration
6. **Review `gpu.zig`** - Understand GPU acceleration
7. **Study `network.zig`** - Learn distributed computing
8. **Check `observability.zig`** - Metrics and tracing fundamentals
9. **Check `discord.zig`** - Discord bot integration
10. **Explore `training.zig`** - Model training and checkpointing
11. **Try `training/train_demo.zig`** - Focused LLM training demo
12. **Try `llm.zig`** - Local LLM inference
13. **Study `orchestration.zig`** - Multi-model routing and fallback
14. **Study `ha.zig`** - High availability features
15. **Train `train_ava.zig`** - Train the Ava assistant from gpt-oss

## Common Patterns

All examples follow these Zig 0.16 best practices:

1. **Modern Main Signature (Zig 0.16):**

   ```zig
   pub fn main(init: std.process.Init) !void {
       var gpa = std.heap.GeneralPurposeAllocator(.{}){};
       defer _ = gpa.deinit();
       const allocator = gpa.allocator();
       const io = init.io;
       // ... your code
   }
   ```

2. **Framework Initialization:**

   ```zig
   var framework = try abi.Framework.builder(allocator)
       .withDefaults()
       .build();
   defer framework.deinit();
   ```

3. **Error Handling:**

   ```zig
   pub fn main(init: std.process.Init) !void {
       try someOperation();
   }
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

See the [Documentation README](../docs/README.md) for guides, or check API_REFERENCE.md for detailed API information.

## See Also

- [API Reference](../API_REFERENCE.md) - Detailed API information
- [Documentation README](../docs/README.md) - Documentation site source
