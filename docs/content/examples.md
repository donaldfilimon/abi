---
title: Examples
description: 32 runnable examples covering all major modules
section: Reference
order: 19
---

# Examples

ABI ships with 32 runnable examples in the `examples/` directory. Each example
is wired into `build.zig` as a named build target.

## Build and Run

```bash
# Build all examples at once
zig build examples

# Run a specific example by name
zig build run-hello
zig build run-gpu
zig build run-database
```

Each example uses `@import("abi")` and demonstrates a self-contained use case.
Examples that require hardware (GPU, network) gracefully degrade when the
hardware is unavailable.

---

## Categories

### Core

| Example | Target | Description |
|---------|--------|-------------|
| `hello` | `run-hello` | Minimal framework init, version print |
| `compute` | `run-compute` | Basic SIMD vector operations |
| `concurrency` | `run-concurrency` | Thread pool, channels, DAG scheduling |

### AI and LLM

| Example | Target | Description |
|---------|--------|-------------|
| `llm` | `run-llm` | Local LLM inference with streaming |
| `training` | `run-training` | Training pipeline configuration |
| `orchestration` | `run-orchestration` | Multi-model orchestration |
| `agent` | `run-agent` | Agent with tool use and memory |

### GPU

| Example | Target | Description |
|---------|--------|-------------|
| `gpu` | `run-gpu` | Kernel dispatch, backend detection |
| `multi-gpu` | `run-multi-gpu` | Multi-GPU orchestration |

### Database

| Example | Target | Description |
|---------|--------|-------------|
| `database` | `run-database` | WDBX vector database operations |
| `vectors` | `run-vectors` | Vector similarity search |

### Infrastructure

| Example | Target | Description |
|---------|--------|-------------|
| `network` | `run-network` | Peer discovery, messaging |
| `ha` | `run-ha` | High availability, failover |
| `observability` | `run-observability` | Metrics, tracing, alerting |
| `discord` | `run-discord` | Discord bot integration |

### New Modules

| Example | Target | Description |
|---------|--------|-------------|
| `cache` | `run-cache` | LRU/LFU cache with TTL |
| `search` | `run-search` | Full-text BM25 search |
| `messaging` | `run-messaging` | Pub/sub with MQTT patterns |
| `storage` | `run-storage` | Object storage with metadata |
| `gateway` | `run-gateway` | API routing, rate limiting |
| `pages` | `run-pages` | URL routing, template rendering |

---

## Writing Your Own Example

1. Create a new `.zig` file in the `examples/` directory.
2. Import the framework:

```zig
const std = @import("std");
const abi = @import("abi");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();

    var fw = try abi.initDefault(gpa.allocator());
    defer fw.deinit();

    // Your code here
}
```

3. Add a build target in `build.zig` (or use the existing example target
   infrastructure) to wire it up as `run-<name>`.

---

## Feature Flag Interaction

Examples compile against whichever feature flags are active. If an example
uses a module that is disabled, the stub returns `error.FeatureDisabled`
at runtime. To run GPU examples:

```bash
zig build run-gpu -Denable-gpu=true -Dgpu-backend=metal
```

To run without GPU:

```bash
zig build run-gpu -Denable-gpu=false
# The example will handle FeatureDisabled gracefully
```
