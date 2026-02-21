---
title: Examples
description: 36 runnable examples covering all major modules
section: Reference
order: 2
---

# Examples

ABI ships with 36 runnable examples in the `examples/` directory. Each example
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
| `hello.zig` | `run-hello` | Minimal framework init, version print |
| `compute.zig` | `run-compute` | SIMD vector operations: dot product, cosine similarity, L2 distance |
| `concurrency.zig` | `run-concurrency` | Lock-free MPMC queue, Chase-Lev work-stealing deque |
| `concurrent_pipeline.zig` | `run-concurrent-pipeline` | Pipeline with Channel, ThreadPool, and DagPipeline |
| `config.zig` | `run-config` | Configuration system with GPU, AI, and database settings via Builder |
| `registry.zig` | `run-registry` | Feature registry with comptime and runtime toggle modes |
| `tensor_ops.zig` | `run-tensor-ops` | Matrix multiply, tensor transforms, SIMD vector ops, v2 primitives |
| `embeddings.zig` | `run-embeddings` | SIMD-accelerated vector operations for embeddings |

### AI and LLM

| Example | Target | Description |
|---------|--------|-------------|
| `llm.zig` | `run-llm` | Local GGUF model loading, tokenization, text generation, streaming |
| `llm_real.zig` | `run-llm-real` | Live inference via Ollama, LM Studio, or vLLM backends |
| `agent.zig` | `run-agent` | Agent initialization, tool registration, query processing |
| `ai_core.zig` | `run-ai-core` | Agents, tool registries, prompt builders, model discovery |
| `ai_inference.zig` | `run-ai-inference` | LLM engine configuration, embeddings, streaming generation |
| `ai_training.zig` | `run-ai-training` | Training pipeline configuration, optimizer selection, checkpoints |
| `ai_reasoning.zig` | `run-ai-reasoning` | Abbey reasoning, RAG pipelines, evaluation templates |
| `training.zig` | `run-training` | Model training with optimizers, checkpoints, gradient checkpointing |
| `train_ava.zig` | `run-train-ava` | Fine-tune the Ava assistant model from gpt-oss base |
| `orchestration.zig` | `run-orchestration` | Multi-model routing, load balancing, fallback, ensemble mode |
| `streaming.zig` | `run-streaming` | SSE encoding, stream events, circuit breaker patterns |

### GPU

| Example | Target | Description |
|---------|--------|-------------|
| `gpu.zig` | `run-gpu` | Device discovery, buffer management, high-level ops, profiling |

### Data

| Example | Target | Description |
|---------|--------|-------------|
| `database.zig` | `run-database` | WDBX vector database with HNSW indexing and similarity search |
| `cache.zig` | `run-cache` | LRU/LFU cache with TTL, eviction, and statistics |
| `search.zig` | `run-search` | Inverted index with BM25 ranking, document indexing |
| `storage.zig` | `run-storage` | Unified object storage abstraction with metadata and memory backend |

### Infrastructure

| Example | Target | Description |
|---------|--------|-------------|
| `network.zig` | `run-network` | Distributed compute, node registration, Raft consensus |
| `gateway.zig` | `run-gateway` | API gateway: radix-tree routing, rate limiting, circuit breaker |
| `messaging.zig` | `run-messaging` | Pub/sub with MQTT-style topic patterns and wildcard matching |
| `pages.zig` | `run-pages` | Dashboard/UI pages, URL routing, template rendering |
| `web.zig` | `run-web` | HTTP client, persona routing, chat handling, JSON utilities |

### Operations

| Example | Target | Description |
|---------|--------|-------------|
| `observability.zig` | `run-observability` | Counters, gauges, histograms, distributed tracing spans |
| `analytics.zig` | `run-analytics` | Event tracking, sessions, funnels, statistics gathering |
| `auth.zig` | `run-auth` | JWT, API key management, RBAC, rate limiting |
| `ha.zig` | `run-ha` | Multi-region replication, backup orchestration, PITR, failover |

### Services

| Example | Target | Description |
|---------|--------|-------------|
| `discord.zig` | `run-discord` | Discord bot: API connection, guild listing, messaging |
| `cloud.zig` | `run-cloud` | Serverless wrappers for AWS Lambda, GCP Functions, Azure |
| `mobile.zig` | `run-mobile` | Platform lifecycle, sensors, notifications (requires `-Denable-mobile=true`) |

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
    std.debug.print("ABI v{s}\n", .{abi.version()});
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

---

## Related Pages

- [API Overview](api.html) -- Public API surface and import patterns
- [Configuration](configuration.html) -- Feature flags and build options
- [GPU Module](gpu.html) -- GPU backends and kernel DSL

## Zig Skill
Use [$zig](/Users/donaldfilimon/.codex/skills/zig/SKILL.md) for new Zig syntax improvements and validation guidance.
