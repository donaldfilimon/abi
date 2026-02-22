---
title: Home
description: ABI Framework — high-performance AI and systems framework in Zig
layout: default
---

# ABI Framework

ABI is a high-performance AI and systems framework written in Zig 0.16. It unifies
AI inference, GPU compute, vector database, runtime infrastructure, and operational
tooling into a single build with compile-time feature gating.

Version **0.4.0** provides 21 feature modules, 10 GPU backends, 9 LLM provider
connectors, and 36 CLI commands (plus 10 aliases) — all tested with
1263 passing tests (5 skipped) and 2263 passing feature tests.

## What You Can Build

| Capability | Module(s) | Description |
|------------|-----------|-------------|
| LLM inference | `abi.ai.llm`, `abi.connectors` | Local model serving, chat, embeddings, streaming with 9 provider backends |
| GPU compute pipelines | `gpu` | Kernel DSL, multi-GPU orchestration, 10 backends (CUDA, Vulkan, Metal, WebGPU, ...) |
| Vector search | `database`, `search` | WDBX vector database with full-text BM25 search |
| Distributed compute | `network`, `runtime` | Peer discovery, work-stealing thread pool, DAG pipeline scheduling |
| API gateways | `gateway` | Radix-tree routing, rate limiting (token bucket, sliding window, fixed window), circuit breaker |
| Pub/sub messaging | `messaging` | Topic pub/sub with MQTT-style pattern matching, dead letter queues |
| Training pipelines | `abi.ai.training` | Federated learning, synthetic data generation, quantization |
| Agent systems | `abi.ai.core`, `abi.ai.reasoning` | Multi-agent workflows, tool use, memory, Abbey reasoning engine |
| Caching | `cache` | LRU/LFU/FIFO eviction, slab allocation, TTL, atomic stats |
| Object storage | `storage` | Unified file/object storage with pluggable backends |
| Dashboard pages | `pages` | URL routing with path parameters and template rendering |
| Observability | `observability` | Metrics, distributed tracing, alerting |

## Design Goals

- **Single runtime, feature flags at compile time.** Every module has a real
  implementation (`mod.zig`) and a disabled stub (`stub.zig`). Disabled features
  contribute zero binary overhead. No dynamic dispatch tax for features you don't use.

- **Fast startup, predictable resources.** Zig's lack of a garbage collector and
  explicit allocator passing give you deterministic memory behavior. The framework
  lifecycle is a state machine: `uninitialized -> initializing -> running -> stopping -> stopped`.

- **Clear API boundaries with stubs.** When a feature is disabled, its stub returns
  `error.FeatureDisabled` with the same public signature. Code compiles and links
  regardless of which features are turned on.

- **Operational safety.** Zero `@panic` calls in library code. Composable error
  hierarchy (`FrameworkError`). Secure connector teardown (API keys zeroed before free).
  Path traversal validation on all storage operations.

## Quick Start

```bash
# Build the framework
zig build

# See available CLI commands (36 commands + 10 aliases)
zig build run -- --help

# Check system status and enabled features
zig build run -- system-info

# Run the full test suite
zig build test --summary all
```

```zig
const std = @import("std");
const abi = @import("abi");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();

    var fw = try abi.initDefault(gpa.allocator());
    defer fw.deinit();

    std.debug.print("ABI v{s}\n", .{abi.version()});
}
```

## Documentation Structure

| Section | Pages | Purpose |
|---------|-------|---------|
| **Start** | [Home](/), [Installation](/installation/), [Getting Started](/getting-started/) | Installation, first build, orientation |
| **Core** | [Architecture](/architecture/), [Configuration](/configuration/), [Framework Lifecycle](/framework/), [CLI](/cli/) | Module hierarchy, feature gating, build flags, commands |
| **Modules** | GPU, AI, Database, Network, ... | Per-module API reference and usage |
| **Operations** | Deployment, Observability, Security | Production concerns |
| **Reference** | [API Overview](/api-overview/), [API Reference](/api/), Contributing, Examples | Generated API docs, dev workflow |

Continue to [Installation](/installation/) to set up your toolchain, or jump straight
to [Getting Started](/getting-started/) if you already have Zig 0.16 installed.

## Zig Skill
Use the Zig 0.16-dev patterns documented in CLAUDE.md for syntax guidance.
