---
title: API Overview
description: Public API surface, import patterns, and HTTP endpoints
section: Reference
order: 1
permalink: /api-overview/
---

# API Overview

ABI exposes its full feature set through a single Zig import. Every module,
type alias, and convenience function is accessible via `@import("abi")`.

## Import Pattern

```zig
const abi = @import("abi");
```

This single import provides access to every compiled-in module. Disabled
features resolve to stubs that return `error.FeatureDisabled` at the same
call sites, so downstream code compiles regardless of which flags are active.

## Module Access

### Feature Modules (21 comptime-gated)

| Namespace | Module | Build Flag | Description |
|-----------|--------|------------|-------------|
| `abi.ai` | AI monolith | `-Denable-ai` | Full 17-submodule AI surface |
| `abi.ai_core` | AI Core | `-Denable-ai` | Agents, tools, prompts, personas, memory |
| `abi.inference` | Inference | `-Denable-llm` | LLM, embeddings, vision, streaming, transformer |
| `abi.training` | Training | `-Denable-training` | Training pipelines, federated learning |
| `abi.reasoning` | Reasoning | `-Denable-reasoning` | Abbey, RAG, eval, templates, orchestration |
| `abi.gpu` | GPU | `-Denable-gpu` | Kernel DSL, multi-GPU, 10 backends |
| `abi.database` | Database | `-Denable-database` | WDBX vector database |
| `abi.network` | Network | `-Denable-network` | Peer discovery, distributed compute |
| `abi.web` | Web | `-Denable-web` | HTTP client, persona routing, JSON utilities |
| `abi.cloud` | Cloud | `-Denable-cloud` | Serverless adapters (AWS, GCP, Azure) |
| `abi.observability` | Observability | `-Denable-profiling` | Metrics, tracing, profiling |
| `abi.analytics` | Analytics | `-Denable-analytics` | Event tracking, sessions, funnels |
| `abi.auth` | Auth | `-Denable-auth` | JWT, API keys, RBAC, rate limiting (16 security modules) |
| `abi.cache` | Cache | `-Denable-cache` | LRU/LFU/FIFO in-memory cache with TTL |
| `abi.storage` | Storage | `-Denable-storage` | Unified file/object storage |
| `abi.messaging` | Messaging | `-Denable-messaging` | Topic pub/sub, dead letter queues |
| `abi.gateway` | Gateway | `-Denable-gateway` | Radix-tree router, rate limiting, circuit breaker |
| `abi.search` | Search | `-Denable-search` | Full-text BM25 search |
| `abi.pages` | Pages | `-Denable-pages` | Dashboard/UI pages with URL path routing |
| `abi.mobile` | Mobile | `-Denable-mobile` | Platform lifecycle, sensors, notifications (off by default) |
| `abi.benchmarks` | Benchmarks | `-Denable-benchmarks` | Built-in benchmark suite and timing |

### Services (always available)

| Namespace | Module | Description |
|-----------|--------|-------------|
| `abi.runtime` | Runtime | Thread pool, channels, DAG scheduling |
| `abi.platform` | Platform | OS/architecture detection and abstraction |
| `abi.shared` | Shared | Utils, SIMD, time, sync, security, resilience |
| `abi.connectors` | Connectors | 9 LLM providers + Discord + scheduler |
| `abi.ha` | High Availability | Replication, backup, PITR |
| `abi.tasks` | Tasks | Task management system |
| `abi.mcp` | MCP | Model Context Protocol server (JSON-RPC 2.0) |
| `abi.acp` | ACP | Agent Communication Protocol |
| `abi.simd` | SIMD | Vector operations, hardware detection (shorthand for `shared.simd`) |

### Core (always available)

| Namespace | Type/Module | Description |
|-----------|-------------|-------------|
| `abi.Config` | Config struct | Unified configuration for all features |
| `abi.Framework` | Framework struct | Lifecycle state machine and feature orchestration |
| `abi.FrameworkBuilder` | Builder | Fluent builder pattern for framework init |
| `abi.Registry` | Registry | Plugin/feature registry |
| `abi.Feature` | Enum | Feature enumeration |
| `abi.vnext` | vNext API | Staged compatibility for next-gen API surface |
| `abi.config` | Config module | Feature descriptions, builder, validation |
| `abi.errors` | Error module | Composable error hierarchy |

## Convenience Aliases

Two types are re-exported at the top level for ergonomics:

```zig
const Gpu = abi.Gpu;
const GpuBackend = abi.GpuBackend;
```

All other functions use namespaced paths:

```zig
abi.simd.vectorAdd(a, b, len);
abi.simd.hasSimdSupport();
abi.connectors.discord;
```

## Framework Initialization

Two initialization patterns cover the common cases:

```zig
const std = @import("std");
const abi = @import("abi");

// 1. Default configuration (all compile-time features enabled)
var fw = try abi.initDefault(allocator);
defer fw.deinit();

// 2. Custom configuration
var fw = try abi.init(allocator, .{
    .gpu = .{ .backend = .vulkan },
    .cache = .{ .max_entries = 5000, .eviction_policy = .lru },
});
defer fw.deinit();
```

A builder pattern is also available:

```zig
var fw = try abi.Framework.builder(allocator)
    .withGpu(.{ .backend = .vulkan })
    .withAi(.{ .llm = .{ .model_path = "./models/llama.gguf" } })
    .withDatabase(.{ .path = "./data" })
    .build();
defer fw.deinit();
```

The framework manages a lifecycle state machine:
`uninitialized -> initializing -> running -> stopping -> stopped` (or `failed`).

### vNext API

New code can use the staged-compatibility vNext surface:

```zig
var app = try abi.initAppDefault(allocator);
defer app.deinit();

// Access the legacy framework if needed
const fw = app.getFramework();
```

---

## HTTP Endpoints

ABI exposes several HTTP APIs when running in server mode.

### Persona API

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/v1/chat` | Send a chat message |
| `POST` | `/api/v1/chat/abbey` | Chat via Abbey reasoning engine |
| `GET` | `/api/v1/personas` | List available personas |
| `GET` | `/api/v1/personas/metrics` | Persona usage metrics |
| `GET` | `/api/v1/personas/health` | Health check for persona service |

### Streaming API (OpenAI-compatible)

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/v1/chat/completions` | OpenAI-compatible SSE streaming |
| `POST` | `/api/stream` | Direct streaming endpoint |
| `GET` | `/api/stream/ws` | WebSocket streaming |
| `POST` | `/admin/reload` | Reload model configuration |
| `GET` | `/health` | Service health check |

**Streaming features:**

- Server-Sent Events (SSE) and WebSocket transport
- Backend routing to GGUF (local), OpenAI, Ollama, or Anthropic
- Bearer token authentication
- Heartbeat keep-alive for long-running connections
- Model preloading and circuit breakers for backend resilience
- Session caching via SSE `Last-Event-ID` header

### Example: Streaming Chat

```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $TOKEN" \
  -d '{
    "model": "my-model",
    "messages": [{"role": "user", "content": "Hello"}],
    "stream": true
  }'
```

---

## C API

ABI provides C-compatible bindings with 36 exports. These allow embedding ABI in
C, C++, or any language with C FFI. See [C API Bindings](c-bindings.html) for the
full reference.

```c
#include "abi.h"

AbiFramework* fw = abi_init(NULL);
if (fw) {
    abi_gpu_dispatch_kernel(fw, "my_kernel", data, len);
    abi_deinit(fw);
}
```

---

## Import Conventions

- **Public API consumers** use `@import("abi")` -- never deep file paths.
- **Feature modules** cannot import `@import("abi")` (circular dependency) -- they
  use relative imports to `services/shared/`.
- **Internal sub-modules** import via their parent `mod.zig`.
- **Test files** in `src/services/tests/` import `@import("abi")` via the named module.
- **Feature test root** (`src/feature_test_root.zig`) can import both `features/` and
  `services/` directly.

---

## Related Pages

- [Configuration](configuration.html) -- Feature flags, build options, and runtime config
- [CLI Reference](cli.html) — 30 commands and 8 aliases
- [C API Bindings](c-bindings.html) -- FFI reference for the 36 C exports
- [Examples](examples.html) -- 36 runnable examples

## Zig Skill
Use [$zig](/Users/donaldfilimon/.codex/skills/zig/SKILL.md) for new Zig syntax improvements and validation guidance.
