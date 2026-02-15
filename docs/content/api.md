---
title: API Overview
description: Public API surface, import patterns, and HTTP endpoints
section: Reference
order: 17
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

| Namespace | Module | Description |
|-----------|--------|-------------|
| `abi.ai` | AI monolith | Full 17-submodule AI surface |
| `abi.ai_core` | AI Core | Agents, tools, prompts, personas, memory |
| `abi.inference` | Inference | LLM, embeddings, vision, streaming |
| `abi.training` | Training | Training pipelines, federated learning |
| `abi.reasoning` | Reasoning | Abbey, RAG, eval, templates, orchestration |
| `abi.gpu` | GPU | Kernel DSL, multi-GPU, 10 backends |
| `abi.database` | Database | WDBX vector database |
| `abi.network` | Network | Peer discovery, distributed compute |
| `abi.cache` | Cache | LRU/LFU/FIFO in-memory cache |
| `abi.storage` | Storage | Unified object storage |
| `abi.gateway` | Gateway | Radix-tree router, rate limiting, circuit breaker |
| `abi.search` | Search | Full-text BM25 search |
| `abi.messaging` | Messaging | Topic pub/sub, dead letter queues |
| `abi.pages` | Pages | URL routing, template rendering |
| `abi.connectors` | Connectors | 9 LLM providers + Discord + scheduler |
| `abi.shared` | Shared | Utils, SIMD, time, sync, security |
| `abi.simd` | SIMD | Vector operations, hardware detection |
| `abi.mcp` | MCP | Model Context Protocol server |
| `abi.acp` | ACP | Agent Communication Protocol |

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

The framework manages a lifecycle state machine:
`uninitialized -> initializing -> running -> stopping -> stopped` (or `failed`).

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

ABI provides C-compatible bindings in `bindings/c/src/abi_c.zig` with 36 exports.
These allow embedding ABI in C, C++, or any language with C FFI.

Key exports include framework lifecycle (`abi_init`, `abi_deinit`), GPU operations
(`abi_gpu_create_context`, `abi_gpu_dispatch_kernel`), SIMD (`abi_simd_vector_add`),
and database operations (`abi_db_open`, `abi_db_query`).

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
