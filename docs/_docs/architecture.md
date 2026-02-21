---
title: Architecture
description: Module hierarchy, comptime feature gating, and framework lifecycle
section: Core
order: 1
---

# Architecture

ABI's architecture is built around three ideas: a layered module hierarchy, compile-time
feature gating with zero-cost stubs, and a state-machine framework lifecycle.

## Module Hierarchy

```
build.zig                   Top-level build (delegates to build/)
build/                      Split build system
  options.zig                 Feature flag definitions
  modules.zig                 Module registration
  flags.zig                   Flag combination validation (34 combos)
  targets.zig                 Target platform logic
  gpu.zig                     GPU backend selection
  mobile.zig                  Mobile platform handling
  wasm.zig                    WebAssembly target

src/abi.zig                 Public API entry point, comptime feature selection
src/core/
  framework.zig               Framework lifecycle (state machine)
  config/                     Unified config builder (17 domain configs)
  registry/                   Feature registry (runtime enable/disable)
  errors.zig                  Composable error hierarchy

src/features/<name>/        One directory per feature module (21 modules)
  mod.zig                     Real implementation
  stub.zig                    Disabled stub (matching signatures)

src/services/               Always-available infrastructure
  runtime/                    Thread pool, channels, DAG scheduler
  platform/                   OS detection, abstraction
  shared/                     Utils, SIMD, time, sync, security (16 modules)
  ha/                         Replication, backup, PITR
  tasks/                      Task management
  mcp/                        MCP server (JSON-RPC 2.0 over stdio)
  acp/                        Agent Communication Protocol
  connectors/                 9 LLM providers + Discord + scheduler

tools/cli/                  CLI entry point
  commands/                   30 command modules + 8 aliases
```

### Layer Rules

1. **`src/abi.zig`** is the only public API surface. External code imports `@import("abi")`.
2. **Feature modules** (`src/features/`) cannot import `@import("abi")` (circular dependency).
   They use relative imports to `services/shared/` for utilities.
3. **Services** (`src/services/`) are always compiled. They provide runtime, platform,
   and shared infrastructure that feature modules depend on.
4. **CLI tools** (`tools/cli/`) import `@import("abi")` and are leaf consumers of the API.

## Comptime Feature Gating

The central pattern in `src/abi.zig` selects between real and stub implementations
at compile time:

```zig
const build_options = @import("build_options");

// Real implementation when enabled, stub when disabled
pub const gpu = if (build_options.enable_gpu)
    @import("features/gpu/mod.zig")
else
    @import("features/gpu/stub.zig");

pub const database = if (build_options.enable_database)
    @import("features/database/mod.zig")
else
    @import("features/database/stub.zig");

// ... repeated for all 21 feature modules
```

This pattern has two important properties:

- **Zero binary overhead for disabled features.** Zig's lazy compilation means the
  stub path never compiles the real module's code.
- **Type-safe feature boundaries.** Both `mod.zig` and `stub.zig` export identical
  public signatures. Code that calls `abi.gpu.someFunction()` compiles whether GPU
  is enabled or not.

### The Stub Pattern

Every stub follows the same structure:

```zig
// features/gpu/stub.zig
const std = @import("std");

pub const GpuContext = struct {
    pub fn init(allocator: std.mem.Allocator) !GpuContext {
        _ = allocator;
        return error.FeatureDisabled;
    }
    // ... same pub fn signatures as mod.zig
};
```

Stubs return `error.FeatureDisabled` for fallible functions, `null` for optional
returns, and discard all parameters with `_ = param;`. When you add or change a
public declaration in `mod.zig`, you must update `stub.zig` to match.

### All 21 Gated Modules

| Module | Build Flag | Namespace | Description |
|--------|-----------|-----------|-------------|
| AI (monolith) | `enable_ai` | `abi.ai` | Full AI module (17 submodules with stubs + 6 without) |
| AI Core | `enable_ai` | `abi.ai_core` | Agents, tools, prompts, personas, memory |
| Inference | `enable_llm` | `abi.inference` | LLM, embeddings, vision, streaming, transformer |
| Training | `enable_training` | `abi.training` | Training pipelines, federated learning |
| Reasoning | `enable_reasoning` | `abi.reasoning` | Abbey, RAG, eval, templates, orchestration |
| GPU | `enable_gpu` | `abi.gpu` | 10 backends, kernel DSL, multi-GPU |
| Database | `enable_database` | `abi.database` | WDBX vector database |
| Network | `enable_network` | `abi.network` | Distributed compute, peer discovery |
| Web | `enable_web` | `abi.web` | Web/HTTP utilities |
| Analytics | `enable_analytics` | `abi.analytics` | Event tracking and analytics |
| Cloud | `enable_cloud` | `abi.cloud` | Cloud provider integration |
| Auth | `enable_auth` | `abi.auth` | Authentication and security (16 sub-modules) |
| Messaging | `enable_messaging` | `abi.messaging` | Event bus, pub/sub, dead letter queues |
| Cache | `enable_cache` | `abi.cache` | In-memory LRU/LFU/FIFO caching |
| Storage | `enable_storage` | `abi.storage` | Unified file/object storage |
| Search | `enable_search` | `abi.search` | Full-text BM25 search |
| Gateway | `enable_gateway` | `abi.gateway` | API gateway (routing, rate limiting, circuit breaker) |
| Pages | `enable_pages` | `abi.pages` | Dashboard/UI with URL path routing |
| Observability | `enable_profiling` | `abi.observability` | Metrics, tracing, alerting |
| Mobile | `enable_mobile` | `abi.mobile` | Mobile platform (lifecycle, sensors, notifications) |
| Benchmarks | `enable_benchmarks` | `abi.benchmarks` | Performance benchmarking and timing |

## Framework Lifecycle

The `Framework` struct in `src/core/framework.zig` manages feature initialization
through a state machine:

```
uninitialized --> initializing --> running --> stopping --> stopped
                      |                                      ^
                      +---------> failed --------------------+
```

| State | Description |
|-------|-------------|
| `uninitialized` | Before `init()` is called |
| `initializing` | Features being started in dependency order |
| `running` | Normal operation -- all requested features available |
| `stopping` | Graceful shutdown in progress |
| `stopped` | All resources released |
| `failed` | Initialization error (transitions to stopped on deinit) |

### Initialization Patterns

**Default initialization** enables all compile-time features with default configs:

```zig
const abi = @import("abi");

var fw = try abi.initDefault(allocator);
defer fw.deinit();
```

**Custom configuration** lets you select specific features and tune their settings:

```zig
var fw = try abi.Framework.init(allocator, .{
    .gpu = .{ .backend = .vulkan },
    .ai = .{ .llm = .{ .model_path = "./model.gguf" } },
    .database = .{ .path = "./data" },
});
defer fw.deinit();
```

**Builder pattern** provides a fluent API:

```zig
var fw = try abi.Framework.builder(allocator)
    .withGpu(.{ .backend = .vulkan })
    .withAiDefaults()
    .withDatabaseDefaults()
    .build();
defer fw.deinit();
```

For a deep dive into initialization, state transitions, and lifecycle hooks, see
[Framework Lifecycle](framework.html).

### Feature Access at Runtime

```zig
// Check if a feature is enabled
if (fw.isEnabled(.gpu)) {
    const gpu_ctx = try fw.getGpu();
    // Use GPU features...
}

// Runtime context is always available
const rt = fw.getRuntime();
```

## v2 Infrastructure

Core infrastructure modules live under `services/shared/` and `services/runtime/`.
These are always compiled (not feature-gated) and provide the foundation that feature
modules build on.

| Component | Location | Access Path |
|-----------|----------|-------------|
| Swiss hash map | `shared/utils/swiss_map.zig` | `abi.shared.utils.swiss_map` |
| Structured errors | `shared/utils/structured_error.zig` | `abi.shared.utils.structured_error` |
| Profiler | `shared/utils/profiler.zig` | `abi.shared.utils.profiler` |
| Benchmark runner | `shared/utils/benchmark.zig` | `abi.shared.utils.benchmark` |
| Arena pool allocator | `shared/utils/memory/arena_pool.zig` | `abi.shared.memory.ArenaPool` |
| Combinator allocators | `shared/utils/memory/combinators.zig` | `abi.shared.memory.FallbackAllocator` |
| Vyukov MPMC channel | `runtime/concurrency/channel.zig` | `abi.runtime.Channel` |
| Work-stealing thread pool | `runtime/scheduling/thread_pool.zig` | `abi.runtime.ThreadPool` |
| DAG pipeline scheduler | `runtime/scheduling/dag_pipeline.zig` | `abi.runtime.DagPipeline` |
| Radix tree | `shared/utils/radix_tree.zig` | Used by gateway + pages for URL routing |
| Circuit breaker | `shared/resilience/circuit_breaker.zig` | `abi.shared.resilience.{Simple,Atomic,Mutex}CircuitBreaker` |
| SIMD operations | `shared/simd.zig` (5 submodules) | `abi.simd` |

## Import Convention

| Context | Import Style |
|---------|-------------|
| External code / CLI tools | `@import("abi")` |
| Feature modules (`src/features/`) | Relative path: `@import("../../services/shared/utils.zig")` |
| Internal sub-modules | Via parent `mod.zig` |
| Cross-module types | Through `abi.shared.*` or `abi.runtime.*` |

Feature modules must never use `@import("abi")` -- this creates a circular dependency
because `abi.zig` itself imports the feature modules.

## Test Architecture

ABI has two separate test roots because of Zig's module path restrictions:

| Test Root | Path | Purpose |
|-----------|------|---------|
| Main tests | `src/services/tests/mod.zig` | Integration, stress, chaos, parity tests (1270 pass, 5 skip) |
| Feature tests | `src/feature_test_root.zig` | Inline tests inside feature/service modules (1535 pass) |

The feature test root sits at `src/` level so it can reach both `features/` and
`services/` subdirectories. Test discovery uses `test { _ = @import(...); }` blocks --
`comptime {}` blocks do not trigger test discovery.

## Further Reading

- [Configuration](configuration.html) -- all build flags and environment variables
- [Framework Lifecycle](framework.html) -- deep dive into state machine, init patterns, and lifecycle hooks
- [CLI](cli.html) — 30 commands + 8 aliases for system management

## Zig Skill
Use [$zig](/Users/donaldfilimon/.codex/skills/zig/SKILL.md) for new Zig syntax improvements and validation guidance.
