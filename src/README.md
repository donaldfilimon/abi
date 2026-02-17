---
title: "Source Directory"
tags: [source, architecture, modules]
---
# Source Directory

> **Codebase Status:** Synced with repository as of 2026-02-14.

Core source modules of the ABI framework organized by function.

**Contents:** [Structure](#structure) · [Module Hierarchy](#module-hierarchy) · [v2 Integration](#v2-module-integration) · [Key Entry Points](#key-entry-points) · [Feature Gating](#feature-gating-pattern) · [See Also](#see-also)

## Structure

The codebase uses comptime feature gating: each feature module has `mod.zig`
(real implementation) and `stub.zig` (returns `error.FeatureDisabled`), selected
at compile time via `build_options`.

| Directory | Description |
|-----------|-------------|
| `api/` | Executable entry points (`main.zig`) |
| `core/` | Framework orchestration, config, registry, startup, WASM support |
| `features/` | 19 feature modules with comptime gating |
| `lib/` | Library and WASM entry points (`lib_main.zig`, `wasm_main.zig`) |
| `services/` | Always-available infrastructure: runtime, platform, shared, connectors, HA, tasks, tests |

## Module Hierarchy

```
src/
├── abi.zig                  # Public API root — comptime feature selection
├── root.zig                 # Root module (build system entry)
├── comptime_checks.zig      # Compile-time validation
│
├── api/                     # Executable entry points
│   └── main.zig
│
├── core/                    # Framework orchestration
│   ├── config/              # Unified configuration (per-feature configs)
│   ├── framework.zig        # Framework lifecycle state machine
│   ├── mod.zig              # Core module aggregator
│   ├── registry/            # Feature registry and discovery
│   ├── startup/             # Banner and bootstrap (banner.zig, mod.zig)
│   └── wasm/                # WASM-specific support (mod.zig, stub.zig)
│
├── features/                # Feature modules (mod.zig + stub.zig each)
│   ├── ai/                  # AI/ML — 17 submodules, 255+ files
│   ├── ai_core/             # AI agents, tools, prompts, personas, memory
│   ├── ai_inference/        # LLM, embeddings, vision, streaming
│   ├── ai_training/         # Training pipelines, federated learning
│   ├── ai_reasoning/        # Abbey, RAG, eval, templates, orchestration
│   ├── analytics/           # Event tracking and experiments
│   ├── auth/                # Security (re-exports shared/security/)
│   ├── cache/               # In-memory LRU/LFU, TTL, eviction
│   ├── cloud/               # Cloud adapters (AWS, GCP, Azure)
│   ├── database/            # Vector database (HNSW, clustering)
│   ├── gateway/             # API gateway: routing, rate limiting, circuit breaker
│   ├── gpu/                 # GPU compute — 11 backends (CUDA, Vulkan, Metal, WebGPU, TPU, …), DSL, multi-GPU
│   ├── messaging/           # Event bus, pub/sub, message queues
│   ├── mobile/              # Mobile platform support (defaults disabled)
│   ├── network/             # Distributed compute and networking
│   ├── observability/       # Metrics and tracing — gated by enable_profiling
│   ├── search/              # Full-text search with BM25 scoring
│   ├── storage/             # Unified file/object storage
│   └── web/                 # Web/HTTP framework and middleware
│
├── lib/                     # Library entry points
│   ├── lib_main.zig         # Shared/static library entry
│   └── wasm_main.zig        # WASM library entry
│
└── services/                # Always-available infrastructure
    ├── connectors/          # External API connectors (8 LLM + discord + scheduler)
    │   ├── mod.zig          #   Aggregator + loader helpers
    │   ├── anthropic.zig    #   Claude API
    │   ├── openai.zig       #   OpenAI API
    │   ├── ollama.zig       #   Local Ollama
    │   ├── huggingface.zig  #   HuggingFace Hub
    │   ├── cohere.zig       #   Cohere API
    │   ├── mistral.zig      #   Mistral API
    │   ├── lm_studio.zig    #   LM Studio (OpenAI-compatible)
    │   ├── vllm.zig         #   vLLM (OpenAI-compatible)
    │   ├── local_scheduler.zig  #  Job scheduling
    │   └── discord/         #   Discord bot (mod, types, utils, rest)
    │
    ├── mcp/                 # MCP server (JSON-RPC 2.0 over stdio, 5 WDBX tools)
    │
    ├── acp/                 # ACP server (agent communication protocol)
    │
    ├── ha/                  # High availability
    │   ├── consensus.zig    #   Raft consensus
    │   ├── failover.zig     #   Automatic failover
    │   ├── pitr.zig         #   Point-in-time recovery
    │   └── replication.zig  #   Data replication
    │
    ├── platform/            # Platform detection and capabilities
    │   ├── mod.zig          #   Platform aggregator + SIMD detection
    │   ├── android.zig      #   Android support
    │   ├── ios.zig          #   iOS support
    │   ├── linux.zig        #   Linux support
    │   ├── macos.zig        #   macOS support
    │   ├── wasm.zig         #   WASM support
    │   └── windows.zig      #   Windows support
    │
    ├── runtime/             # Scheduling, concurrency, memory
    │   ├── mod.zig          #   Runtime aggregator
    │   ├── concurrency/     #   Vyukov MPMC channel
    │   └── scheduling/      #   Work-stealing thread pool, DAG pipeline
    │
    ├── shared/              # Cross-cutting utilities
    │   ├── mod.zig          #   Shared aggregator
    │   ├── time.zig         #   Cross-platform time (sleepMs, getSeed)
    │   ├── tensor.zig       #   Tensor primitives (v2)
    │   ├── matrix.zig       #   Matrix primitives (v2)
    │   ├── simd.zig         #   SIMD kernels (euclidean, softmax, saxpy, etc.)
    │   ├── security/        #   16 security modules (auth, JWT, CORS, TLS, etc.)
    │   └── utils/           #   v2 utilities (see below)
    │
    ├── tasks/               # Task management and pipelines
    │   ├── scheduler.zig
    │   └── pipeline.zig
    │
    └── tests/               # Test infrastructure
        ├── mod.zig          #   Test root (1220 pass, 5 skip baseline)
        ├── parity/          #   DeclSpec mod/stub parity tests
        ├── integration/     #   Integration tests
        ├── stress/          #   Stress tests
        ├── chaos/           #   Chaos/fault injection tests
        ├── property/        #   Property-based tests
        └── <feature>/       #   Per-feature test suites (ai, analytics, cloud, ...)
```

## v2 Module Integration

Newer v2 primitives are wired through `shared` and `runtime`, not feature-local imports.

| Module | Location | Public Access |
|--------|----------|---------------|
| Primitive helpers | `shared/utils/v2_primitives.zig` | `abi.shared.utils.v2_primitives` |
| Structured errors | `shared/utils/structured_error.zig` | `abi.shared.utils.structured_error` |
| SwissMap | `shared/utils/swiss_map.zig` | `abi.shared.utils.swiss_map` |
| ABIX serialization | `shared/utils/abix_serialize.zig` | `abi.shared.utils.abix_serialize` |
| Profiler | `shared/utils/profiler.zig` | `abi.shared.utils.profiler` |
| Benchmark | `shared/utils/benchmark.zig` | `abi.shared.utils.benchmark` |
| Arena pool | `shared/utils/memory/arena_pool.zig` | `abi.shared.memory.ArenaPool` |
| Allocator combinators | `shared/utils/memory/combinators.zig` | `abi.shared.memory.FallbackAllocator` |
| Tensor ops | `shared/tensor.zig` | `abi.shared.tensor` |
| Matrix ops | `shared/matrix.zig` | `abi.shared.matrix` |
| SIMD kernels | `shared/simd.zig` | `abi.simd` / `abi.shared.simd` |
| MPMC channel | `runtime/concurrency/channel.zig` | `abi.runtime.Channel` |
| Thread pool | `runtime/scheduling/thread_pool.zig` | `abi.runtime.ThreadPool` |
| DAG pipeline | `runtime/scheduling/dag_pipeline.zig` | `abi.runtime.DagPipeline` |

## Key Entry Points

- **Public API**: `abi.zig` — `abi.init()`, `abi.initDefault()`, `Framework.builder()`, `abi.version()`
- **Configuration**: `core/config/mod.zig` — Unified `Config` struct with `Builder` API
- **Framework**: `core/framework.zig` — State machine: uninitialized -> initializing -> running -> stopping -> stopped
- **Runtime**: `services/runtime/mod.zig` — Always-available scheduling and concurrency (not feature-gated)

## Feature Gating Pattern

Every feature module in `features/` follows this comptime pattern in `abi.zig`:

```zig
pub const gpu = if (build_options.enable_gpu)
    @import("features/gpu/mod.zig")    // Real implementation
else
    @import("features/gpu/stub.zig");  // Returns error.FeatureDisabled
```

| Feature | Build Flag | Notes |
|---------|-----------|-------|
| `ai` | `enable_ai` | 17 submodules, each with own mod/stub |
| `analytics` | `enable_analytics` | Event tracking, experiments |
| `cloud` | `enable_cloud` | Cloud adapters (decoupled from web) |
| `database` | `enable_database` | Vector DB, HNSW, clustering |
| `gpu` | `enable_gpu` | 11 backends via `-Dgpu-backend=` (cuda, vulkan, metal, webgpu, tpu, …) |
| `network` | `enable_network` | Distributed compute |
| `observability` | `enable_profiling` | Metrics and tracing |
| `web` | `enable_web` | HTTP framework |

All default to `true` except `enable_mobile`. Validate combinations: `zig build validate-flags`.

## Module Integration Contract

Top-level feature modules expose:

1. **Types and functions** — re-exported from internal implementation files
2. **`Context` struct** — for `Framework` integration (init/deinit lifecycle)
3. **`isEnabled()` function** — returns `true` in mod.zig, `false` in stub.zig
4. **Matching `stub.zig`** — identical public signatures, returns `error.FeatureDisabled`

```zig
// Example: src/features/gpu/mod.zig
pub const Context = struct {
    allocator: std.mem.Allocator,
    config: config_module.GpuConfig,
    gpu: ?Gpu = null,

    pub fn init(allocator: std.mem.Allocator, cfg: config_module.GpuConfig) !*Context { ... }
    pub fn deinit(self: *Context) void { ... }
};

pub fn isEnabled() bool {
    return build_options.enable_gpu;
}
```

## Import Conventions

- Public API: `@import("abi")` — never deep file paths across module boundaries
- Feature modules cannot `@import("abi")` (circular) — use relative imports
- For time/sync in feature code: relative import to `services/shared/time.zig`
- Files that import `abi`: use `abi.shared.time` and `abi.shared.sync`

## See Also

- [CLAUDE.md](../CLAUDE.md) — Build commands, gotchas, architecture guide
- [AGENTS.md](../AGENTS.md) — Agent instructions, v2 notes, style guide
- [API Reference](../docs/api/index.md) — Auto-generated (`abi gendocs` or `zig build gendocs`)
