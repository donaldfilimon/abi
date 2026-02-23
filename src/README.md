# Source Directory

Core source modules of the ABI framework organized by function.

## Structure

The codebase uses comptime feature gating: each feature module has `mod.zig`
(real implementation) and `stub.zig` (returns `error.FeatureDisabled`), selected
at compile time via `build_options`.

| Directory | Description |
|-----------|-------------|
| `api/` | Executable entry points (`main.zig`) |
| `core/` | Framework orchestration, config, registry, startup |
| `features/` | 24 feature modules with comptime gating |
| `services/` | Always-available infrastructure: runtime, platform, shared, connectors, HA, tasks, tests |

## Module Hierarchy

```
src/
├── abi.zig                  # Public API root — comptime feature selection
├── feature_test_root.zig    # Feature-gated test discovery root
│
├── api/                     # Executable entry points
│   └── main.zig
│
├── core/                    # Framework orchestration
│   ├── config/              # Unified configuration (per-feature configs + stubs/)
│   ├── errors.zig           # Error hierarchy
│   ├── feature_catalog.zig  # Canonical feature definitions (24 entries)
│   ├── framework.zig        # Framework lifecycle state machine
│   ├── framework/           # Framework internals (builder, context_init, lifecycle, shutdown, state_machine)
│   ├── health.zig           # Health monitoring
│   ├── mod.zig              # Core module aggregator
│   ├── registry/            # Feature registry and discovery
│   ├── startup/             # Banner and bootstrap
│   └── stub_context.zig     # Reusable stub context helpers
│
├── features/                # Feature modules (mod.zig + stub.zig each)
│   ├── ai/                  # AI/ML — 17 submodules, 255+ files
│   ├── ai_core/             # AI agents, tools, prompts, personas, memory
│   ├── ai_inference/        # LLM, embeddings, vision, streaming
│   ├── ai_training/         # Training pipelines, federated learning
│   ├── ai_reasoning/        # Abbey, RAG, eval, templates, orchestration
│   ├── analytics/           # Event tracking and experiments
│   ├── auth/                # Security (re-exports shared/security/)
│   ├── benchmarks/          # Built-in benchmark suite
│   ├── cache/               # In-memory LRU/LFU, TTL, eviction
│   ├── cloud/               # Cloud adapters (AWS, GCP, Azure)
│   ├── database/            # Vector database (HNSW, clustering)
│   ├── gateway/             # API gateway: routing, rate limiting, circuit breaker
│   ├── gpu/                 # GPU compute — 10 backends, DSL, multi-GPU
│   ├── messaging/           # Event bus, pub/sub, message queues
│   ├── mobile/              # Mobile platform support (defaults disabled)
│   ├── network/             # Distributed compute and networking
│   ├── observability/       # Metrics and tracing — gated by enable_profiling
│   ├── pages/               # Dashboard/UI pages with URL path routing
│   ├── search/              # Full-text search with BM25 scoring
│   ├── storage/             # Unified file/object storage
│   └── web/                 # Web/HTTP framework and middleware
│
└── services/                # Always-available infrastructure
    ├── connectors/          # External API connectors (15 LLM + discord + scheduler)
    │   ├── mod.zig          #   Aggregator + loader helpers
    │   ├── anthropic.zig    #   Claude API
    │   ├── openai.zig       #   OpenAI API
    │   ├── ollama.zig       #   Local Ollama
    │   ├── huggingface.zig  #   HuggingFace Hub
    │   ├── cohere.zig       #   Cohere API
    │   ├── mistral.zig      #   Mistral API
    │   ├── lm_studio.zig    #   LM Studio (OpenAI-compatible)
    │   ├── vllm.zig         #   vLLM (OpenAI-compatible)
    │   ├── mlx.zig          #   MLX (OpenAI-compatible)
    │   ├── local_scheduler.zig  #  Job scheduling
    │   └── discord/         #   Discord bot (mod, types, utils, rest)
    │
    ├── mcp/                 # MCP server (JSON-RPC 2.0 over stdio, 5 WDBX tools)
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
    │   ├── android.zig, ios.zig, linux.zig, macos.zig, wasm.zig, windows.zig
    │
    ├── runtime/             # Scheduling, concurrency, memory
    │   ├── concurrency/     #   Vyukov MPMC channel
    │   ├── engine/          #   Runtime engine
    │   ├── memory/          #   Memory management
    │   ├── scheduling/      #   Work-stealing thread pool, DAG pipeline
    │   └── stubs/           #   Runtime stubs
    │
    ├── shared/              # Cross-cutting utilities
    │   ├── resilience/      #   Circuit breakers (atomic, mutex, simple)
    │   ├── security/        #   16 security modules (auth, JWT, CORS, TLS, etc.)
    │   ├── simd/            #   SIMD kernels
    │   └── utils/           #   v2 utilities, crypto, encoding, fs, http, json, memory, net
    │
    ├── tasks/               # Task management and pipelines
    └── tests/               # Test infrastructure
        ├── mod.zig          #   Test root (1270 pass, 5 skip baseline)
        ├── parity/          #   DeclSpec mod/stub parity tests
        ├── integration/     #   Integration tests
        ├── stress/          #   Stress tests
        ├── chaos/           #   Chaos/fault injection tests
        ├── property/        #   Property-based tests
        └── <feature>/       #   Per-feature test suites
```

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

All 24 features default to `true` except `enable_mobile`. Validate combinations: `zig build validate-flags`.

## Import Conventions

- Public API: `@import("abi")` — never deep file paths across module boundaries
- Feature modules cannot `@import("abi")` (circular) — use relative imports
- For time/sync in feature code: relative import to `services/shared/time.zig`

## See Also

- [CLAUDE.md](../CLAUDE.md) — Build commands, gotchas, architecture guide
- [AGENTS.md](../AGENTS.md) — Agent instructions, v2 notes, style guide
- [API Reference](../docs/api/index.md) — Auto-generated (`abi gendocs` or `zig build gendocs`)

## Zig Skill
Use [$zig](/Users/donaldfilimon/.codex/skills/zig/SKILL.md) for ABI Zig 0.16-dev syntax improvements, modular build layout guidance, and targeted validation workflows.
