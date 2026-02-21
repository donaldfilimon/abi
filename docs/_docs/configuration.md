---
title: Configuration
description: Feature flags, config builder, and environment variables
section: Core
order: 2
---

# Configuration

ABI provides three layers of configuration: compile-time feature flags, the runtime
config builder, and environment variables for secrets and service endpoints.

## Config Builder Pattern

The framework supports fluent configuration through the builder API:

```zig
const abi = @import("abi");

var fw = try abi.Framework.builder(allocator)
    .withGpu(.{ .backend = .metal })
    .withAi(.{
        .llm = .{ .model_path = "./models/llama-7b.gguf" },
    })
    .withDatabase(.{ .path = "./data/vectors" })
    .withCache(.{
        .max_entries = 10_000,
        .eviction_policy = .lru,
    })
    .withGateway(.{
        .rate_limit = .{ .max_requests = 1000, .window_ms = 60_000 },
    })
    .build();
defer fw.deinit();
```

Alternatively, pass a struct literal directly:

```zig
var fw = try abi.Framework.init(allocator, .{
    .gpu = .{ .backend = .vulkan },
    .database = .{ .path = "./data" },
});
defer fw.deinit();
```

Or use defaults for all compile-time-enabled features:

```zig
var fw = try abi.initDefault(allocator);
defer fw.deinit();
```

See [Framework Lifecycle](framework.html) for a deep dive into all three initialization
patterns and the builder method reference.

## Feature Flags

All features are controlled by `-D` flags passed to `zig build`. Every flag defaults
to `true` except `-Denable-mobile`.

```bash
zig build -Denable-ai=true -Denable-gpu=false -Denable-mobile=true
```

### Complete Flag Reference (21 Modules)

| Feature Module | Build Flag | Default | Description |
|----------------|-----------|---------|-------------|
| `ai` | `-Denable-ai` | `true` | AI core functionality (monolith, 17 submodules with stubs + 6 without) |
| `ai_core` | `-Denable-ai` | `true` | Agents, tools, prompts, personas, memory |
| `inference` | `-Denable-llm` | `true` | LLM, embeddings, vision, streaming, transformer |
| `training` | `-Denable-training` | `true` | Training pipelines, federated learning |
| `reasoning` | `-Denable-reasoning` | `true` | Abbey reasoning, RAG, eval, templates, orchestration |
| `gpu` | `-Denable-gpu` | `true` | GPU acceleration and compute (10 backends) |
| `database` | `-Denable-database` | `true` | Vector database (WDBX) |
| `network` | `-Denable-network` | `true` | Distributed compute network |
| `web` | `-Denable-web` | `true` | Web/HTTP utilities |
| `analytics` | `-Denable-analytics` | `true` | Analytics event tracking |
| `cloud` | `-Denable-cloud` | `true` | Cloud provider integration (decoupled from web) |
| `auth` | `-Denable-auth` | `true` | Authentication and security (16 sub-modules) |
| `messaging` | `-Denable-messaging` | `true` | Event bus, pub/sub, dead letter queues |
| `cache` | `-Denable-cache` | `true` | In-memory LRU/LFU/FIFO caching |
| `storage` | `-Denable-storage` | `true` | Unified file/object storage |
| `search` | `-Denable-search` | `true` | Full-text BM25 search |
| `gateway` | `-Denable-gateway` | `true` | API gateway (routing, rate limiting, circuit breaker) |
| `pages` | `-Denable-pages` | `true` | Dashboard/UI pages with URL path routing |
| `observability` | `-Denable-profiling` | `true` | Metrics, tracing, alerting |
| `mobile` | `-Denable-mobile` | **`false`** | Mobile platform (lifecycle, sensors, notifications) |
| `benchmarks` | `-Denable-benchmarks` | `true` | Performance benchmarking and timing |

Note that `observability` uses `-Denable-profiling`, not `-Denable-observability`.
The `cloud` module has its own flag decoupled from `web`.

### Validate Flag Combinations

The build system validates 34 flag combinations to catch compilation issues early:

```bash
zig build validate-flags
```

This compiles solo-on, solo-off, all-on, and all-off combinations for every feature.

## GPU Backend Selection

Select the GPU backend with `-Dgpu-backend=`:

```bash
zig build -Dgpu-backend=metal         # macOS (recommended on Apple Silicon)
zig build -Dgpu-backend=cuda          # NVIDIA GPUs
zig build -Dgpu-backend=vulkan        # Cross-platform
zig build -Dgpu-backend=auto          # Auto-detect (default)
```

### All GPU Backends

| Backend | Flag Value | Notes |
|---------|-----------|-------|
| Auto-detect | `auto` | Default -- picks best available |
| None | `none` | CPU only |
| CUDA | `cuda` | NVIDIA, sm_XX arch detection |
| Vulkan | `vulkan` | Cross-platform |
| Metal | `metal` | macOS / iOS, MPS + CoreML integration |
| stdgpu | `stdgpu` | Standard GPU interface |
| WebGPU | `webgpu` | Browser and native WebGPU |
| WebGL2 | `webgl2` | Browser fallback |
| OpenGL | `opengl` | Legacy desktop |
| OpenGL ES | `opengles` | Mobile / embedded |
| TPU | `tpu` | Google TPU (runtime linked) |
| FPGA | `fpga` | FPGA accelerators |
| Simulated | `simulated` | Software fallback, always available for testing |

Multiple backends can be specified with commas:

```bash
zig build -Dgpu-backend=vulkan,cuda
```

Prefer one primary backend to avoid conflicts. On macOS, `metal` is the natural choice.

## Environment Variables

Environment variables configure connector endpoints and secrets at runtime. They are
read by `ConfigLoader` or directly by individual connectors.

### LLM Provider Keys

| Variable | Purpose | Required By |
|----------|---------|-------------|
| `ABI_OPENAI_API_KEY` | OpenAI API key | `abi.connectors.openai` |
| `ABI_ANTHROPIC_API_KEY` | Anthropic (Claude) API key | `abi.connectors.anthropic` |
| `ABI_HF_API_TOKEN` | HuggingFace API token | `abi.connectors.huggingface` |

### Local Model Servers

| Variable | Default | Purpose |
|----------|---------|---------|
| `ABI_OLLAMA_HOST` | `http://127.0.0.1:11434` | Ollama server host |
| `ABI_OLLAMA_MODEL` | -- | Default Ollama model name |
| `ABI_LM_STUDIO_HOST` | `http://localhost:1234` | LM Studio server host |
| `ABI_LM_STUDIO_MODEL` | -- | Default LM Studio model |
| `ABI_LM_STUDIO_API_KEY` | -- | LM Studio API key (optional) |
| `ABI_VLLM_HOST` | `http://localhost:8000` | vLLM server host |
| `ABI_VLLM_MODEL` | -- | Default vLLM model |
| `ABI_VLLM_API_KEY` | -- | vLLM API key (optional) |
| `ABI_MLX_HOST` | `http://localhost:8080` | MLX server host |
| `ABI_MLX_MODEL` | -- | Default MLX model |
| `ABI_MLX_API_KEY` | -- | MLX API key (optional) |

### Other Services

| Variable | Purpose |
|----------|---------|
| `ABI_MASTER_KEY` | Master encryption key for secrets (production) |
| `DISCORD_BOT_TOKEN` | Discord bot authentication token |

### Connector Availability

All connectors expose an `isAvailable()` function that checks for required environment
variables without allocation:

```zig
const abi = @import("abi");

if (abi.connectors.openai.isAvailable()) {
    // ABI_OPENAI_API_KEY is set and non-empty
    var client = try abi.connectors.openai.init(allocator);
    defer client.deinit();
}
```

Empty environment variables (e.g., `ABI_OPENAI_API_KEY=""`) are treated as unset.

## Config Types by Domain

Each feature module has a corresponding config struct in `src/core/config/`:

| Config Type | Source File | Builder Method |
|-------------|------------|----------------|
| `GpuConfig` | `config/gpu.zig` | `.withGpu(...)` |
| `AiConfig` | `config/ai.zig` | `.withAi(...)` |
| `DatabaseConfig` | `config/database.zig` | `.withDatabase(...)` |
| `NetworkConfig` | `config/network.zig` | `.withNetwork(...)` |
| `ObservabilityConfig` | `config/observability.zig` | `.withObservability(...)` |
| `WebConfig` | `config/web.zig` | `.withWeb(...)` |
| `CloudConfig` | `config/cloud.zig` | `.withCloud(...)` |
| `AnalyticsConfig` | `config/analytics.zig` | `.withAnalytics(...)` |
| `AuthConfig` | `config/auth.zig` | `.withAuth(...)` |
| `MessagingConfig` | `config/messaging.zig` | `.withMessaging(...)` |
| `CacheConfig` | `config/cache.zig` | `.withCache(...)` |
| `StorageConfig` | `config/storage.zig` | `.withStorage(...)` |
| `SearchConfig` | `config/search.zig` | `.withSearch(...)` |
| `GatewayConfig` | `config/gateway.zig` | `.withGateway(...)` |
| `PagesConfig` | `config/pages.zig` | `.withPages(...)` |
| `MobileConfig` | `config/mobile.zig` | `.withMobile(...)` |
| `BenchmarksConfig` | `config/benchmarks.zig` | `.withBenchmarks(...)` |

All config fields in the unified `Config` struct are optional (`?ConfigType`). A `null`
value means that feature uses its defaults or is not explicitly configured.

## Further Reading

- [Architecture](architecture.html) -- module hierarchy and comptime gating
- [Framework Lifecycle](framework.html) -- init patterns, state machine, builder details
- [CLI](cli.html) -- runtime configuration commands (`config init`, `config show`, `config validate`)

## Zig Skill
Use [$zig](/Users/donaldfilimon/.codex/skills/zig/SKILL.md) for new Zig syntax improvements and validation guidance.
