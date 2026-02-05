# CLAUDE.md

Comprehensive guidance for Claude Code working with the ABI Framework. Read
`AGENTS.md` first for baseline rules; this file adds deeper details and examples.
`GEMINI.md` is a condensed quick reference.

| | |
|---|---|
| **Version** | 0.4.0 |
| **Entry Point** | `src/abi.zig` |
| **Zig Required** | 0.16.x (`0.16.0-dev.2471+e9eadee00` or later) |
| **Version File** | `.zigversion` (pins exact Zig version) |

## Quick Start

| Command | Purpose |
|---------|---------|
| `zig build` | Build the project |
| `zig build test --summary all` | Run full test suite |
| `zig fmt .` | Format code (required after edits) |
| `zig build run -- --help` | CLI help |
| `zig test src/file.zig --test-filter "pattern"` | Focused tests |

---

## Critical Gotchas

| Category | Issue | Solution |
|----------|-------|----------|
| Zig 0.16 | File system ops | Use `std.Io.Dir.cwd()` not `std.fs.cwd()` |
| Zig 0.16 | Timer API | Use `std.time.Timer.start()` not `Instant.now()` |
| Zig 0.16 | ArrayListUnmanaged | Use `.empty` not `.init()` |
| Zig 0.16 | Format specifiers | Use `{t}` for errors/enums |
| Feature | Stub/real sync | Update `mod.zig` and `stub.zig` together |
| Build | `--test-filter` syntax | Use `zig test file.zig --test-filter` |
| Build | build.zig file checks | Use `pathExists()` helper |
| GPU | Backend conflicts | Prefer one primary backend |
| WASM | Limitations | `database`, `network`, `gpu` auto-disabled |
| Sleep | Cross-platform | Prefer `abi.shared.time.sleepMs()` / `sleepNs()` |
| HTTP | Server init | Use `&reader.interface` and `&writer.interface` |

---

## Zig 0.16 API Patterns

### I/O Backend Initialization (Critical)

```zig
const std = @import("std");

var io_backend = std.Io.Threaded.init(allocator, .{
    .environ = std.process.Environ.empty, // .empty for library, init.environ for CLI
});
defer io_backend.deinit();
const io = io_backend.io();

const content = try std.Io.Dir.cwd().readFileAlloc(
    io,
    path,
    allocator,
    .limited(10 * 1024 * 1024),
);
defer allocator.free(content);
```

### Other Changes

```zig
// Error/enum formatting: use {t}
std.debug.print("Error: {t}, State: {t}", .{err, state});

// ArrayListUnmanaged
var list = std.ArrayListUnmanaged(u8).empty;

// Timing
var timer = std.time.Timer.start() catch return error.TimerFailed;
const elapsed_ns = timer.read();

// Sleep (preferred for cross-platform)
const abi = @import("abi");
abi.shared.time.sleepMs(10);

// HTTP server init
var server: std.http.Server = .init(
    &connection_reader.interface,
    &connection_writer.interface,
);
```

---

## Architecture

Flat domain structure with unified configuration. Each feature has `mod.zig`
(real) and `stub.zig` (feature-gated placeholder).

```
src/
├── abi.zig              # Public API module root
├── api/                 # Entry points
│   └── main.zig         # CLI entrypoint fallback
├── core/                # Framework orchestration and config
│   ├── config/          # Unified config (Config + Builder + per-feature)
│   ├── framework.zig    # Lifecycle states, builder pattern
│   ├── flags.zig        # Feature flags
│   └── registry/        # Feature registry (comptime, runtime, dynamic)
├── features/            # Feature modules
│   ├── ai/              # AI module (280+ files, see AI section below)
│   ├── gpu/             # GPU acceleration (backends, kernels, DSL, codegen)
│   ├── database/        # Vector database (WDBX with HNSW/IVF-PQ)
│   ├── network/         # Distributed compute, Raft consensus
│   ├── observability/   # Metrics, tracing, system info
│   └── web/             # Web/HTTP server support
└── services/            # Shared infrastructure
    ├── runtime/         # Task execution, concurrency, scheduling, memory
    ├── platform/        # Platform abstraction layer
    ├── shared/          # Utilities, security (15 modules), SIMD
    ├── connectors/      # External connectors (Discord, etc.)
    ├── cloud/           # Cloud provider integrations
    ├── ha/              # High availability (failover, replication, PITR)
    ├── tasks/           # Task management system
    └── tests/           # Test suite (chaos, e2e, integration, parity, property, stress)
```

Import rules:
- Public API imports use `@import("abi")`.
- Nested modules import via their parent `mod.zig`.

---

## AI Module Structure

The AI module is the largest in the codebase (280+ files). It contains 23 submodules:

### Submodules with Stub Pattern (mod.zig + stub.zig)

These require keeping both files in sync when modified:

| Submodule | Purpose |
|-----------|---------|
| `agents/` | Agent runtime and behaviors |
| `database/` | AI-specific database utilities |
| `documents/` | Document processing |
| `embeddings/` | Text/vector embeddings |
| `eval/` | Evaluation metrics (BLEU, ROUGE, Perplexity) |
| `explore/` | Code exploration capabilities |
| `llm/` | Local LLM inference |
| `memory/` | Agent memory systems |
| `models/` | Model management and downloading |
| `multi_agent/` | Multi-agent coordination |
| `orchestration/` | Workflow orchestration |
| `personas/` | Agent personas and personalities |
| `rag/` | Retrieval-augmented generation |
| `streaming/` | Streaming generation (SSE, WebSocket) |
| `templates/` | Prompt templates |
| `training/` | Training pipelines |
| `vision/` | Vision/image processing |

### Submodules without Stubs

| Submodule | Purpose |
|-----------|---------|
| `abbey/` | **Advanced reasoning system** - meta-learning, self-reflection, theory of mind, neural attention, episodic/semantic/working memory |
| `core/` | Shared AI types and configuration |
| `prompts/` | Prompt building and rendering |
| `tools/` | Agent tools (file, edit, search, Discord, OS) |
| `transformer/` | Transformer model implementations |
| `federated/` | Federated learning support |

### Abbey Reasoning System

The `abbey/` submodule provides advanced cognitive capabilities:

```
src/features/ai/abbey/
├── advanced/        # Meta-learning, self-reflection, theory of mind, compositional reasoning
├── neural/          # Attention mechanisms, learning, GPU ops, tensor operations
├── memory/          # Episodic, semantic, working memory systems
├── mod.zig          # Main module
├── emotions.zig     # Emotional modeling
├── calibration.zig  # Confidence calibration
├── reasoning.zig    # Reasoning engine
└── context.zig      # Context management
```

---

## GPU Module Structure

The GPU module supports 11 backends with a unified API:

```
src/features/gpu/
├── backends/        # 16 backend implementations + vtable abstractions
│   ├── cuda/        # CUDA with cuBLAS, NVRTC, custom kernels
│   ├── vulkan/      # Vulkan compute
│   ├── metal/       # Metal with Accelerate/AMX
│   ├── fpga/        # Intel/Xilinx FPGA (attention, matmul, distance, kv_cache)
│   └── ...          # stdgpu, webgpu, opengl, opengles, webgl2, fallback, simulated
├── dsl/             # Kernel DSL with codegen, optimizer, SPIR-V
├── kernels/         # Activation, batch, elementwise, linalg, matrix, normalization, reduction, vision
├── mega/            # Multi-GPU orchestration (coordinator, failover, hybrid, metrics, power, queue, scheduler)
└── peer_transfer/   # GPU-to-GPU data movement across backends
```

---

## Feature Flags

```bash
zig build -Denable-ai=true -Denable-gpu=false
zig build -Dgpu-backend=vulkan,cuda
```

| Flag | Default | Description |
|------|---------|-------------|
| `-Denable-ai` | true | AI agent system |
| `-Denable-llm` | true | Local LLM inference (requires AI) |
| `-Denable-vision` | true | Vision/image processing (requires AI) |
| `-Denable-explore` | true | Code exploration (requires AI) |
| `-Denable-gpu` | true | GPU acceleration |
| `-Denable-database` | true | Vector database |
| `-Denable-network` | true | Distributed compute |
| `-Denable-web` | true | Web/HTTP support |
| `-Denable-profiling` | true | Metrics/tracing |
| `-Denable-mobile` | false | Mobile cross-compilation |

### GPU Backends

All supported backends (comma-separated for multiple):

| Backend | Description |
|---------|-------------|
| `auto` | Auto-detect best available |
| `none` | Disable GPU entirely |
| `cuda` | NVIDIA CUDA |
| `vulkan` | Vulkan compute (default on desktop) |
| `metal` | Apple Metal with Accelerate/AMX |
| `stdgpu` | Zig std.gpu integration |
| `webgpu` | WebGPU (default for web) |
| `webgl2` | WebGL 2.0 (default for web) |
| `opengl` | OpenGL compute |
| `opengles` | OpenGL ES |
| `fpga` | FPGA (Intel/Xilinx) |

```bash
# Examples
zig build -Dgpu-backend=cuda,vulkan
zig build -Dgpu-backend=metal
zig build -Dgpu-backend=auto
```

---

## Services Layer

### Runtime Services (`src/services/runtime/`)

| Component | Purpose |
|-----------|---------|
| `concurrency/` | Chase-Lev queue, epoch-based memory, lock-free structures, MPMC queue, priority queue |
| `engine/` | Task execution with NUMA support, work stealing, result cache |
| `scheduling/` | Async execution, futures, task groups, cancellation |

### Shared Services (`src/services/shared/`)

| Component | Purpose |
|-----------|---------|
| `security/` | 15 modules: API keys, audit, certificates, CORS, encryption, JWT, mTLS, RBAC, rate limiting, TLS, validation |
| `utils/` | 40+ utilities: crypto, encoding, filesystem, HTTP, JSON, retry |
| `memory/` | Pools, alignment, zerocopy, ring buffers, thread-cache |
| `simd.zig` | SIMD acceleration (vectorAdd, vectorDot, L2Norm, cosineSimilarity) |
| `logging.zig` | Structured logging |
| `plugins.zig` | Plugin management |

---

## CLI Structure

The CLI is located in `tools/cli/` with 24 commands:

```
tools/cli/
├── main.zig              # CLI entry point
├── mod.zig               # Module exports
├── commands/             # Command implementations
│   ├── agent.zig         # AI agent chat
│   ├── bench.zig         # Benchmarking
│   ├── completions.zig   # Shell completions
│   ├── config.zig        # Configuration management
│   ├── convert.zig       # Format conversion
│   ├── db.zig            # Database operations
│   ├── discord.zig       # Discord integration
│   ├── embed.zig         # Embedding generation
│   ├── explore.zig       # Code exploration
│   ├── gpu.zig           # GPU status/operations
│   ├── gpu_dashboard.zig # GPU monitoring TUI
│   ├── llm.zig           # LLM inference
│   ├── model.zig         # Model management
│   ├── multi_agent.zig   # Multi-agent orchestration
│   ├── network.zig       # Network operations
│   ├── plugins.zig       # Plugin management
│   ├── profile.zig       # Profiling
│   ├── simd.zig          # SIMD operations
│   ├── system_info.zig   # System information
│   ├── task.zig          # Task management
│   ├── toolchain.zig     # Toolchain info
│   ├── train.zig         # Training pipelines
│   └── tui.zig           # TUI launcher
└── tui/                  # TUI components
    ├── async_loop.zig
    ├── events.zig
    ├── gpu_monitor.zig
    ├── streaming_dashboard.zig
    ├── training_metrics.zig
    ├── widgets/
    └── themes/
```

### CLI Command Groups

| Group | Commands |
|-------|----------|
| AI/LLM | `agent`, `llm`, `model`, `embed`, `explore`, `multi_agent` |
| Database | `db`, `convert` |
| GPU | `gpu`, `gpu-dashboard` |
| Training | `train` |
| System | `config`, `system-info`, `simd`, `bench`, `profile`, `task`, `toolchain`, `network`, `plugins`, `completions`, `tui` |
| External | `discord` |

---

## Test Infrastructure

Tests are organized in `src/services/tests/`:

```
src/services/tests/
├── mod.zig          # Test suite entry point
├── chaos/           # Chaos testing (fault injection)
├── e2e/             # End-to-end tests
├── integration/     # Integration tests
├── parity/          # Real vs stub parity testing
├── property/        # Property-based testing
├── stress/          # Stress/load testing
└── *.zig            # Domain-specific tests (cloud, concurrency, connectors, database, LLM, etc.)
```

Unit tests live alongside code as `*_test.zig` files.

### Test Commands

```bash
# Full test suite
zig build test --summary all

# Specific test categories
zig test src/services/tests/integration/mod.zig
zig test src/services/tests/stress/mod.zig
zig test src/services/tests/chaos/mod.zig
zig test src/services/tests/parity/mod.zig
zig test src/services/tests/property/mod.zig

# Filtered tests
zig test src/file.zig --test-filter "pattern"
```

Skip hardware-gated tests with `error.SkipZigTest`.

---

## Common Workflows

```bash
# Build + test
zig build test --summary all

# Single file test
zig test src/services/runtime/engine/engine.zig --test-filter "pattern"

# Full check
zig build full-check

# CLI smoke tests
zig build cli-tests

# Benchmarks
zig build benchmarks
```

---

## Configuration System

Unified `Config` with a builder pattern:

```zig
const abi = @import("abi");

var builder = abi.config.Builder.init(allocator);
const config = builder
    .withAiDefaults()
    .withGpuDefaults()
    .withDatabaseDefaults()
    .build();
```

Configuration modules in `src/core/config/`:
- `ai_config.zig`
- `gpu_config.zig`
- `database_config.zig`
- `network_config.zig`
- `observability_config.zig`
- `web_config.zig`
- `plugin_config.zig`

---

## Environment Variables

| Variable | Purpose |
|----------|---------|
| `ABI_OPENAI_API_KEY` | OpenAI connector |
| `ABI_ANTHROPIC_API_KEY` | Claude connector |
| `ABI_OLLAMA_HOST` | Ollama host (default: `http://127.0.0.1:11434`) |
| `ABI_OLLAMA_MODEL` | Default Ollama model |
| `ABI_HF_API_TOKEN` | HuggingFace token |
| `ABI_MASTER_KEY` | Secrets encryption (production) |
| `DISCORD_BOT_TOKEN` | Discord bot token |

See `README.md` for the full list.

---

## Language Bindings

Bindings live in `bindings/`. Build the C shared library first:

```bash
cd bindings/c && zig build
# Outputs: bindings/c/zig-out/lib/libabi.{dylib,so}
```

| Language | Location | Setup |
|----------|----------|-------|
| C | `bindings/c/` | `#include "abi.h"` |
| Python | `bindings/python/` | `pip install -e .` (depends on C lib) |
| Go | `bindings/go/` | `go get` (depends on C lib) |
| JavaScript | `bindings/js/` | Node.js FFI wrapper |
| Rust | `bindings/rust/` | Rust FFI crate |

Set library path: `LD_LIBRARY_PATH` (Linux) or `DYLD_LIBRARY_PATH` (macOS) to `bindings/c/zig-out/lib`.

---

## Examples

18 examples are provided in `examples/`:

| Example | Purpose |
|---------|---------|
| `agent.zig` | AI agent chat |
| `compute.zig` | Compute engine usage |
| `concurrency.zig` | Concurrent task execution |
| `config.zig` | Configuration loading |
| `database.zig` | Vector database operations |
| `discord.zig` | Discord bot integration |
| `embeddings.zig` | Text embeddings |
| `gpu.zig` | GPU acceleration |
| `ha.zig` | High availability |
| `hello.zig` | Hello world |
| `llm.zig` | LLM inference |
| `network.zig` | Distributed networking |
| `observability.zig` | Metrics and tracing |
| `orchestration.zig` | Multi-agent orchestration |
| `registry.zig` | Feature registry |
| `streaming.zig` | Streaming generation |
| `train_ava.zig` | Training with Ava |
| `training.zig` | Training pipelines |

Build and run examples:

```bash
zig build examples
./zig-out/bin/example-hello
./zig-out/bin/example-agent
```

---

## Quick File Navigation

| Task | Key Files |
|------|-----------|
| Public API | `src/abi.zig` |
| Feature module | `src/features/<feature>/mod.zig`, `src/features/<feature>/stub.zig` |
| AI submodule | `src/features/ai/<submodule>/mod.zig`, `src/features/ai/<submodule>/stub.zig` |
| Abbey reasoning | `src/features/ai/abbey/` |
| GPU backends | `src/features/gpu/backends/` |
| CLI commands | `tools/cli/commands/` |
| CLI main | `tools/cli/main.zig`, `tools/cli/mod.zig` |
| CLI TUI | `tools/cli/tui/` |
| Build config | `build.zig`, `build.zig.zon` |
| Tests entry | `src/services/tests/mod.zig` |
| Docs site | `docs/` |
| Cloud integrations | `src/services/cloud/` |
| High availability | `src/services/ha/` |
| Connectors | `src/services/connectors/` |
| Security | `src/services/shared/security/` |
| Examples | `examples/` |

---

## References

- `AGENTS.md` - Baseline rules for AI agents
- `GEMINI.md` - Quick reference for Gemini
- `CONTRIBUTING.md` - Development workflow
- `docs/README.md` - Documentation system
- `SECURITY.md` - Security practices
- `DEPLOYMENT_GUIDE.md` - Production deployment
- `PLAN.md` - Development roadmap
- `ROADMAP.md` - Version history
