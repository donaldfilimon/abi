# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## ⚠️ Before Making Changes

**CRITICAL**: This codebase frequently has work-in-progress changes. Before making any modifications:

1. Run `git status` to see uncommitted work
2. Run `git diff --stat` to understand the scope of existing changes
3. Review existing changes before adding new ones to avoid conflicts or duplicating work
4. If there are many staged/unstaged changes, ask the user about their status before proceeding

## TL;DR for Common Tasks

```bash
# Most common workflow
zig build test --summary all && zig fmt .   # Test + format (always run before commits)

# Single file test (faster iteration)
zig test src/path/to/file.zig --test-filter "specific test"

# Build with specific features
zig build -Denable-ai=true -Dgpu-backend=vulkan
```

## Quick Reference

```bash
# Build and test
zig build                              # Build the project
zig build test --summary all           # Run tests with detailed output
zig fmt .                              # Format code (run after edits)
zig build lint                         # Check formatting (CI uses this)
zig build typecheck                    # Type check without running tests
zig build run -- --help                # CLI help

# Single file testing (use zig test, NOT zig build test)
zig test src/runtime/engine/engine.zig
zig test src/tests/mod.zig --test-filter "pattern"

# Test categories
zig test src/tests/stress/mod.zig               # Stress tests
zig test src/tests/integration/mod.zig          # Integration tests
zig test src/tests/e2e/mod.zig                  # End-to-end tests

# Feature-gated builds
zig build -Denable-ai=true -Denable-gpu=false -Denable-database=true
zig build -Dgpu-backend=cuda,vulkan         # GPU backends (comma-separated)

# Runtime feature flags (CLI)
zig build run -- --list-features          # List features and their status
zig build run -- --enable-gpu db stats    # Enable feature for this run
zig build run -- --disable-ai llm info    # Disable feature for this run

# Additional build targets
zig build benchmarks                   # Run comprehensive benchmarks
zig build bench-all                    # Run all benchmark suites
zig build gendocs                      # Generate API documentation
zig build docs-site                    # Generate documentation website
zig build wasm                         # Build WASM bindings
zig build check-wasm                   # Check WASM compilation
zig build examples                     # Build all examples
zig build cli-tests                    # Run CLI command smoke tests
zig build full-check                   # Format + tests + CLI smoke + benchmarks
zig build profile                      # Build with performance profiling
zig build check-perf                   # Run performance verification
zig build mobile                       # Build for mobile targets (Android/iOS)

# Run examples
zig build run-hello                    # Run hello example
zig build run-database                 # Run database example
zig build run-agent                    # Run agent example
zig build run-gpu                      # Run GPU example

# Debugging
zig build -Doptimize=Debug             # Debug build with symbols
gdb ./zig-out/bin/abi                  # Debug with GDB
lldb ./zig-out/bin/abi                 # Debug with LLDB (macOS)
```

## Critical Gotchas

| Issue | Solution |
|-------|----------|
| `--test-filter` syntax | Use `zig test file.zig --test-filter "pattern"`, NOT `zig build test --test-filter` |
| File system operations | Use `std.Io.Dir.cwd()` instead of deprecated `std.fs.cwd()` (Zig 0.16) |
| Reserved keywords | Escape with `@"error"` syntax, not bare `error` |
| Feature disabled errors | Rebuild with `-Denable-<feature>=true` |
| GPU backend conflicts | Prefer one primary backend; CUDA+Vulkan may cause issues |
| WASM limitations | `database`, `network`, `gpu` auto-disabled; no `std.Io.Threaded` |
| libc linking | CLI and examples require libc for environment variable access |
| Import paths | Always use `@import("abi")` for public API, not direct file paths |
| Stub/Real module sync | Changes to `mod.zig` must be mirrored in `stub.zig` with identical signatures |
| Format specifiers | Use `{t}` for errors/enums, not `@errorName()`/`@tagName()` |
| ArrayListUnmanaged | Use `.empty` not `.init()` for unmanaged variants |
| Timer API | Use `std.time.Timer.start()` not `std.time.Instant.now()` |
| Sleep API | Use `std.Io.Clock.Duration.sleep()` not `std.time.sleep()` |
| HTTP Server init | Use `&reader.interface` and `&writer.interface` for `std.http.Server.init()` |
| Slow builds | Clear `.zig-cache` or reduce parallelism with `zig build -j 2` |
| Debug builds | Use `-Doptimize=Debug` for debugging, `-Doptimize=ReleaseFast` for performance |
| GPU (CUDA) | Requires NVIDIA drivers + toolkit; use Vulkan or `stdgpu` fallback |
| GPU (Metal) | macOS only; includes Accelerate framework (AMX) and unified memory support |
| WASM getCpuCount | Use `getCpuCount()` only with WASM/freestanding guards; 9+ files affected |
| Streaming API | Use `src/ai/streaming/` for real-time LLM responses; backend selection via config |

## Feature Flags

| Flag | Default | Description |
|------|---------|-------------|
| `-Denable-ai` | true | Full ABI agent system (LLM, vision, agent, training sub-features) |
| `-Denable-gpu` | true | GPU acceleration framework |
| `-Denable-database` | true | Vector database integration (WDBX) |
| `-Denable-network` | true | Distributed compute capabilities |
| `-Denable-web` | true | Web utilities and HTTP support |
| `-Denable-profiling` | true | Performance profiling and metrics |
| `-Denable-explore` | true | Codebase exploration (requires `-Denable-ai`) |
| `-Denable-llm` | true | Local LLM inference (requires `-Denable-ai`) |
| `-Denable-vision` | true | Vision/image processing (requires `-Denable-ai`) |
| `-Denable-mobile` | false | Mobile cross-compilation (Android/iOS) |

### GPU Backends

**New unified syntax (recommended):**
```bash
zig build -Dgpu-backend=vulkan              # Single backend
zig build -Dgpu-backend=cuda,vulkan         # Multiple backends (comma-separated)
zig build -Dgpu-backend=none                # Disable all GPU backends
zig build -Dgpu-backend=auto                # Auto-detect available backends
```

**Available backends:** `none`, `auto`, `cuda`, `vulkan`, `stdgpu`, `metal`, `webgpu`, `opengl`, `opengles`, `webgl2`, `fpga`

**Deprecated (still works with warning):** `-Dgpu-vulkan`, `-Dgpu-cuda`, `-Dgpu-metal`, `-Dgpu-webgpu`, `-Dgpu-opengl`

## Architecture

Flat domain structure with unified configuration. Each domain has `mod.zig` (entry point) and `stub.zig` (feature-gated placeholder).

```
src/
├── abi.zig              # Public API entry point: init(), shutdown(), version()
├── config.zig           # Unified configuration system (single Config struct)
├── config/              # Modular configuration system
│   ├── mod.zig          # Config entry point
│   ├── ai.zig           # AI-specific configuration
│   ├── cloud.zig        # Cloud provider configuration
│   ├── database.zig     # Database configuration
│   ├── gpu.zig          # GPU configuration
│   ├── network.zig      # Network configuration
│   ├── observability.zig # Observability configuration
│   ├── plugin.zig       # Plugin configuration
│   └── web.zig          # Web configuration
├── cpu.zig              # CPU fallback for GPU operations
├── flags.zig            # Feature flags management
├── framework.zig        # Framework orchestration with builder pattern
├── io.zig               # I/O utilities
├── ai/                  # AI module with sub-features
│   ├── mod.zig          # AI public API
│   ├── stub.zig         # Stub when AI disabled
│   ├── abbey/           # Abbey persona subsystem (advanced, memory, neural)
│   ├── agents/          # Agent runtime
│   ├── core/            # Integrated core types and configuration
│   ├── database/        # Database-related AI functionality (convert, export, wdbx)
│   ├── discovery.zig    # AI model/capability discovery
│   ├── documents/       # Document handling and processing
│   ├── embeddings/      # Vector embeddings
│   ├── eval/            # Model evaluation and benchmarking
│   ├── explore/         # Codebase exploration
│   ├── federated/       # Federated learning
│   ├── gpu_agent.zig    # GPU-specific agent implementation
│   ├── llm/             # Local LLM inference (streaming, tokenization)
│   ├── memory/          # Agent memory systems
│   ├── model_registry.zig # Model registry functionality
│   ├── multi_agent/     # Multi-agent coordination
│   ├── orchestration/   # Multi-model routing, ensemble, fallback
│   ├── personas/        # AI persona definitions (abbey, abi, aviva, etc.)
│   ├── prompts/         # Prompt management
│   ├── rag/             # Retrieval-augmented generation
│   ├── streaming/       # Streaming response handling
│   ├── templates/       # Template system
│   ├── tools/           # Agent tools
│   ├── training/        # Training pipelines
│   ├── transformer/     # Transformer architecture
│   └── vision/          # Vision/image processing
├── cloud/               # Cloud function adapters (AWS Lambda, Azure, GCP)
├── connectors/          # API connectors (OpenAI, Ollama, Anthropic, HuggingFace)
├── database/            # Vector database (WDBX with HNSW/IVF-PQ)
├── gpu/                 # GPU acceleration (Vulkan, CUDA, Metal, etc.)
│   ├── kernels/         # Split kernel implementations (elementwise, matrix, etc.)
│   ├── backends/fpga/   # FPGA backend stubs and types
│   └── dsl/codegen/     # Shader codegen with generic comptime template
│       ├── generic.zig  # Comptime CodeGenerator(BackendConfig) template
│       ├── configs/     # Backend-specific config structs (glsl, wgsl, msl, cuda)
│       ├── spirv/       # Split SPIRV generator modules
│       └── *.zig        # Thin wrappers re-exporting generic generators
├── ha/                  # High availability (backup, PITR, replication)
├── network/             # Distributed compute and Raft consensus
├── observability/       # Consolidated metrics, tracing, monitoring
├── registry/            # Feature registry (comptime, runtime, dynamic)
│   ├── mod.zig          # Public API facade with Registry struct
│   ├── types.zig        # Core types (Feature, RegistrationMode, Error)
│   ├── registration.zig # registerComptime, registerRuntimeToggle, registerDynamic
│   └── lifecycle.zig    # initFeature, deinitFeature, enable/disable
├── runtime/             # Always-on infrastructure
│   ├── engine/          # Work-stealing task execution
│   ├── scheduling/      # Futures, cancellation, task groups
│   ├── concurrency/     # Lock-free primitives (see Concurrency Primitives)
│   └── memory/          # Memory pools and allocators
├── shared/              # Consolidated shared components
│   ├── legacy/          # Legacy core utilities
│   ├── security/        # TLS, mTLS, API keys, RBAC
│   ├── utils/           # Sub-modules (config, crypto, json, net, etc.)
│   ├── logging.zig      # Logging
│   ├── platform.zig     # Platform detection
│   ├── plugins.zig      # Plugin registry primitives
│   ├── simd.zig         # SIMD vector operations
│   └── utils.zig        # Unified utilities (time, math, string, crypto, http, json, etc.)
├── tasks.zig            # Centralized task management
├── web/                 # Web/HTTP utilities
└── tests/               # Comprehensive test suite
    ├── mod.zig          # Test entry point
    ├── chaos/           # Chaos testing (fault injection, recovery)
    ├── e2e/             # End-to-end tests
    ├── integration/     # Integration tests
    ├── property/        # Property-based testing
    └── stress/          # Stress tests (concurrency, load)
```

**Import guidance:**
- **Public API**: Always use `@import("abi")` - never import files directly
- **Feature Modules**: Access via `abi.gpu`, `abi.ai`, `abi.database`, etc.
- **Shared Utilities**: Import from `src/shared/utils.zig` for all utils sub-modules, or specific files for targeted imports
- **Internal AI**: Implementation files import from `../../core/mod.zig` for types

**Stub pattern:** Each feature module has a `stub.zig` that provides the same API surface when the feature is disabled. When modifying a module's public API, update both `mod.zig` and `stub.zig` to maintain compatibility. The AI module has extensive sub-feature stubs (`src/ai/*/stub.zig`) for agents, embeddings, llm, vision, training, etc.

**Comptime generics pattern:** Use comptime configuration structs to eliminate code duplication. Example from GPU codegen:

```zig
// configs/wgsl_config.zig - Define config struct with backend-specific values
pub const config = BackendConfig{
    .language = .wgsl,
    .type_names = .{ .f32_ = "f32", .i32_ = "i32", ... },
    .atomics = .{ .add_fn = "atomicAdd", ... },
};

// generic.zig - Generic template instantiated with config
pub fn CodeGenerator(comptime cfg: BackendConfig) type {
    return struct {
        pub const backend_config = cfg;
        // Shared logic uses backend_config.type_names, etc.
    };
}

// Pre-instantiate generators for each backend
pub const WgslGenerator = CodeGenerator(wgsl_config.config);
pub const GlslGenerator = CodeGenerator(glsl_config.config);

// wgsl.zig - Backend-specific file is thin wrapper
const generic = @import("generic.zig");
pub const Generator = generic.WgslGenerator;
```

This pattern reduces each backend from ~1,000+ lines to ~50-100 lines while keeping all shared logic in `generic.zig`.

**Table-driven build system:** `build.zig` uses arrays of `BuildTarget` structs to define examples and benchmarks. When adding new examples or benchmarks, add them to the appropriate array (`example_targets` or `benchmark_targets`) rather than duplicating build code. The `buildTargets()` function handles compilation uniformly.

### Configuration System

Two-level configuration architecture:
- **`src/config.zig`**: Unified `Config` struct with builder pattern for framework initialization
- **`src/config/`**: Modular per-feature configs (ai.zig, gpu.zig, database.zig, etc.)

Single `Config` struct with optional feature configs and builder pattern:

```zig
pub const Config = struct {
    gpu: ?GpuConfig = null,
    ai: ?AiConfig = null,
    database: ?DatabaseConfig = null,
    network: ?NetworkConfig = null,
    observability: ?ObservabilityConfig = null,
    web: ?WebConfig = null,
    plugins: PluginConfig = .{},
};

// Builder pattern usage
const config = abi.Config.init()
    .withAI(true)
    .withGPU(true)
    .withDatabase(true);

var framework = try abi.Framework.init(allocator, config);
defer framework.deinit();
```

## Common Workflows

### Adding a new public API function
1. Add to the real module (`src/<feature>/mod.zig`)
2. Mirror the same signature in `src/<feature>/stub.zig`
3. If the function needs types from core, import from `../../core/mod.zig`
4. Run `zig build test` to verify both paths compile

### Adding a new example
1. Add entry to `example_targets` array in `build.zig`
2. Create the example file in `examples/`
3. Run `zig build examples` to verify compilation

### Writing tests
- Use `error.SkipZigTest` for hardware-gated tests (GPU, network):
  ```zig
  test "gpu operation" {
      const gpu = initGpu() catch return error.SkipZigTest;
      defer gpu.deinit();
      // ... test code
  }
  ```
- Unit tests live in library files and `src/tests/mod.zig`
- Run filtered tests with `zig test file.zig --test-filter "pattern"`
- For tests requiring I/O, initialize `std.Io.Threaded` in test setup (see Zig 0.16 Patterns)

### Multi-Model Training
The training module supports LLM, Vision (ViT), and Multimodal (CLIP) models:
```bash
zig build run -- train llm --epochs 10 --batch 32     # LLM training
zig build run -- train vision --model vit            # ViT image classification
zig build run -- train clip                          # CLIP multimodal training
```

Key features: gradient clipping, mixed precision (FP16/BF16), contrastive learning, checkpoint resume.

## Zig 0.16 Patterns

> See [docs/migration/zig-0.16-migration.md](docs/migration/zig-0.16-migration.md) for comprehensive examples.

### I/O Backend Initialization (CRITICAL)

Zig 0.16 requires explicit I/O backend initialization for file and network operations. Requires an allocator; see `examples/` for full setup with `GeneralPurposeAllocator`.

```zig
// Initialize once, use for all file/network operations
var io_backend = std.Io.Threaded.init(allocator, .{
    .environ = std.process.Environ.empty,  // .empty for library, .init() for CLI
});
defer io_backend.deinit();
const io = io_backend.io();

// File read
const content = try std.Io.Dir.cwd().readFileAlloc(io, path, allocator, .limited(10 * 1024 * 1024));
defer allocator.free(content);

// File write
var file = try std.Io.Dir.cwd().createFile(io, path, .{ .truncate = true });
defer file.close(io);
try file.writer(io).writeAll(content);

// Directory operations
std.Io.Dir.cwd().makePath(io, "path/to/dir") catch |err| switch (err) {
    error.PathAlreadyExists => {},
    else => return err,
};

// Directory iteration
var dir = try std.Io.Dir.cwd().openDir(io, "path", .{ .iterate = true });
defer dir.close(io);
```

### Other Zig 0.16 Changes

```zig
// Error/enum formatting: use {t} instead of @errorName()/@tagName()
std.debug.print("Error: {t}, State: {t}", .{err, state});

// ArrayListUnmanaged: use .empty not .init()
var list = std.ArrayListUnmanaged(u8).empty;

// Timing: use Timer.start() not Instant.now()
var timer = std.time.Timer.start() catch return error.TimerFailed;
const elapsed_ns = timer.read();

// HTTP server: use .interface for reader/writer (from stream.reader/writer)
var server: std.http.Server = .init(
    &connection_reader.interface,  // connection_reader = stream.reader(io, &recv_buffer)
    &connection_writer.interface,  // connection_writer = stream.writer(io, &send_buffer)
);
```

## Concurrency Primitives

The `src/runtime/concurrency/` module provides lock-free data structures for high-performance concurrent code:

| Primitive | File | Description |
|-----------|------|-------------|
| Chase-Lev Deque | `chase_lev.zig` | Work-stealing deque for task scheduling |
| Epoch-Based Reclamation | `epoch.zig` | Safe memory reclamation for lock-free structures |
| Lock-Free Primitives | `lockfree.zig` | Atomic operations and CAS utilities |
| MPMC Queue | `mpmc_queue.zig` | Multi-producer multi-consumer bounded queue |
| Priority Queue | `priority_queue.zig` | Concurrent priority queue |

```zig
// Example: Using the MPMC queue
const concurrency = @import("abi").runtime.concurrency;
var queue = try concurrency.MpmcQueue(u64).init(allocator, 1024);
defer queue.deinit();

try queue.push(42);
if (queue.pop()) |value| {
    // Process value
}
```

## GPU Memory Pool

The GPU module includes an optimized memory pool for LLM workloads:

```zig
const gpu = @import("abi").gpu;
var pool = try gpu.MemoryPool.init(allocator, .{
    .strategy = .best_fit,  // or .first_fit
    .auto_defrag = true,
    .size_classes = &.{ 4096, 16384, 65536 },  // Common LLM buffer sizes
});
defer pool.deinit();

const buffer = try pool.allocate(32768);
defer pool.free(buffer);

// Check fragmentation
const stats = pool.getStats();
if (stats.fragmentation_ratio > 0.3) {
    try pool.defragment();
}
```

**Features:** best-fit allocation, buffer splitting, fragmentation tracking, auto-defragmentation.

## Test Infrastructure

The `src/tests/` directory contains a comprehensive test suite:

```bash
# Run all tests
zig build test --summary all

# Run specific test categories
zig test src/tests/integration/mod.zig          # Integration tests
zig test src/tests/stress/mod.zig               # Stress tests
zig test src/tests/property/mod.zig             # Property-based tests
zig test src/tests/e2e/mod.zig                  # End-to-end tests
zig test src/tests/chaos/mod.zig                # Chaos/fault injection tests

# Run with filter
zig test src/tests/mod.zig --test-filter "database"
```

| Directory | Purpose |
|-----------|---------|
| `chaos/` | Fault injection, recovery testing |
| `e2e/` | Full system end-to-end tests |
| `integration/` | Cross-module integration tests |
| `property/` | Property-based/fuzzing tests |
| `stress/` | High-load concurrency stress tests |

## CLI Commands

| Command | Purpose |
|---------|---------|
| `db` | Database operations (add, query, stats, optimize, backup, restore, serve) |
| `agent` | AI agent interaction (interactive, one-shot, 13 personas) |
| `llm` | LLM inference (chat, generate, serve, info, bench, download, list) |
| `model` | Model management (list, info, download, remove, search, path) |
| `train` | Training pipeline (run, llm, vision, clip, resume, monitor, info) |
| `gpu` | GPU management (backends, devices, summary, default, status) |
| `gpu-dashboard` | Interactive GPU + Agent monitoring TUI |
| `bench` | Benchmarks (all, simd, memory, ai, quick, concurrency) |
| `simd` | SIMD performance demonstration |
| `task` | Task management (add, list, done, stats) |
| `tui` | Interactive TUI command launcher with themes |
| `explore` | Codebase search (quick/medium/thorough/deep levels) |
| `embed` | Generate embeddings (openai, ollama, mistral, cohere) |
| `config` | Configuration management (init, show, validate) |
| `network` | Network registry (list, register, status) |
| `multi-agent` | Multi-agent coordinator workflows |
| `plugins` | Plugin management (list, enable, disable, info) |
| `profile` | User profile and settings |
| `discord` | Discord bot operations |
| `convert` | Dataset conversion (tokenbin, text, jsonl, wdbx) |
| `completions` | Shell completions (bash, zsh, fish, powershell) |
| `system-info` | Framework and feature status |
| `toolchain` | Zig toolchain management (temporarily disabled for Zig 0.16 migration) |

### Model Management

```bash
zig build run -- model list                          # List cached models
zig build run -- model info llama-7b                 # Show model details
zig build run -- model download TheBloke/Model:Q4_K_M  # Download from HuggingFace
zig build run -- model remove llama-7b               # Remove cached model
zig build run -- model search llama                  # Search HuggingFace models
zig build run -- model path llama-7b                 # Get local model path
```

Models are cached in platform-aware directories (`~/.abi/models/` on Unix, `%APPDATA%\abi\models\` on Windows). The HuggingFace shorthand format is `TheBloke/Model:QuantType`.

### LLM CLI Examples

```bash
zig build run -- llm chat --model llama-7b           # Interactive chat
zig build run -- llm generate "Once upon" --max 100  # Text generation
zig build run -- llm info --model mistral            # Model information
zig build run -- llm list                            # List available models
```

The LLM feature (`src/ai/llm/`) provides local GGUF model inference with:
- **Tokenization**: BPE and SentencePiece (Viterbi)
- **Quantization**: Q4_0, Q4_1, Q5_0, Q5_1, Q8_0 with roundtrip encoding
- **Transformer Ops**: MatMul, attention, RoPE, RMSNorm, SiLU with SIMD
- **KV Cache**: Standard, sliding window, and paged attention (vLLM-style)
- **GPU Acceleration**: CUDA kernels for softmax, RMSNorm, SiLU with CPU fallback
- **Sampling**: Greedy, top-k, top-p, temperature, tail-free, mirostat (v1/v2)
- **Export**: GGUF writer for trained model export

### Streaming Inference API

The streaming module (`src/ai/streaming/`) provides real-time token streaming:

```bash
# Start streaming server with a local GGUF model
zig build run -- llm serve -m ./models/llama-7b.gguf --preload

# With authentication and custom address
zig build run -- llm serve -m ./model.gguf -a 0.0.0.0:8000 --auth-token my-secret
```

**Endpoints:**
- `POST /v1/chat/completions` - OpenAI-compatible chat completions (SSE)
- `POST /api/stream` - Custom ABI streaming endpoint (SSE)
- `GET /api/stream/ws` - WebSocket streaming (bidirectional, supports cancellation)
- `POST /admin/reload` - Hot-reload model without restart
- `GET /health` - Health check

**Features:**
- **SSE/WebSocket** support for real-time responses
- **Backend routing**: local GGUF, OpenAI, Ollama, Anthropic
- **Bearer token auth** with configurable validation
- **Heartbeat keep-alive** for long-running connections
- **Model preloading** to reduce first-request latency
- **Circuit breakers**: Per-backend failure isolation with automatic recovery
- **Session caching**: Resume interrupted streams via SSE Last-Event-ID

**Stream Recovery (Circuit Breaker Pattern):**
```zig
const streaming = @import("abi").ai.streaming;

// Initialize recovery with circuit breakers
var recovery = try streaming.StreamRecovery.init(allocator, .{
    .circuit_breaker = .{ .failure_threshold = 5 },
});
defer recovery.deinit();

// Check backend availability before use
if (recovery.isBackendAvailable(.openai)) {
    // Backend circuit is closed, safe to use
}

// Record outcomes to update circuit state
recovery.recordSuccess(.openai);
recovery.recordFailure(.openai);  // Opens circuit after threshold

// Session cache for reconnection
var cache = streaming.SessionCache.init(allocator, .{});
try cache.storeToken("session-id", event_id, "token", .local, prompt_hash);
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ABI_OPENAI_API_KEY` | - | OpenAI API key |
| `ABI_OLLAMA_HOST` | `http://127.0.0.1:11434` | Ollama host |
| `ABI_OLLAMA_MODEL` | `gpt-oss` | Default Ollama model |
| `ABI_HF_API_TOKEN` | - | HuggingFace token |
| `ABI_ANTHROPIC_API_KEY` | - | Anthropic/Claude API key |
| `ABI_MASTER_KEY` | - | 32-byte key for secrets encryption (required in production) |
| `DISCORD_BOT_TOKEN` | - | Discord bot token |

## Security Considerations

| Setting | Default | Production Recommendation |
|---------|---------|---------------------------|
| JWT `allow_none_algorithm` | false | Keep false (logs warning if enabled) |
| Secrets `require_master_key` | false | Set true for production |
| Rate limiting | off | Enable for public APIs |

**Critical for production:**
1. Set `ABI_MASTER_KEY` environment variable (32+ bytes)
2. Enable rate limiting on public endpoints
3. Review `docs/SECURITY_AUDIT.md` for known issues

## Platform Notes

### Windows

| Issue | Solution |
|-------|----------|
| Path separators | Use forward slashes `/` in Zig code; backslashes work in shell commands |
| Binary location | `zig-out\bin\abi.exe` (note `.exe` extension) |
| Cache clearing | `rmdir /s /q .zig-cache` or `Remove-Item -Recurse .zig-cache` (PowerShell) |
| Environment variables | Use `set VAR=value` (cmd) or `$env:VAR="value"` (PowerShell) |
| Line endings | Git handles CRLF/LF; `zig fmt` normalizes to LF |

```powershell
# Windows PowerShell examples
zig build run -- --help
zig build test --summary all
$env:ABI_OPENAI_API_KEY="sk-..."
.\zig-out\bin\abi.exe db stats
```

### macOS/Linux

```bash
# Use LLDB on macOS, GDB on Linux
lldb ./zig-out/bin/abi    # macOS
gdb ./zig-out/bin/abi     # Linux
```

## Debugging

```bash
# Debug build
zig build -Doptimize=Debug

# Run with GDB (Linux)
gdb ./zig-out/bin/abi
(gdb) break src/runtime/engine/engine.zig:150
(gdb) run -- db stats

# Run with LLDB (macOS)
lldb ./zig-out/bin/abi
(lldb) breakpoint set --file engine.zig --line 150
```

**Memory leak detection:**
```zig
var gpa = std.heap.GeneralPurposeAllocator(.{
    .stack_trace_frames = 10,
}){};
defer {
    const check = gpa.deinit();
    if (check == .leak) @panic("Memory leak detected");
}
```

## Connectors

Use external AI providers through the connectors module:

```zig
// OpenAI
const openai = @import("abi").connectors.openai;
var client = try openai.Client.init(allocator, .{});
const response = try client.chat("Hello", .{});

// Ollama (local)
const ollama = @import("abi").connectors.ollama;
var client = try ollama.Client.init(allocator, .{ .host = "http://127.0.0.1:11434" });
```

Connectors require corresponding environment variables (see Environment Variables section).

## Language Bindings

| Language | Location | Build | Notes |
|----------|----------|-------|-------|
| **Rust** | `bindings/rust/` | `cargo build` | Safe wrappers; SIMD works without native lib |
| **Go** | `bindings/go/` | `go build ./...` | cgo bindings; requires native lib |
| **Python** | `bindings/python/` | `pip install -e .` | Streaming FFI, training API, observability |
| **WASM** | `bindings/wasm/` | `zig build wasm && npm run build` | @abi-framework/wasm package |
| **C** | `bindings/c/` | Headers only | FFI integration headers |

**Prerequisites**: All bindings except pure-Rust SIMD require the native library (`zig build`) first.

## VS Code Extension

The `vscode-abi/` directory contains a VS Code extension:
- **Commands**: Build, test, run examples
- **AI Chat**: Sidebar webview for agent interaction
- **GPU Status**: Tree view with device monitoring
- **Tasks**: Custom task provider for common operations
- **Snippets**: 15 Zig snippets for ABI patterns
- **Diagnostics**: Compilation error highlighting

Build: `cd vscode-abi && npm install && npm run compile`

## Docker Deployment

A `docker-compose.yml` is provided for containerized deployments:

```bash
# Standard deployment
docker compose up -d abi

# GPU-enabled deployment (requires NVIDIA Container Toolkit)
docker compose up -d abi-gpu

# With Ollama for local LLM inference
docker compose --profile ollama up -d
```

The Dockerfile uses multi-stage builds with optimized `.dockerignore` for faster builds.

## Adding a New Feature Module

1. Create the module directory under `src/<feature>/`
2. Create `mod.zig` (real implementation) and `stub.zig` (disabled placeholder)
3. Both files must export identical public APIs; stub returns `error.<Feature>Disabled`
4. Add feature flag in `build.zig` (`-Denable-<feature>`)
5. Register in `src/registry/` if it needs runtime toggling
6. Add entry to `src/config.zig` Config struct if it needs configuration
7. Update `src/abi.zig` to conditionally import via `if (build_options.enable_<feature>)`

## Reference

Key documentation (all in `docs/`):
- [PLAN.md](PLAN.md) - Development roadmap and sprint status
- [deployment.md](docs/deployment.md) - Production deployment guide
- [SECURITY_AUDIT.md](docs/SECURITY_AUDIT.md) - Security audit findings and status
- [migration/zig-0.16-migration.md](docs/migration/zig-0.16-migration.md) - Zig 0.16 I/O patterns (critical)
- [troubleshooting.md](docs/troubleshooting.md) - Common issues and solutions
- [gpu.md](docs/gpu.md) - GPU backend details
- [ai.md](docs/ai.md) - AI module and agents guide
- [agents.md](docs/agents.md) - Agent personas and interaction
- [database.md](docs/database.md) - Vector database (WDBX) usage
- [network.md](docs/network.md) - Distributed compute and Raft consensus
- [streaming.md](docs/streaming.md) - SSE/WebSocket streaming API
- [models.md](docs/models.md) - Model download, caching, and hot-reload
- [benchmarking.md](docs/benchmarking.md) - Performance benchmarking guide
- [cli-testing.md](docs/cli-testing.md) - CLI test procedures

## Experimental Feature Flags

During CI or local builds you may need to toggle experimental flags via `-D` options:

```bash
zig build -Denable-telemetry=true        # Enable experimental telemetry
zig build -Denable-scripting=true        # Enable embedded scripting backends
```

These flags are integrated into `build_options` and must have corresponding stub implementations in `stub.zig` to keep API parity.

## Code Style

| Rule | Convention |
|------|------------|
| Indentation | 4 spaces, no tabs |
| Line length | Under 100 characters |
| Types | `PascalCase` |
| Functions/Variables | `camelCase` |
| Imports | Explicit only (no `usingnamespace`) |
| Error handling | `!` return types, specific error enums |
| Cleanup | Prefer `defer`/`errdefer` |
| ArrayList | Prefer `std.ArrayListUnmanaged` with explicit allocator passing |

## Quick File Navigation

| Task | Key Files |
|------|-----------|
| Add new CLI command | `tools/cli/commands/` + register in `tools/cli/main.zig` |
| Add new feature module | `src/<feature>/mod.zig` + `src/<feature>/stub.zig` + `build.zig` |
| Add new connector | `src/connectors/<provider>.zig` |
| Add new GPU backend | `src/gpu/backends/` + `src/gpu/dsl/codegen/configs/` |
| Modify public API | `src/abi.zig` (entry point) |
| Add new example | `examples/` + add to `example_targets` in `build.zig` |
| Add new test category | `src/tests/<category>/mod.zig` + import in `src/tests/mod.zig` |
| Streaming API changes | `src/ai/streaming/` (server, backends, handlers) |
| Model management | `src/ai/llm/model_manager.zig` + `tools/cli/commands/model.zig` |

## Post-Edit Checklist

After making changes, always run:

```bash
zig fmt .                        # Format code
zig build test --summary all     # Run all tests
zig build lint                   # Verify formatting passes CI
```

If modifying a feature module's public API, verify both enabled and disabled builds compile:

```bash
zig build -Denable-<feature>=true   # Real module
zig build -Denable-<feature>=false  # Stub module
```
