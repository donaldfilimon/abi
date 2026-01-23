# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Working with the Codebase

**IMPORTANT**: Before making changes, always check `git status` for uncommitted work. This codebase frequently has work-in-progress changes across multiple files. Review existing changes before adding new ones to avoid conflicts or duplicating work.

## Quick Reference

```bash
# Build and test
zig build                              # Build the project
zig build test --summary all           # Run tests with detailed output
zig fmt .                              # Format code (run after edits)
zig build run -- --help                # CLI help

# Single file testing (use zig test, NOT zig build test)
zig test src/runtime/engine/engine.zig
zig test src/tests/mod.zig --test-filter "pattern"

# Feature-gated builds
zig build -Denable-ai=true -Denable-gpu=false -Denable-database=true
zig build -Dgpu-backend=cuda,vulkan         # GPU backends (comma-separated)

# Runtime feature flags (CLI)
zig build run -- --list-features          # List features and their status
zig build run -- --enable-gpu db stats    # Enable feature for this run
zig build run -- --disable-ai llm info    # Disable feature for this run

# Additional build targets
zig build benchmarks                   # Run comprehensive benchmarks
zig build gendocs                      # Generate API documentation
zig build wasm                         # Build WASM bindings
zig build check-wasm                   # Check WASM compilation
zig build examples                     # Build all examples

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
| GPU backend conflicts | Enable only one GPU backend at a time |
| WASM limitations | `database`, `network`, `gpu` features auto-disabled for WASM targets |
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
├── framework.zig        # Framework orchestration with builder pattern
├── ai/                  # AI module with sub-features
│   ├── mod.zig          # AI public API
│   ├── stub.zig         # Stub when AI disabled
│   ├── core/            # Integrated core types and configuration
│   ├── implementation/  # Consolidated AI implementation layer
│   ├── agents/          # Agent runtime
│   ├── embeddings/      # Vector embeddings
│   ├── llm/             # Local LLM inference (streaming, tokenization)
│   ├── orchestration/   # Multi-model routing, ensemble, fallback
│   └── training/        # Training pipelines
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
│   ├── concurrency/     # Lock-free primitives
│   └── memory/          # Memory pools and allocators
├── shared/              # Consolidated shared components
│   ├── legacy/          # Legacy core utilities
│   ├── security/        # TLS, mTLS, API keys, RBAC
│   ├── utils/           # Sub-modules (config, crypto, json, net, etc.)
│   ├── logging.zig      # Logging
│   ├── platform.zig     # Platform detection
│   ├── plugins.zig      # Plugin registry primitives
│   ├── simd.zig         # SIMD vector operations
│   ├── utils.zig        # Base utilities (time, math, string)
│   └── utils_combined.zig  # Unified re-export of all utils sub-modules
├── tasks.zig            # Centralized task management
├── web/                 # Web/HTTP utilities
└── tests/               # Integration test suite
```

**Import guidance:**
- **Public API**: Always use `@import("abi")` - never import files directly
- **Feature Modules**: Access via `abi.gpu`, `abi.ai`, `abi.database`, etc.
- **Shared Utilities**: Import from `src/shared/utils_combined.zig` for all utils sub-modules, or specific files for targeted imports
- **Internal AI**: Implementation files import from `../../core/mod.zig` for types

**Stub pattern:** Each feature module has a `stub.zig` that provides the same API surface when the feature is disabled. When modifying a module's public API, update both `mod.zig` and `stub.zig` to maintain compatibility.

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

### Configuration System (`src/config.zig`)

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

## Zig 0.16 Patterns

> See [docs-src/migration/zig-0.16-migration.md](docs-src/migration/zig-0.16-migration.md) for comprehensive examples.

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

## CLI Commands

| Command | Purpose |
|---------|---------|
| `db` | Database operations (add, search, stats, backup, restore) |
| `agent` | AI agent interaction (interactive, one-shot, personas) |
| `llm` | LLM inference (chat, generate, info, download) |
| `train` | Training pipeline (run, info, resume) |
| `gpu` | GPU management (backends, devices, summary) |
| `task` | Task management (add, list, done, stats) |
| `tui` | Interactive TUI launcher |
| `system-info` | Framework and feature status |

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

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ABI_OPENAI_API_KEY` | - | OpenAI API key |
| `ABI_OLLAMA_HOST` | `http://127.0.0.1:11434` | Ollama host |
| `ABI_OLLAMA_MODEL` | `gpt-oss` | Default Ollama model |
| `ABI_HF_API_TOKEN` | - | HuggingFace token |
| `ABI_ANTHROPIC_API_KEY` | - | Anthropic/Claude API key |
| `DISCORD_BOT_TOKEN` | - | Discord bot token |

## Debugging

```bash
# Debug build
zig build -Doptimize=Debug

# Run with GDB
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

## Reference

- [README.md](README.md) - Project overview
- [API_REFERENCE.md](API_REFERENCE.md) - Public API reference
- [docs-src/troubleshooting.md](docs-src/troubleshooting.md) - Common issues and solutions
- [docs-src/migration/zig-0.16-migration.md](docs-src/migration/zig-0.16-migration.md) - Zig 0.16 patterns
- [docs-src/todo.md](docs-src/todo.md) - Development TODO & Zig 0.16 environment init
- [docs-src/agents.md](docs-src/agents.md) - Agents guide with environment setup
- [docs-src/gpu.md](docs-src/gpu.md) - GPU backend details
- [docs-src/ai.md](docs-src/ai.md) - AI module guide
- [docs-src/cloud-deployment.md](docs-src/cloud-deployment.md) - Cloud function deployment
- [docs-src/SECURITY_AUDIT.md](docs-src/SECURITY_AUDIT.md) - Security findings and recommendations

## New Feature Flags
During CI or local builds you may need to toggle new experimental flags via `-D` options. Examples:

```bash
zig build -Denable-telemetry=true        # Enable experimental telemetry
zig build -Denable-scripting=true       # Enable embedded scripting backends
```
These flags are integrated into `build_options` and must have corresponding stub implementations in `stub.zig` to keep API parity.

## Important Gotchas
* **Stub‑Real API parity** – Every change to a `mod.zig` file must be mirrored in its `stub.zig`. The CI parity checker will flag mismatches.
* **GPU backend list** – `-Dgpu-backend` accepts a comma‑separated list; use `-Dgpu-backend=none` to disable all backends.
* **WASM builds** – When targeting WebAssembly the `database`, `network`, and `gpu` features are auto‑disabled; declare them explicitly for deterministic runs.
* **Compile‑time vs Runtime Flags** – `-D...` flags affect compilation. Runtime toggles (`--enable-gpu`) only modify a binary's runtime behaviour.
* **Dependency order** – Changing the import order in `build.zig` can alter the feature graph.

★ *Tip:* After editing run `zig fmt .` and verify with
```bash
zig build test --summary all
```
