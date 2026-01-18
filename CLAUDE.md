# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

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
| Stub API mismatch | When adding to real module, mirror the change in the corresponding `stub.zig` |
| Format specifiers | Use `{t}` for errors/enums, not `@errorName()`/`@tagName()` |

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

**GPU Backends:** `-Dgpu-vulkan` (default), `-Dgpu-cuda`, `-Dgpu-metal`, `-Dgpu-webgpu`, `-Dgpu-opengl`

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
│   ├── llm/             # Local LLM inference
│   └── training/        # Training pipelines
├── connectors/          # API connectors (OpenAI, Ollama, Anthropic, HuggingFace)
├── database/            # Vector database (WDBX with HNSW/IVF-PQ)
├── gpu/                 # GPU acceleration (Vulkan, CUDA, Metal, etc.)
├── ha/                  # High availability (backup, PITR, replication)
├── network/             # Distributed compute and Raft consensus
├── observability/       # Consolidated metrics, tracing, monitoring
├── registry/            # Plugin registry (comptime, runtime, dynamic)
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
│   └── utils.zig        # Base utilities (time, math, string)
├── tasks.zig            # Centralized task management
├── web/                 # Web/HTTP utilities
└── tests/               # Integration test suite
```

**Import guidance:**
- **Public API**: Always use `@import("abi")` - never import files directly
- **Feature Modules**: Access via `abi.gpu`, `abi.ai`, `abi.database`, etc.
- **Shared Utilities**: Import from `src/shared/utils.zig` or specific files
- **Internal AI**: Implementation files import from `../../core/mod.zig` for types

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

## Zig 0.16 Patterns

### File I/O (std.Io.Threaded)
```zig
var io_backend = std.Io.Threaded.init(allocator, .{ .environ = std.process.Environ.empty });
defer io_backend.deinit();
const io = io_backend.io();
const content = try std.Io.Dir.cwd().readFileAlloc(io, path, allocator, .limited(10 * 1024 * 1024));
```

### Error/Enum Formatting
```zig
std.debug.print("Error: {t}", .{err});   // Use {t}, not @errorName()
std.debug.print("State: {t}", .{state}); // Use {t}, not @tagName()
```

### Memory Management
```zig
var list = std.ArrayListUnmanaged(u8).empty;  // Prefer Unmanaged variant
try list.append(allocator, item);
list.deinit(allocator);
```

### High-Precision Timing
```zig
var timer = std.time.Timer.start() catch return error.TimerFailed;
// ... work ...
const elapsed_ns = timer.read();
```

### HTTP Server Initialization
```zig
var connection_reader = stream.reader(io, &recv_buffer);
var connection_writer = stream.writer(io, &send_buffer);
var server: std.http.Server = .init(
    &connection_reader.interface,  // Use .interface for *Io.Reader
    &connection_writer.interface,
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

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ABI_OPENAI_API_KEY` | - | OpenAI API key |
| `ABI_OLLAMA_HOST` | `http://127.0.0.1:11434` | Ollama host |
| `ABI_OLLAMA_MODEL` | `gpt-oss` | Default Ollama model |
| `ABI_HF_API_TOKEN` | - | HuggingFace token |
| `DISCORD_BOT_TOKEN` | - | Discord bot token |

## Reference

- [README.md](README.md) - Project overview
- [API_REFERENCE.md](API_REFERENCE.md) - Public API reference
- [docs/troubleshooting.md](docs/troubleshooting.md) - Common issues and solutions
- [docs/migration/zig-0.16-migration.md](docs/migration/zig-0.16-migration.md) - Zig 0.16 patterns
- [docs/gpu.md](docs/gpu.md) - GPU backend details
- [docs/ai.md](docs/ai.md) - AI module guide
