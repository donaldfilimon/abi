# ABI Framework

Modern Zig 0.16 framework for modular AI services, vector search, and high-performance systems tooling.

## Highlights

- **AI Runtime** - LLM inference (Llama-CPP parity), agent runtime, training pipelines ([docs/ai.md](docs/ai.md))
- **Vector Database** - WDBX with HNSW/IVF-PQ indexing, hybrid search, diagnostics ([docs/database.md](docs/database.md))
- **Compute Engine** - Work-stealing scheduler, NUMA-aware, lock-free primitives
- **GPU Backends** - CUDA, Vulkan, Metal, WebGPU with unified API, graceful degradation ([docs/gpu.md](docs/gpu.md))
- **Distributed Network** - Node discovery, Raft consensus, load balancing ([docs/network.md](docs/network.md))
- **Observability** - Metrics, tracing, profiling, circuit breakers ([docs/monitoring.md](docs/monitoring.md))
- **Interactive CLI** - TUI launcher, training commands, database operations

## What's New (2026.01)

- **Multi-GPU Device Enumeration** - Enumerate all GPUs across all backends with `device.enumerateAllDevices()`
- **Backend Auto-Detection** - Feature-based backend selection with configurable fallback chains
- **Unified Execution Coordinator** - Automatic GPU→SIMD→scalar fallback via `ExecutionCoordinator`
- **std.gpu Integration** - Zig 0.16 std.gpu compatibility layer with CPU fallback
- **Plugin Registry** - Three-mode feature registration (comptime, runtime-toggle, dynamic)
- **Runtime Consolidation** - Unified `src/runtime/` with engine, scheduling, concurrency, memory
- **CLI Runtime Flags** - `--list-features`, `--enable-*`, `--disable-*` for runtime feature control
- **GPU Diagnostics** - Comprehensive GPU state debugging with `DiagnosticsInfo.collect()`
- **GPU Error Context** - Structured error reporting with backend, operation, and timing info
- **Graceful Degradation** - Automatic CPU fallback when GPU unavailable via `FailoverManager`
- **SIMD CPU Fallback** - AVX/SSE/NEON accelerated operations in `stdgpu` backend
- **Database Diagnostics** - Memory stats, health checks, configuration debugging
- **AI Error Context** - Structured error context for agent operations with retry tracking
- **O(1) Kernel Cache** - Doubly-linked list LRU for constant-time cache operations

## Documentation

- **[Online Docs](https://donaldfilimon.github.io/abi/)** - Searchable static site
- [Introduction](docs/intro.md) - Architecture overview
- [API Reference](API_REFERENCE.md) - Public API summary
- [Quickstart](QUICKSTART.md) - Getting started guide
- [Migration Guide](docs/migration/zig-0.16-migration.md) - Zig 0.16 patterns
- [Troubleshooting](docs/troubleshooting.md) - Common issues

## Requirements

- Zig 0.16.x (minimum 0.16.0)

## Build

```bash
zig build                    # Build the project
zig build test --summary all # Run tests with output
zig build -Doptimize=ReleaseFast

# Feature-gated builds
zig build -Denable-ai=true -Denable-gpu=false -Denable-database=true
```

## Feature Flags

| Flag | Default | Description |
|------|---------|-------------|
| `-Denable-ai` | true | AI features and connectors |
| `-Denable-llm` | true | Local LLM inference |
| `-Denable-gpu` | true | GPU acceleration |
| `-Denable-web` | true | Web utilities and HTTP |
| `-Denable-database` | true | Vector database (WDBX) |
| `-Denable-network` | true | Distributed compute |
| `-Denable-profiling` | true | Profiling and metrics |

**GPU Backends:** `-Dgpu-vulkan` (default), `-Dgpu-cuda`, `-Dgpu-metal`, `-Dgpu-webgpu`, `-Dgpu-opengl`

## Quick Example

```zig
const std = @import("std");
const abi = @import("abi");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Using the unified Config with builder pattern
    const config = abi.Config.init()
        .withAI(true)
        .withGPU(true)
        .withDatabase(true);

    var framework = try abi.Framework.init(allocator, config);
    defer framework.deinit();

    std.debug.print("ABI version: {s}\n", .{abi.version()});

    // Access feature modules through the framework
    if (framework.ai()) |ai| {
        // Use AI features
        _ = ai;
    }
}
```

**Backward-compatible initialization** (re-exports maintain API compatibility):

```zig
var framework = try abi.init(allocator, .{});
defer abi.shutdown(&framework);
```

## Training Example

```zig
const config = abi.ai.TrainingConfig{
    .epochs = 10,
    .batch_size = 32,
    .sample_count = 1024,
    .model_size = 512,
    .learning_rate = 0.001,
    .optimizer = .adamw,
};

var result = try abi.ai.trainWithResult(allocator, config);
defer result.deinit();

std.debug.print("Final loss: {d:.6}\n", .{result.report.final_loss});
```

**CLI:**
```bash
zig build run -- train run --epochs 10 --batch-size 32
zig build run -- train resume ./checkpoint.ckpt
zig build run -- llm chat model.gguf
```

## CLI Commands

```bash
zig build run -- --help            # Show help
zig build run -- tui               # Interactive launcher
zig build run -- db stats          # Database statistics
zig build run -- gpu backends      # List GPU backends
zig build run -- agent             # AI agent mode

# Runtime feature flags (new)
zig build run -- --list-features              # List available features and status
zig build run -- --enable-gpu db stats        # Enable feature for this run
zig build run -- --disable-ai llm info        # Disable feature for this run
```

## Architecture

```
abi/
├── src/
│   ├── abi.zig          # Public API entry point
│   ├── config.zig       # Unified configuration system
│   ├── framework.zig    # Framework orchestration
│   ├── registry/        # Plugin registry system (comptime, runtime, dynamic)
│   ├── runtime/         # Always-on infrastructure
│   │   ├── engine/      # Work-stealing task execution
│   │   ├── scheduling/  # Futures, cancellation, task groups
│   │   ├── concurrency/ # Lock-free primitives
│   │   └── memory/      # Memory pools and allocators
│   ├── gpu/             # GPU backends and unified API
│   ├── ai/              # AI module with sub-features
│   │   ├── llm/         # Local LLM inference
│   │   ├── embeddings/  # Vector embeddings
│   │   ├── agents/      # AI agent runtime
│   │   └── training/    # Training pipelines
│   ├── database/        # Vector database (WDBX)
│   ├── network/         # Distributed compute and Raft
│   ├── observability/   # Metrics, tracing, profiling
│   ├── web/             # Web/HTTP utilities
│   └── shared/          # Shared utilities and platform helpers
├── tools/cli/           # CLI implementation
├── benchmarks/          # Performance benchmarks
└── docs/                # Documentation
```

## Testing

```bash
zig build test --summary all                    # All tests
zig test src/runtime/engine/engine.zig          # Single file
zig test src/tests/mod.zig --test-filter "pat"  # Filter tests
zig build benchmarks                            # Run benchmarks
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ABI_OPENAI_API_KEY` | - | OpenAI API key |
| `ABI_OLLAMA_HOST` | `http://127.0.0.1:11434` | Ollama host |
| `ABI_OLLAMA_MODEL` | `gpt-oss` | Default Ollama model |
| `ABI_HF_API_TOKEN` | - | HuggingFace token |
| `DISCORD_BOT_TOKEN` | - | Discord bot token |

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development workflow and [CLAUDE.md](CLAUDE.md) for coding guidelines.

## Status

- **Zig 0.16 Migration**: Complete
- **Llama-CPP Parity**: Complete (see [TODO.md](TODO.md))
- **Feature Stubs**: All verified and tested
- **Refactoring (Phases 1-6)**: Complete
  - Plugin registry system (`src/registry/`)
  - Runtime consolidation (`src/runtime/`)
  - CLI runtime flags (`--list-features`, `--enable-*`, `--disable-*`)

See [ROADMAP.md](ROADMAP.md) for upcoming milestones and [docs/plans/2026-01-17-src-refactoring.md](docs/plans/2026-01-17-src-refactoring.md) for refactoring details.

## License

See [LICENSE](LICENSE) for details.
