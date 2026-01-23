# Project Context for Ralph
> **Codebase Status:** Synced with repository as of 2026-01-22.

## ABI Framework Overview

ABI is a modern Zig 0.16 framework designed for modular AI services, vector search, and high-performance tooling.

### Core Highlights

| Feature | Description |
|---------|-------------|
| **AI Runtime** | LLM inference (Llama-CPP parity), agent runtime, training pipelines |
| **Vector Database** | WDBX with HNSW/IVF-PQ indexing and hybrid search |
| **Runtime Engine** | Work-stealing scheduler, NUMA awareness, lock-free primitives |
| **GPU Backends** | CUDA, Vulkan, Metal, WebGPU - unified API |
| **Distributed Network** | Node discovery, Raft consensus, load balancing |
| **Observability** | Metrics, tracing, profiling, circuit breakers |
| **CLI** | TUI launcher, runtime feature flags, database ops |

## Documentation

- [Online Docs](https://donaldfilimon.github.io/abi/) - searchable static site
- [Introduction](docs/intro.md) - architecture overview
- [API Reference](API_REFERENCE.md) - public API summary
- [Quickstart](QUICKSTART.md) - getting-started guide
- [Migration Guide](docs/migration/zig-0.16-migration.md) - Zig 0.16 patterns
- [Troubleshooting](docs/troubleshooting.md) - common issues

## Build & Test

```bash
zig build                    # Build the project
zig build test --summary all # Run tests
zig build -Doptimize=ReleaseFast
```

Feature-gated builds:

```bash
zig build -Denable-ai=true \
          -Denable-gpu=false \
          -Denable-database=true
```

Runtime feature flags (CLI):

```bash
zig build run -- --list-features          # List features and status
zig build run -- --enable-gpu db stats    # Enable feature for this run
zig build run -- --disable-ai llm info    # Disable feature for this run
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

**GPU Backends**: `-Dgpu-vulkan` (default), `-Dgpu-cuda`, `-Dgpu-metal`, `-Dgpu-webgpu`, `-Dgpu-opengl`

## Quick Example

```zig
const std = @import("std");
const abi = @import("abi");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var framework = try abi.init(allocator, .{});
    defer abi.shutdown(&framework);

    std.debug.print("ABI version: {s}\n", .{abi.version()});
}
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

## CLI Commands

| Command | Description |
|---------|-------------|
| `--help` | Show help |
| `--list-features` | List available features and status |
| `--enable-<feature>` | Enable a feature at runtime |
| `--disable-<feature>` | Disable a feature at runtime |
| `tui` | Interactive launcher |
| `db stats` | Database statistics |
| `gpu backends` | List GPU backends |
| `agent` | AI agent mode |

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
│   ├── gpu/             # GPU acceleration (unified API)
│   ├── ai/              # AI module (LLM, embeddings, agents, training)
│   ├── database/        # Vector database (WDBX)
│   ├── network/         # Distributed compute
│   ├── web/             # Web utilities and HTTP
│   ├── observability/   # Metrics, tracing, profiling
│   ├── tasks/           # Task management system
│   ├── shared/          # Logging, security, platform utilities
│   ├── core/            # I/O, diagnostics, collections
│   └── features/        # Legacy features (connectors, HA, monitoring)
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

See [CONTRIBUTING.md](CONTRIBUTING.md) for workflow and [CLAUDE.md](CLAUDE.md) for coding guidelines.

## License

[LICENSE](LICENSE)

## Roadmap

The roadmap is split into phases:

### Completed Phases
1. **GPU Modular Refactor** - Moved to `src/gpu/`
2. **Documentation Infrastructure** - API generator, diagrams
3. **Benchmark Framework** - Runner, competitive benches
4. **High Availability Infrastructure** - Failover, PITR
5. **Ecosystem Packaging** - Docker, Zig registry
6. **Runtime Consolidation** - Migrated and removed legacy compute/
   - Plugin registry system (`src/registry/`)
   - CLI runtime flags (`--list-features`, `--enable-*`, `--disable-*`)
   - Task engine, scheduling, concurrency, memory modules
   - Removed deprecated `src/compute/` re-export layer (commit 64334a1)

### In Progress
- **Phase 2: Observability Consolidation** - Unify monitoring implementations
- **Phase 3: AI Module Migration** - Move features/ai/ to ai/

All open items tracked in [ROADMAP.md](ROADMAP.md).

## Migration Status

| Module | Location | Status |
|--------|----------|--------|
| GPU | `src/gpu/` | ✅ Fully migrated |
| Database | `src/database/` | ✅ Fully migrated |
| Network | `src/network/` | ✅ Fully migrated |
| Web | `src/web/` | ✅ Fully migrated |
| Runtime | `src/runtime/` | ✅ Fully migrated |
| Registry | `src/registry/` | ✅ New (complete) |
| Tasks | `src/tasks/` | ✅ Standalone module |
| AI | `src/ai/` + `src/features/ai/` | 🔄 Partial (wrapper + implementation) |
| Observability | `src/observability/` + `src/features/monitoring/` | 🔄 Needs consolidation |
