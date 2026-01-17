# ABI Framework

Modern Zig 0.16 framework for modular AI services, vector search, and high-performance systems tooling.

## Highlights

- **AI Runtime** - LLM inference (Llama-CPP parity), agent runtime, training pipelines ([docs/ai.md](docs/ai.md))
- **Vector Database** - WDBX with HNSW/IVF-PQ indexing, hybrid search ([docs/database.md](docs/database.md))
- **Compute Engine** - Work-stealing scheduler, NUMA-aware, lock-free primitives
- **GPU Backends** - CUDA, Vulkan, Metal, WebGPU with unified API ([docs/gpu.md](docs/gpu.md))
- **Distributed Network** - Node discovery, Raft consensus, load balancing ([docs/network.md](docs/network.md))
- **Observability** - Metrics, tracing, profiling, circuit breakers ([docs/monitoring.md](docs/monitoring.md))
- **Interactive CLI** - TUI launcher, training commands, database operations

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

**CLI:**
```bash
zig build run -- train run --epochs 10 --batch-size 32
zig build run -- train resume ./checkpoint.ckpt
zig build run -- llm chat model.gguf
```

## CLI Commands

```bash
zig build run -- --help       # Show help
zig build run -- tui          # Interactive launcher
zig build run -- db stats     # Database statistics
zig build run -- gpu backends # List GPU backends
zig build run -- agent        # AI agent mode
```

## Architecture

```
abi/
├── src/
│   ├── abi.zig          # Public API entry point
│   ├── core/            # I/O, diagnostics, collections
│   ├── compute/         # Runtime, GPU, memory, profiling
│   ├── features/        # AI, database, network, monitoring
│   ├── framework/       # Lifecycle and orchestration
│   └── shared/          # Logging, security, utilities
├── tools/cli/           # CLI implementation
├── benchmarks/          # Performance benchmarks
└── docs/                # Documentation
```

## Testing

```bash
zig build test --summary all                    # All tests
zig test src/compute/runtime/engine.zig         # Single file
zig test src/tests/mod.zig --test-filter "pat"  # Filter tests
zig build benchmarks                            # Run benchmarks
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ABI_OPENAI_API_KEY` | - | OpenAI API key |
| `ABI_OLLAMA_HOST` | `http://127.0.0.1:11434` | Ollama host |
| `ABI_OLLAMA_MODEL` | `llama3.2` | Default Ollama model |
| `ABI_HF_API_TOKEN` | - | HuggingFace token |
| `DISCORD_BOT_TOKEN` | - | Discord bot token |

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development workflow and [CLAUDE.md](CLAUDE.md) for coding guidelines.

## Status

- **Zig 0.16 Migration**: Complete
- **Llama-CPP Parity**: Complete (see [TODO.md](TODO.md))
- **Feature Stubs**: All verified and tested

See [ROADMAP.md](ROADMAP.md) for upcoming milestones.

## License

See [LICENSE](LICENSE) for details.
