# ABI Framework Documentation

Welcome to the **ABI Framework** documentation. ABI is a modern Zig 0.16.x framework for modular AI services, vector search, and high-performance systems tooling.

> **New to ABI?** Start with [Quickstart](../QUICKSTART.md) or [Introduction](intro.md).
> **For AI Agents**: See [AGENTS.md](../AGENTS.md) for comprehensive coding patterns and build commands.
> **For Developers**: See [CLAUDE.md](../CLAUDE.md) for full development guidelines.

---

## Getting Started

```zig
const std = @import("std");
const abi = @import("abi");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var framework = try abi.init(allocator, .{});
    defer abi.shutdown(&framework);

    std.debug.print("ABI v{s} ready\n", .{abi.version()});
}
```

```bash
zig build                           # Build
zig build test --summary all        # Run tests
zig build run -- --help             # CLI help
```

---

## Documentation

### Core Guides

| Guide | Description |
|-------|-------------|
| [Introduction](intro.md) | Architecture overview, philosophy, and layer structure |
| [Framework](framework.md) | Initialization, configuration, lifecycle, and plugins |
| [Troubleshooting](troubleshooting.md) | Common issues, error resolution, and debugging tips |

### Feature Documentation

| Feature | Build Flag | Description |
|---------|------------|-------------|
| [AI & Agents](ai.md) | `-Denable-ai` | LLM connectors, agents, training, federated learning |
| [Compute Engine](compute.md) | (always enabled) | Work-stealing scheduler, task execution, NUMA |
| [Database (WDBX)](database.md) | `-Denable-database` | Vector database, HNSW indexing, hybrid search |
| [GPU Acceleration](gpu.md) | `-Denable-gpu` | Unified API, DSL, CUDA/Vulkan/Metal/WebGPU backends |
| [Monitoring](monitoring.md) | `-Denable-profiling` | Logging, metrics, alerting, tracing, profiling |
| [Network](network.md) | `-Denable-network` | Distributed compute, node discovery, Raft consensus |
| [Explore](explore.md) | `-Denable-explore` | Codebase search, AST parsing, natural language queries |

### Technical References

| Document | Description |
|----------|-------------|
| [Zig 0.16 Migration](migration/zig-0.16-migration.md) | API changes, `std.Io` patterns, compatibility notes |
| [Performance Baseline](PERFORMANCE_BASELINE.md) | Benchmark metrics, optimization guidelines |
| [GPU Backend Details](gpu-backend-improvements.md) | Backend implementations, recovery, metrics |

### Developer Resources

| Document | Description |
|----------|-------------|
| [AGENTS.md](../AGENTS.md) | Quick reference for AI agents (coding patterns, style guidelines, testing) |
| [CLAUDE.md](../CLAUDE.md) | Comprehensive development guide (architecture, patterns, CLI) |
| [CONTRIBUTING.md](../CONTRIBUTING.md) | Contribution workflow and style conventions |
| [TODO.md](../TODO.md) | Pending implementations and Llama‑CPP parity tasks |
| [ROADMAP.md](../ROADMAP.md) | Future milestones and project planning |

---

## Quick Reference

### Build Commands

```bash
# Standard builds
zig build                                    # Debug build
zig build -Doptimize=ReleaseFast             # Production build
zig build test --summary all                 # Run all tests

# Feature-gated builds
zig build -Denable-ai=true -Denable-gpu=false
zig build -Dgpu-cuda=true -Dgpu-vulkan=false

# Additional targets
zig build examples                           # Build examples
zig build wasm                               # Build WASM bindings
zig build benchmarks                         # Run benchmarks
```

### CLI Commands

```bash
zig build run -- db stats                    # Database statistics
zig build run -- gpu devices                 # List GPU devices
zig build run -- explore "fn init"           # Search codebase
zig build run -- llm info model.gguf         # LLM model info
zig build run -- agent --message "Hello"     # Run AI agent
zig build run -- system-info                 # System status
```

### Feature Flags

| Flag | Default | Description |
|------|---------|-------------|
| `-Denable-ai` | `true` | AI agents, connectors, training |
| `-Denable-gpu` | `true` | GPU acceleration |
| `-Denable-database` | `true` | WDBX vector database |
| `-Denable-web` | `true` | HTTP client/server |
| `-Denable-network` | `true` | Distributed compute |
| `-Denable-profiling` | `true` | Metrics and profiling |
| `-Denable-explore` | `true` | Codebase exploration |
| `-Denable-llm` | `true` | Local LLM inference |

---

## Architecture Overview

```
abi/
├── src/
│   ├── abi.zig          # Public API entry point
│   ├── framework/       # Lifecycle, configuration
│   ├── compute/         # Engine, GPU, memory, concurrency
│   ├── features/        # AI, database, monitoring, network
│   └── shared/          # Utilities, logging, security
├── tools/cli/           # CLI implementation
├── benchmarks/          # Performance benchmarks
├── examples/            # Example programs
└── docs/                # Documentation
```

**Layers:**
1. **Public API** (`abi.zig`) - Entry point with `init()`, `shutdown()`, `version()`
2. **Framework** - Feature orchestration and lifecycle management
3. **Compute Engine** - Work-stealing scheduler, GPU integration, memory arenas
4. **Feature Stacks** - AI, database, GPU, network, monitoring modules
5. **Shared Utilities** - Platform abstractions, SIMD, crypto, filesystem

---

## Environment Variables

### AI Connectors

| Variable | Default | Description |
|----------|---------|-------------|
| `ABI_OPENAI_API_KEY` | - | OpenAI API key |
| `ABI_OPENAI_BASE_URL` | `https://api.openai.com/v1` | OpenAI base URL |
| `ABI_OLLAMA_HOST` | `http://127.0.0.1:11434` | Ollama server URL |
| `ABI_HF_API_TOKEN` | - | HuggingFace API token |
| `DISCORD_BOT_TOKEN` | - | Discord bot token |

### Network Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `ABI_LOCAL_SCHEDULER_URL` | `http://127.0.0.1:8081` | Local scheduler URL |

---

## Examples

Run individual examples:

```bash
zig build run-hello          # Hello world
zig build run-database       # WDBX database
zig build run-agent          # AI agent
zig build run-compute        # Compute engine
zig build run-gpu            # GPU operations
zig build run-network        # Distributed compute
```

---

## See Also

- [CLAUDE.md](../CLAUDE.md) - Development guidelines for Claude Code
- [README.md](../README.md) - Project overview
- [CONTRIBUTING.md](../CONTRIBUTING.md) - Contribution guidelines
*See [../TODO.md](../TODO.md) and [../ROADMAP.md](../ROADMAP.md) for the Llama‑CPP parity task list and upcoming milestones.*
