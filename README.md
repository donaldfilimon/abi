<div align="center">

# ABI Framework

### **Next-Generation AI Infrastructure for Zig**

*High-performance AI runtime, vector databases, and GPU acceleration — all in pure Zig*

<br/>

[![Zig 0.16](https://img.shields.io/badge/Zig-0.16-F7A41D?style=for-the-badge&logo=zig&logoColor=white)](https://ziglang.org/)
[![Production Ready](https://img.shields.io/badge/Status-Production_Ready-00C853?style=for-the-badge)](https://github.com/donaldfilimon/abi)
[![License](https://img.shields.io/github/license/donaldfilimon/abi?style=for-the-badge&color=blue)](LICENSE)

<br/>

[![Build](https://img.shields.io/badge/build-passing-00C853?logo=github-actions&logoColor=white)](https://github.com/donaldfilimon/abi/actions)
[![Tests](https://img.shields.io/badge/tests-passing-00C853?logo=checkmarx&logoColor=white)](https://github.com/donaldfilimon/abi)
[![Coverage](https://img.shields.io/badge/coverage-85%25-F9A825?logo=codecov&logoColor=white)](https://github.com/donaldfilimon/abi)
[![Docs](https://img.shields.io/badge/docs-latest-2196F3?logo=gitbook&logoColor=white)](https://donaldfilimon.github.io/abi/)

<br/>

[**Get Started**](#-quick-start) · [**Documentation**](https://donaldfilimon.github.io/abi/) · [**Examples**](#-examples) · [**Benchmarks**](#-performance) · [**Contributing**](CONTRIBUTING.md)

<br/>

---

</div>

## Why ABI?

ABI brings **production-grade AI infrastructure** to Zig with zero compromises:

| | Feature | What You Get |
|:--:|---------|-------------|
| **AI** | LLM Runtime | Local inference with Llama-CPP parity, streaming, quantization (Q4/Q5/Q8) |
| **Agents** | Intelligent Agents | Multi-persona agents with memory, tool use, and reinforcement learning |
| **GPU** | Unified Compute | CUDA, Vulkan, Metal, WebGPU, FPGA — one API, automatic fallback |
| **DB** | Vector Search | WDBX with HNSW/IVF-PQ indexing, sub-millisecond similarity search |
| **Perf** | Zero-Cost Abstractions | Work-stealing scheduler, lock-free primitives, NUMA-aware allocation |

<br/>

## Feature Matrix

<table>
<tr>
<td width="50%">

### Core Capabilities

| Feature | Status |
|---------|:------:|
| LLM Inference (GGUF) | ✅ |
| Agent Runtime | ✅ |
| Training Pipelines | ✅ |
| Vector Embeddings | ✅ |
| WDBX Vector Database | ✅ |
| Work-Stealing Scheduler | ✅ |
| Lock-Free Primitives | ✅ |

</td>
<td width="50%">

### GPU & Acceleration

| Backend | Status |
|---------|:------:|
| CUDA | ✅ |
| Vulkan | ✅ |
| Metal (macOS) | ✅ |
| WebGPU | ✅ |
| OpenGL/ES | ✅ |
| FPGA (Alveo/Agilex) | ✅ |
| SIMD Fallback | ✅ |

</td>
</tr>
</table>

<br/>

## Quick Start

### Installation

```bash
# Clone and build
git clone https://github.com/donaldfilimon/abi.git
cd abi
zig build

# Verify installation
zig build run -- --version

# Run tests
zig build test --summary all
```

### Hello, ABI

```zig
const std = @import("std");
const abi = @import("abi");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Initialize with builder pattern
    const config = abi.Config.init()
        .withAI(true)
        .withGPU(true)
        .withDatabase(true);

    var framework = try abi.Framework.init(allocator, config);
    defer framework.deinit();

    std.debug.print("ABI v{s} ready!\n", .{abi.version()});
}
```

<br/>

## Examples

<details open>
<summary><b>AI Agent Chat</b></summary>

```zig
const abi = @import("abi");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Create an intelligent agent
    var agent = try abi.ai.Agent.init(allocator, .{
        .name = "assistant",
        .temperature = 0.7,
        .enable_history = true,
    });
    defer agent.deinit();

    // Natural conversation
    const response = try agent.chat("Explain Zig's comptime in one sentence.", allocator);
    defer allocator.free(response);

    std.debug.print("Agent: {s}\n", .{response});
}
```

</details>

<details>
<summary><b>Vector Database Search</b></summary>

```zig
const abi = @import("abi");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Create vector database with 384-dimensional embeddings
    var db = try abi.wdbx.createDatabase(allocator, .{ .dimension = 384 });
    defer db.deinit();

    // Insert vectors
    try db.insertVector(1, &embedding1);
    try db.insertVector(2, &embedding2);

    // Lightning-fast similarity search
    const results = try db.searchVectors(&query_embedding, 10);
    defer allocator.free(results);

    for (results) |result| {
        std.debug.print("ID: {d}, Score: {d:.4}\n", .{ result.id, result.score });
    }
}
```

</details>

<details>
<summary><b>GPU-Accelerated Compute</b></summary>

```zig
const abi = @import("abi");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Auto-detect best GPU backend
    var gpu = try abi.Gpu.init(allocator, .{
        .enable_profiling = true,
        .memory_mode = .automatic,
    });
    defer gpu.deinit();

    // Create buffers
    const a = try gpu.createBufferFromSlice(f32, &data_a, .{});
    defer gpu.destroyBuffer(a);
    const b = try gpu.createBufferFromSlice(f32, &data_b, .{});
    defer gpu.destroyBuffer(b);
    const result = try gpu.createBuffer(size, .{});
    defer gpu.destroyBuffer(result);

    // Execute with automatic GPU -> SIMD -> scalar fallback
    _ = try gpu.vectorAdd(a, b, result);
}
```

</details>

<details>
<summary><b>Training Pipeline</b></summary>

```zig
const abi = @import("abi");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const config = abi.ai.TrainingConfig{
        .epochs = 10,
        .batch_size = 32,
        .learning_rate = 0.001,
        .optimizer = .adamw,
    };

    var result = try abi.ai.trainWithResult(allocator, config);
    defer result.deinit();

    std.debug.print("Final loss: {d:.6}\n", .{result.report.final_loss});
}
```

</details>

<br/>

## CLI Reference

```bash
# General
zig build run -- --help              # Show all commands
zig build run -- --version           # Version info
zig build run -- system-info         # System status
zig build run -- tui                 # Interactive TUI

# AI & Agents
zig build run -- agent               # Interactive agent
zig build run -- agent --persona coder
zig build run -- llm chat --model llama-7b

# Database
zig build run -- db stats
zig build run -- db add --id 1 --embed "text"
zig build run -- db backup --path backup.db

# GPU
zig build run -- gpu backends        # List backends
zig build run -- gpu devices         # List devices
zig build run -- gpu summary         # GPU summary

# Training
zig build run -- train run --epochs 10
zig build run -- train resume ./checkpoint.ckpt

# Feature Flags (runtime)
zig build run -- --list-features
zig build run -- --enable-gpu db stats
```

<br/>

## Performance

Benchmarks from `zig build benchmarks` on a typical development workstation:

| Benchmark | Performance |
|-----------|------------:|
| Framework Init | **175 ops/s** |
| Config Loading | **66.5M ops/s** |
| Logging | **332K ops/s** |
| Memory Alloc (1KB) | **465K ops/s** |
| SIMD Dot Product | **84.9M ops/s** |
| SIMD Vector Add | **84.7M ops/s** |
| Task Dispatch | **93K ops/s** |
| Vector Insert | **68K ops/s** |
| Vector Search | **57K ops/s** |
| JSON Parse | **83K ops/s** |

<br/>

## Build Options

### Feature Flags

```bash
# Enable/disable features at compile time
zig build -Denable-ai=true       # AI features (default: true)
zig build -Denable-gpu=true      # GPU acceleration (default: true)
zig build -Denable-database=true # Vector database (default: true)
zig build -Denable-network=true  # Distributed compute (default: true)
zig build -Denable-web=true      # Web utilities (default: true)
zig build -Denable-llm=true      # Local LLM inference (default: true)
```

### GPU Backend Selection

```bash
# Single backend
zig build -Dgpu-backend=vulkan
zig build -Dgpu-backend=cuda
zig build -Dgpu-backend=metal

# Multiple backends
zig build -Dgpu-backend=cuda,vulkan

# Auto-detect
zig build -Dgpu-backend=auto
```

### Build Targets

```bash
zig build                        # Default build
zig build -Doptimize=ReleaseFast # Optimized release
zig build -Doptimize=Debug       # Debug symbols
zig build test --summary all     # Run all tests
zig build benchmarks             # Run benchmarks
zig build wasm                   # WebAssembly build
zig build gendocs                # Generate docs
```

<br/>

## Architecture

```
abi/
├── src/
│   ├── abi.zig              # Public API entry
│   ├── config.zig           # Unified configuration
│   ├── framework.zig        # Framework orchestration
│   │
│   ├── ai/                  # AI Module
│   │   ├── llm/             # LLM inference (GGUF, streaming, quantization)
│   │   ├── agents/          # Agent runtime with personas
│   │   ├── embeddings/      # Vector embeddings
│   │   ├── training/        # Training pipelines
│   │   └── orchestration/   # Multi-model routing
│   │
│   ├── gpu/                 # GPU Module
│   │   ├── backends/        # CUDA, Vulkan, Metal, WebGPU, FPGA
│   │   ├── mega/            # Cross-backend orchestration
│   │   └── dsl/             # Shader codegen
│   │
│   ├── database/            # WDBX Vector Database
│   ├── runtime/             # Engine, scheduling, memory
│   ├── network/             # Distributed compute, Raft
│   ├── observability/       # Metrics, tracing, profiling
│   └── shared/              # Utilities, security, logging
│
├── tools/cli/               # CLI implementation
├── benchmarks/              # Performance benchmarks
├── examples/                # Example applications
└── docs/                    # Documentation
```

<br/>

## Documentation

| Resource | Description |
|----------|-------------|
| [**Online Docs**](https://donaldfilimon.github.io/abi/) | Searchable documentation site |
| [**API Reference**](API_REFERENCE.md) | Complete public API |
| [**Quickstart**](QUICKSTART.md) | Getting started guide |
| [**AI Guide**](docs/ai.md) | LLM, agents, training |
| [**GPU Guide**](docs/gpu.md) | Multi-backend GPU |
| [**Database Guide**](docs/database.md) | WDBX vector database |
| [**Network Guide**](docs/network.md) | Distributed compute |
| [**Troubleshooting**](docs/troubleshooting.md) | Common issues |

<br/>

## Environment Variables

| Variable | Description |
|----------|-------------|
| `ABI_OPENAI_API_KEY` | OpenAI API key |
| `ABI_ANTHROPIC_API_KEY` | Anthropic/Claude API key |
| `ABI_OLLAMA_HOST` | Ollama host (default: `http://127.0.0.1:11434`) |
| `ABI_OLLAMA_MODEL` | Default Ollama model |
| `ABI_HF_API_TOKEN` | HuggingFace token |
| `DISCORD_BOT_TOKEN` | Discord bot token |

<br/>

## Project Status

| Milestone | Status |
|-----------|:------:|
| Zig 0.16 Migration | ✅ Complete |
| Llama-CPP Parity | ✅ Complete |
| Plugin Registry | ✅ Complete |
| Runtime Consolidation | ✅ Complete |
| FPGA Acceleration | ✅ Complete |
| Mega GPU Orchestration | ✅ Complete |

<br/>

## Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines and [CLAUDE.md](CLAUDE.md) for coding standards.

```bash
# Development workflow
git clone https://github.com/donaldfilimon/abi.git
cd abi
zig build test --summary all  # Run tests
zig fmt .                     # Format code
zig build lint                # Check formatting
```

<br/>

---

<div align="center">

**[Documentation](https://donaldfilimon.github.io/abi/)** · **[Issues](https://github.com/donaldfilimon/abi/issues)** · **[Discussions](https://github.com/donaldfilimon/abi/discussions)**

<br/>

Built with [Zig](https://ziglang.org/) · Licensed under [MIT](LICENSE)

<br/>

<sub>Made with precision and zero garbage collection</sub>

</div>
