# GEMINI.md

This file provides guidance for Google Gemini AI when working with the ABI framework codebase.

## Quick Start

```bash
# Essential commands
zig build                              # Build the project
zig build test --summary all           # Run all tests
zig fmt .                              # Format code after edits
zig build run -- --help                # CLI help

# Run example programs
zig build run-hello                    # Run hello example
zig build run-database                 # Run database example
zig build run-agent                    # Run agent example
```

## Project Overview

**ABI Framework** is a Zig 0.16 framework providing:
- **AI Services**: Agents, LLM inference, training, embeddings
- **GPU Compute**: Unified API across Vulkan, CUDA, Metal, WebGPU
- **Vector Database**: WDBX with HNSW indexing
- **Distributed Computing**: Node discovery, Raft consensus
- **High Availability**: Replication, backup, and failover orchestration

## Architecture

The codebase uses a domain-driven flat structure with unified configuration.

```
src/
├── abi.zig              # Public API: init(), shutdown(), version()
├── config.zig           # Unified configuration system
├── framework.zig        # Framework orchestration handle
├── ai/                  # AI module (core/, implementation/, sub-features/)
├── connectors/          # External AI API connectors (OpenAI, Ollama, etc.)
├── database/            # Vector database (WDBX)
├── gpu/                 # GPU acceleration and hardware backends
├── ha/                  # High availability (Replication, Backup)
├── network/             # Distributed compute and Raft
├── observability/       # Consolidated metrics, tracing, and monitoring
├── registry/            # Plugin registry system
├── runtime/             # Always-on infrastructure (Task engine)
├── shared/              # Consolidated utilities and platform helpers
├── tasks.zig            # Integrated task management system
├── web/                 # Web/HTTP utilities
└── tests/               # Integration test suite

tools/cli/               # CLI with 16 commands
benchmarks/              # Performance benchmarks
examples/                # Example programs
docs/                    # Documentation
```

## Key Patterns

### 1. Feature Gating
All major features use compile-time flags:
```zig
const impl = if (build_options.enable_ai) @import("mod.zig") else @import("stub.zig");
```
Build with flags: `zig build -Denable-ai=true -Denable-gpu=false`

### 2. Module Convention
- `mod.zig` - Module entry point
- `stub.zig` - Disabled feature placeholder (returns `error.FeatureDisabled`)

### 3. Type Naming
- **Types**: PascalCase (`GpuBuffer`, `TaskConfig`)
- **Functions**: camelCase (`createEngine`, `runTask`)
- **Constants**: SCREAMING_SNAKE_CASE (`MAX_TASKS`)

### 4. Memory Management
Always use `std.ArrayListUnmanaged`, `defer`/`errdefer` for cleanup, and explicit allocators.

## Critical Rules

### DO
- Read files before editing them.
- Run `zig fmt .` after code changes.
- Run `zig build test --summary all` to verify changes.
- Use specific error types (never `anyerror`).
- Follow existing patterns in the codebase.
- Use `std.Io.Dir.cwd()` instead of deprecated `std.fs.cwd()`.

### DON'T
- Create new directories for small files; consolidate in the parent domain.
- Break the `mod.zig`/`stub.zig` API parity.
- Break public API stability in `abi.zig`.
- Guess at API signatures - read the source.

## Feature Flags

| Flag | Default | Description |
|------|---------|-------------|
| `-Denable-ai` | true | AI agents, LLM, training |
| `-Denable-gpu` | true | GPU acceleration |
| `-Denable-database` | true | Vector database (WDBX) |
| `-Denable-network` | true | Distributed compute |
| `-Denable-web` | true | HTTP client/server |
| `-Denable-profiling` | true | Metrics and profiling |

## GPU Backends

| Backend | Flag | Platform |
|---------|------|----------|
| Vulkan | `-Dgpu-vulkan` | Cross-platform (default) |
| CUDA | `-Dgpu-cuda` | NVIDIA |
| Metal | `-Dgpu-metal` | Apple |
| WebGPU | `-Dgpu-webgpu` | Web/Native |
| stdgpu | `-Dgpu-stdgpu` | CPU fallback |

## Zig 0.16 Specifics

### File I/O
```zig
var io_backend = std.Io.Threaded.init(allocator, .{ .environ = std.process.Environ.empty });
defer io_backend.deinit();
const io = io_backend.io();
const content = try std.Io.Dir.cwd().readFileAlloc(io, path, allocator, .limited(10 * 1024 * 1024));
```

### Formatting
Use the `{t}` format specifier for errors and enums:
```zig
std.debug.print("Status: {t}", .{status});
```

### Memory
Use unmanaged containers for explicit allocator control:
```zig
var list = std.ArrayListUnmanaged(u8).empty;
try list.append(allocator, item);
```

## CLI Commands

| Command | Purpose |
|---------|---------|
| `db` | Database operations |
| `agent` | AI agent interaction |
| `llm` | LLM inference |
| `train` | Model training |
| `gpu` | GPU management |
| `task` | Task management |
| `tui` | Interactive UI launcher |
| `system-info` | System and feature status |

## File Organization

| Domain | Location | Purpose |
|--------|----------|---------|
| **Public API** | `src/abi.zig` | Main library entry point |
| **AI** | `src/ai/` | API in `mod.zig`, impl in `implementation/` |
| **GPU** | `src/gpu/` | Unified API and backends |
| **Shared** | `src/shared/` | Consolidated utils, logging, platform |
| **Tasks** | `src/tasks.zig` | Centralized task management |
| **HA** | `src/ha/` | High availability orchestration |

## Debugging

**Debug builds:** `zig build -Doptimize=Debug` (default).

**Memory leak detection:**
```zig
var gpa = std.heap.GeneralPurposeAllocator(.{ .stack_trace_frames = 10 }){};
defer {
    const check = gpa.deinit();
    if (check == .leak) @panic("Memory leak detected");
}
```

## Need Help?
- Read `CLAUDE.md` for detailed engineering guidelines.
- Run `zig build run -- --help` for CLI command details.
- Check `docs/` for feature-specific documentation.