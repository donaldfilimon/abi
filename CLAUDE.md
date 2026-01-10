# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ABI is a modern Zig 0.16.x framework for modular AI services, vector search, and high-performance systems tooling. It provides a layered architecture with feature-gated compilation.

## Build Commands

```bash
zig build                    # Build the project
zig build test               # Run all tests
zig build test --summary all # Run tests with detailed output
zig build benchmark          # Run performance benchmarks
zig build run -- --help      # Run CLI with help
zig fmt .                    # Format all code
zig fmt --check .            # Check formatting without changes
```

### Running Tests for a Specific Module

```bash
zig test src/compute/runtime/engine.zig     # Test specific file
zig test --test-filter="engine init"        # Run tests matching pattern
```

### Feature Flags

Build with specific features enabled/disabled:

```bash
zig build -Denable-ai=true -Denable-gpu=false -Denable-web=true -Denable-database=true
```

Core flags (defaults in parentheses):
- `-Denable-ai` (true) - AI features and connectors
- `-Denable-gpu` (true) - GPU acceleration
- `-Denable-web` (true) - Web utilities and HTTP
- `-Denable-database` (true) - Vector database and storage
- `-Denable-network` (false) - Distributed network compute
- `-Denable-profiling` (false) - Profiling and metrics

GPU backends: `-Dgpu-cuda`, `-Dgpu-vulkan`, `-Dgpu-metal`, `-Dgpu-webgpu`, `-Dgpu-opengl`, `-Dgpu-opengles`, `-Dgpu-webgl2`

## Architecture

### Layered Structure

1. **Public API** (`src/abi.zig`) - Main entry point with curated re-exports. Use `abi.init()`, `abi.shutdown()`, `abi.version()`.

2. **Framework Layer** (`src/framework/`) - Lifecycle management, feature orchestration, runtime configuration, and plugin system.

3. **Compute Engine** (`src/compute/`) - Work-stealing scheduler, lock-free concurrent data structures, memory arena allocation, GPU integration with CPU fallback.

4. **Feature Stacks** (`src/features/`) - Vertical feature modules:
   - `ai/` - LLM connectors (OpenAI, HuggingFace, Ollama), agents, transformers, training pipelines
   - `gpu/` - GPU backend implementations with fallback runtimes
   - `database/` - WDBX vector database with backup/restore
   - `web/` - HTTP client/server helpers
   - `monitoring/` - Logging, metrics, tracing
   - `network/` - Distributed task serialization

5. **Shared Utilities** (`src/shared/`) - Platform abstractions, SIMD acceleration, crypto, JSON, filesystem helpers.

### Key Patterns

- **Feature Gating**: Compile-time feature enabling via build options checked with `build_options.enable_*`
- **VTable Pattern**: Polymorphic workload execution for CPU and GPU variants (`WorkloadVTable`, `GPUWorkloadVTable`)
- **Allocator Ownership**: Explicit memory management, prefer `std.ArrayListUnmanaged` over `std.ArrayList`
- **Lifecycle Management**: Init/deinit patterns with `defer`/`errdefer` cleanup

## Zig 0.16 Conventions

### Memory Management

```zig
// Preferred - explicit allocator passing
var list = std.ArrayListUnmanaged(u8).empty;
try list.append(allocator, item);
list.deinit(allocator);

// Avoid - hidden allocator dependency
var list = std.ArrayList(u8).init(allocator);
```

### Format Specifiers

```zig
// Use {t} for enums/errors instead of @tagName()
std.debug.print("Status: {t}\n", .{status});

// Avoid
std.debug.print("Status: {s}\n", .{@tagName(status)});
```

### Style

- 4 spaces, no tabs, lines under 100 chars
- PascalCase types, snake_case functions/variables
- Explicit imports only (no `usingnamespace`)
- Use specific error sets, not `anyerror`

## Environment Variables

Connector-specific:
- `ABI_OPENAI_API_KEY` / `OPENAI_API_KEY`
- `ABI_HF_API_TOKEN` / `HF_API_TOKEN`
- `ABI_OLLAMA_HOST` / `OLLAMA_HOST` (default: `http://127.0.0.1:11434`)
