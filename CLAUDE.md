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
zig build run -- --version   # Show version info
zig fmt .                    # Format all code
zig fmt --check .            # Check formatting without changes
zig build -Doptimize=ReleaseFast  # Optimized release build
```

### Running Tests for a Specific Module

```bash
zig test src/compute/runtime/engine.zig     # Test specific file
zig build test -- --test-filter "engine init"  # Run tests matching pattern
zig build test -Denable-gpu=true -Denable-network=true  # Test with features
```

### CLI Entrypoint

CLI resolution prefers `tools/cli/main.zig` and falls back to `src/main.zig` if not present.

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
- `-Denable-network` (true) - Distributed network compute
- `-Denable-profiling` (true) - Profiling and metrics

GPU backends: `-Dgpu-cuda`, `-Dgpu-vulkan`, `-Dgpu-stdgpu`, `-Dgpu-metal`, `-Dgpu-webgpu`, `-Dgpu-opengl`, `-Dgpu-opengles`, `-Dgpu-webgl2`

### Additional Build Targets

```bash
zig build examples           # Build all example programs
zig build wasm               # Build WASM bindings (outputs to zig-out/wasm/)
zig build check-wasm         # Check WASM compilation without installing
zig build profile            # Build with profiling enabled (ReleaseFast)
zig build benchmarks         # Run comprehensive benchmarks
zig build benchmark-legacy   # Run legacy compute benchmarks
```

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
// Use modern format specifiers instead of manual conversions
std.debug.print("Status: {t}\n", .{status});     // {t} for enum/error values
std.debug.print("Size: {B}\n", .{size});         // {B} for byte sizes (SI)
std.debug.print("Size: {Bi}\n", .{size});        // {Bi} for byte sizes (binary)
std.debug.print("Duration: {D}\n", .{dur});      // {D} for durations
std.debug.print("Data: {b64}\n", .{data});       // {b64} for base64

// Avoid @tagName()
std.debug.print("Status: {s}\n", .{@tagName(status)});  // Don't do this
```

### Style

- 4 spaces, no tabs, lines under 100 chars
- PascalCase types, snake_case functions/variables
- Explicit imports only (no `usingnamespace`)
- Use specific error sets, not `anyerror`

## Environment Variables

Connector-specific:
- `ABI_OPENAI_API_KEY` / `OPENAI_API_KEY`
- `ABI_OPENAI_BASE_URL` (default: `https://api.openai.com/v1`)
- `ABI_OPENAI_MODE` (`responses`, `chat`, or `completions`)
- `ABI_HF_API_TOKEN` / `HF_API_TOKEN` / `HUGGING_FACE_HUB_TOKEN`
- `ABI_HF_BASE_URL` (default: `https://api-inference.huggingface.co`)
- `ABI_OLLAMA_HOST` / `OLLAMA_HOST` (default: `http://127.0.0.1:11434`)
- `ABI_OLLAMA_MODEL` (default: `llama3.2`)
- `ABI_LOCAL_SCHEDULER_URL` / `LOCAL_SCHEDULER_URL` (default: `http://127.0.0.1:8081`)
- `ABI_LOCAL_SCHEDULER_ENDPOINT` (default: `/schedule`)

## Key API Notes

### Compute Engine Timeouts

When using `runWorkload(engine, workload, timeout_ms)`:
- `timeout_ms=0`: Immediately returns `EngineError.Timeout` if not ready
- `timeout_ms>0`: Waits specified milliseconds before timeout
- `timeout_ms=null`: Waits indefinitely

### WDBX Backup/Restore Security

Backup and restore operations are restricted to the `backups/` directory only. Filenames must not contain path traversal sequences (`..`), absolute paths, or Windows drive letters.

### WASM Build Constraints

When building for WASM (`zig build wasm`), these features are automatically disabled:
- `enable-database` - No `std.Io.Threaded` support
- `enable-network` - No socket support
- `enable-gpu` - Native GPU backends unavailable
- `enable-web` - WebGPU bindings simplified

WASM bindings are in `bindings/wasm/abi_wasm.zig`.

## CLI Commands

The CLI (`zig build run -- <command>`) provides these subcommands:

- `db <subcommand>` - Database operations (add, query, stats, optimize, backup, restore, serve)
- `agent [--message]` - Run AI agent (interactive or one-shot mode)
- `config [command]` - Configuration management (init, show, validate)
- `explore [options] <query>` - Search and explore codebase
- `gpu [subcommand]` - GPU info (backends, devices, summary, default)
- `network [command]` - Network registry (list, register, status)
- `simd` - Run SIMD performance demo
- `system-info` - Show system and framework status

### Database CLI Examples

```bash
zig build run -- db stats                              # Show database statistics
zig build run -- db add --id 1 --embed "text to embed" # Add vector via embedding
zig build run -- db add --id 2 --vector "1.0,2.0,3.0"  # Add raw vector
zig build run -- db backup --path mybackup.db          # Backup to file
zig build run -- db restore --path mybackup.db         # Restore from file
```

### Explore CLI Examples

```bash
zig build run -- explore "pub fn" --level quick        # Quick search
zig build run -- explore "fn init" --level thorough    # Thorough search
```
