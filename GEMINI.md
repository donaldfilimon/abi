# GEMINI.md

This file provides guidance for Google Gemini AI when working with the ABI framework codebase.

## Quick Start

```bash
# Essential commands
zig build                              # Build the project
zig build test --summary all           # Run all tests
zig fmt .                              # Format code after edits
zig build run -- --help                # CLI help

# Additional build targets
zig build benchmarks                   # Run comprehensive benchmarks
zig build bench-competitive            # Run competitive benchmarks
zig build benchmark-legacy             # Run legacy performance benchmarks
zig build gendocs                      # Generate API documentation
zig build profile                      # Build with performance profiling
zig build wasm                         # Build WASM bindings
zig build check-wasm                   # Check WASM compilation
zig build examples                     # Build all examples

# Run example programs
zig build run-hello                    # Run hello example
zig build run-database                 # Run database example
zig build run-agent                    # Run agent example
zig build run-compute                  # Run compute example
zig build run-gpu                      # Run GPU example
zig build run-network                  # Run network example
zig build run-discord                  # Run discord example
```

## Project Overview

**ABI Framework** is a Zig 0.16 framework providing:
- **AI Services**: Agents, LLM inference, training, embeddings
- **GPU Compute**: Unified API across Vulkan, CUDA, Metal, WebGPU
- **Vector Database**: WDBX with HNSW indexing
- **Distributed Computing**: Node discovery, Raft consensus

## Architecture

```
src/
├── abi.zig              # Public API: init(), shutdown(), version()
├── compute/
│   ├── gpu/             # GPU acceleration (73 files)
│   ├── runtime/         # Async runtime, futures, tasks
│   ├── concurrency/     # Lock-free structures
│   └── memory/          # Memory management
├── features/
│   ├── ai/              # AI agents, LLM (147 files)
│   ├── database/        # Vector database (31 files)
│   └── network/         # Distributed compute
├── framework/           # Lifecycle orchestration
└── shared/              # Utilities, logging, security

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

```zig
// Always use defer for cleanup
const data = try allocator.alloc(u8, size);
defer allocator.free(data);

// Use errdefer for error paths
var resource = try Resource.init(allocator);
errdefer resource.deinit();
```

## Critical Rules

### DO

- Read files before editing them
- Run `zig fmt .` after code changes
- Run `zig build test --summary all` to verify changes
- Use specific error types (never `anyerror`)
- Follow existing patterns in the codebase
- Use `defer`/`errdefer` for all resource cleanup

### DON'T

- Use deprecated Zig 0.15 APIs
  - `std.fs.cwd()` → use `std.Io.Dir.cwd()`
- Create new files unless absolutely necessary
- Add features beyond what was requested
- Skip verification steps
- Guess at API signatures - read the source

## Feature Flags

| Flag | Default | Description |
|------|---------|-------------|
| `-Denable-ai` | true | AI agents, LLM, training |
| `-Denable-gpu` | true | GPU acceleration |
| `-Denable-database` | true | Vector database (WDBX) |
| `-Denable-network` | true | Distributed compute |
| `-Denable-web` | true | HTTP client/server |
| `-Denable-profiling` | true | Metrics and profiling |
| `-Denable-explore` | true | Codebase exploration |
| `-Denable-llm` | true | Local LLM inference |

See [docs/feature-flags.md](docs/feature-flags.md) for complete reference.

**Cache Options:**

| Option | Default | Description |
|--------|---------|-------------|
| `-Dcache-dir` | `.zig-cache` | Directory for build cache |
| `-Dglobal-cache-dir` | (none) | Directory for global build cache |

## WASM Target Limitations

When building for WebAssembly (`zig build wasm`), these features are auto-disabled:

| Feature | Status | Reason |
|---------|--------|--------|
| `enable-database` | Disabled | No `std.Io.Threaded` support |
| `enable-network` | Disabled | No socket operations |
| `enable-gpu` | Disabled | Native GPU unavailable |
| `enable-web` | Disabled | Simplifies initial pass |
| `enable-profiling` | Disabled | Platform limitations |
| All GPU backends | Disabled | Including WebGPU (for now) |

Use `zig build check-wasm` to verify WASM compilation without full build.

## GPU Backends

| Backend | Flag | Platform |
|---------|------|----------|
| Vulkan | `-Dgpu-vulkan` | Cross-platform (default) |
| CUDA | `-Dgpu-cuda` | NVIDIA |
| Metal | `-Dgpu-metal` | Apple |
| WebGPU | `-Dgpu-webgpu` | Web/Native |
| OpenGL | `-Dgpu-opengl` | Desktop (legacy) |
| OpenGL ES | `-Dgpu-opengles` | Mobile/Embedded |
| WebGL2 | `-Dgpu-webgl2` | Web browsers |
| stdgpu | `-Dgpu-stdgpu` | CPU fallback |

## Zig 0.16 Specifics

### File I/O

```zig
var io_backend = std.Io.Threaded.init(allocator, .{ .environ = std.process.Environ.empty });
defer io_backend.deinit();
const io = io_backend.io();

const content = try std.Io.Dir.cwd().readFileAlloc(io, path, allocator, .limited(10 * 1024 * 1024));
defer allocator.free(content);
```

### Timing

```zig
var timer = std.time.Timer.start() catch return error.TimerFailed;
const elapsed_ns = timer.read();
```

### Memory

```zig
// Unmanaged containers for explicit allocator control
var list = std.ArrayListUnmanaged(u8).empty;
try list.append(allocator, item);
list.deinit(allocator);
```

### Reserved Keywords

Escape reserved keywords:
```zig
@"error"   // not: error
@"type"    // not: type
```

## Training

The framework supports local LLM training:

```bash
# Basic training (uses synthetic data)
zig build run -- train run --epochs 2 --batch-size 16

# LLM fine-tuning
zig build run -- train llm models/gpt2.gguf --epochs 1 --batch-size 4

# Show training configuration
zig build run -- train info
```

**Default Model**: GPT-2 Small (124M parameters) - open source, no authentication required.

## CLI Commands

16 commands available:

| Command | Purpose |
|---------|---------|
| `db` | Database operations |
| `agent` | AI agent interaction |
| `llm` | LLM inference |
| `train` | Model training |
| `bench` | Benchmarking |
| `embed` | Embeddings |
| `gpu` | GPU management |
| `network` | Network nodes |
| `config` | Configuration |
| `explore` | Code search |
| `discord` | Discord bot |
| `simd` | SIMD demo |
| `system-info` | System info |
| `tui` | Interactive UI |
| `version` | Version info |
| `help` | Help text |

## Testing

```bash
# Run all tests
zig build test --summary all

# Test single file
zig test src/compute/runtime/engine.zig

# Test with filter
zig test src/tests/mod.zig --test-filter "pattern"
```

## Error Handling

Use specific error sets:

```zig
const MyError = error{
    InvalidInput,
    ResourceExhausted,
    OperationFailed,
};

fn myFunction() MyError!Result {
    // ...
}
```

## Common Tasks

### Adding a New Feature

1. Check if feature flag exists in `build.zig`
2. Create `mod.zig` with implementation
3. Create `stub.zig` that mirrors the API but returns `error.FeatureDisabled`
4. Add to parent `mod.zig` with conditional import
5. Run tests

### Fixing a Bug

1. Read the relevant source files first
2. Understand the existing code pattern
3. Make minimal changes
4. Run `zig fmt .`
5. Run tests to verify

### Modifying GPU Code

1. Check `src/gpu/mod.zig` for type definitions
2. Use `GpuBuffer` (PascalCase) naming convention
3. Ensure backend compatibility
4. Test with multiple backends if possible

## File Organization

| Directory | Purpose |
|-----------|---------|
| `src/abi.zig` | Public API entry point |
| `src/gpu/` | GPU acceleration |
| `src/features/ai/` | AI/LLM features |
| `src/database/` | Vector database |
| `tools/cli/commands/` | CLI implementations |
| `benchmarks/` | Performance benchmarks |
| `examples/` | Example programs |
| `docs/` | Documentation |

## Example Programs

Available in `examples/` directory:

| Example | Command | Purpose |
|---------|---------|---------|
| hello | `zig build run-hello` | Basic framework usage |
| database | `zig build run-database` | Vector database operations |
| agent | `zig build run-agent` | AI agent demonstration |
| compute | `zig build run-compute` | Runtime and task execution |
| gpu | `zig build run-gpu` | GPU acceleration demo |
| network | `zig build run-network` | Distributed compute example |
| discord | `zig build run-discord` | Discord bot integration |

Build all examples: `zig build examples`

## Debugging

**Debug builds:** `zig build -Doptimize=Debug` (default) or `zig build -Doptimize=ReleaseSafe` for release with debug info.

**Memory leak detection:**
```zig
var gpa = std.heap.GeneralPurposeAllocator(.{ .stack_trace_frames = 10 }){};
defer {
    const check = gpa.deinit();
    if (check == .leak) @panic("Memory leak detected");
}
```

See [CLAUDE.md](CLAUDE.md) for detailed debugging guides including GDB/LLDB reference and TrackingAllocator usage.

## Reference Documents

| Document | Purpose |
|----------|---------|
| [CLAUDE.md](CLAUDE.md) | Detailed development guidelines |
| [AGENTS.md](AGENTS.md) | AI agent guidance |
| [README.md](README.md) | Project overview |
| [docs/feature-flags.md](docs/feature-flags.md) | Feature flags reference |
| [benchmarks/README.md](benchmarks/README.md) | Benchmark documentation |
| [API_REFERENCE.md](API_REFERENCE.md) | API documentation |

## Need Help?

- Check existing code for patterns
- Read CLAUDE.md for detailed guidance
- Run `zig build run -- help` for CLI options
- Check docs/ for feature-specific documentation
