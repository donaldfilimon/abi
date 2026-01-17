# AGENTS.md

This file provides guidance for AI agents (Claude, GPT, Gemini, Copilot, and others) working with the ABI framework codebase.

## Quick Start for AI Agents

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

## Codebase Overview

**ABI Framework** is a Zig 0.16 framework for modular AI services, GPU compute, and vector databases.

### Architecture

The codebase uses a flat domain structure with unified configuration:

```
src/
├── abi.zig              # Public API entry point
├── config.zig           # Unified configuration system
├── framework.zig        # Framework orchestration with builder pattern
├── runtime/             # Always-on infrastructure
├── gpu/                 # GPU acceleration
├── ai/                  # AI module (llm/, embeddings/, agents/, training/)
├── database/            # Vector database (WDBX)
├── network/             # Distributed compute
├── observability/       # Metrics, tracing, profiling
├── web/                 # Web/HTTP utilities
├── internal/            # Shared utilities
└── shared/              # Legacy shared utilities

# Legacy (maintained for compatibility)
├── core/               # I/O, diagnostics
├── compute/            # Runtime, concurrency
└── features/           # Legacy feature organization

tools/cli/               # CLI with 16 commands
benchmarks/              # Performance benchmarks
examples/                # Example programs
docs/                    # Documentation
```

### Key Patterns

1. **Framework Initialization**:
   ```zig
   const abi = @import("abi");
   
   // Default init (all compile-time enabled features)
   var fw = try abi.init(allocator);
   defer fw.deinit();
   
   // Builder pattern
   var fw = try abi.Framework.builder(allocator)
       .withGpu(.{ .backend = .vulkan })
       .withAi(.{ .llm = .{} })
       .build();
   defer fw.deinit();
   
   // Access features
   if (fw.isEnabled(.gpu)) {
       const gpu = try fw.getGpu();
   }
   ```

2. **Configuration**: Use `Config` struct or `Builder` pattern
   ```zig
   const config = abi.Config{
       .gpu = .{ .backend = .cuda },
       .ai = .{ .llm = .{ .model_path = "./model.gguf" } },
   };
   ```

3. **Feature Gating**: Compile-time flags with stub modules
   ```zig
   const impl = if (build_options.enable_ai) @import("mod.zig") else @import("stub.zig");
   ```

4. **Module Convention**: `mod.zig` (entry), `stub.zig` (disabled placeholder)

5. **Type Naming**: PascalCase for types (`GpuBuffer`), camelCase for functions

6. **Memory**: Use `defer`/`errdefer` for cleanup, explicit allocators

## Critical Rules

### DO

- Read files before editing (use Read tool first)
- Run `zig fmt .` after code changes
- Run `zig build test --summary all` to verify changes
- Use specific error types (never `anyerror`)
- Follow existing patterns in the codebase
- Check CLAUDE.md for detailed guidelines

### DON'T

- Use deprecated Zig 0.15 APIs (`std.fs.cwd()` → use `std.Io.Dir.cwd()`)
- Create new files unless absolutely necessary
- Add features beyond what was requested
- Skip verification steps
- Guess at API signatures - read the source

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

### CLI Commands

All 16 commands follow the same pattern in `tools/cli/commands/`:

```zig
pub fn run(allocator: std.mem.Allocator, args: []const [:0]const u8) !void {
    // Parse args, execute, output results
}
```

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

**Feature enum** (for runtime checking):
```zig
pub const Feature = enum {
    gpu, ai, llm, embeddings, agents, training,
    database, network, observability, web,
};

// Check at runtime
if (fw.isEnabled(.gpu)) { ... }

// Check at compile time
if (Feature.gpu.isCompileTimeEnabled()) { ... }
```

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

## Training

The framework includes local LLM training capabilities:

```bash
# Basic training (uses synthetic data)
zig build run -- train run --epochs 2 --batch-size 16

# LLM fine-tuning
zig build run -- train llm models/gpt2.gguf --epochs 1 --batch-size 4

# Show configuration
zig build run -- train info
```

**Default Model**: GPT-2 Small (124M parameters) - open source, no authentication required.

## Testing

```bash
# Run all tests
zig build test --summary all

# Test single file
zig test src/compute/runtime/engine.zig

# Test with filter
zig test src/tests/mod.zig --test-filter "pattern"
```

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

## File Organization

| Directory | Purpose |
|-----------|---------|
| `src/abi.zig` | Public API entry point |
| `src/config.zig` | Unified configuration system |
| `src/framework.zig` | Framework orchestration |
| `src/runtime/` | Always-on infrastructure |
| `src/gpu/` | GPU acceleration with unified multi-backend API |
| `src/ai/` | AI module (llm, embeddings, agents, training) |
| `src/database/` | Vector database (WDBX) |
| `src/network/` | Distributed compute |
| `src/observability/` | Metrics, tracing, profiling |
| `src/web/` | Web/HTTP utilities |
| `src/internal/` | Shared utilities |
| `tools/cli/commands/` | CLI command implementations |
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

## Memory Management

```zig
// Always use defer for cleanup
const data = try allocator.alloc(u8, size);
defer allocator.free(data);

// Use errdefer for error paths
var resource = try Resource.init(allocator);
errdefer resource.deinit();
```

## Zig 0.16 Specifics

- Use `std.Io.Dir.cwd()` not `std.fs.cwd()`
- Use `std.time.Timer.start()` for timing
- Use `std.ArrayListUnmanaged` for explicit allocator control
- Escape reserved keywords with `@"error"` syntax

## Reference Documents

| Document | Purpose |
|----------|---------|
| [CLAUDE.md](CLAUDE.md) | Detailed development guidelines |
| [README.md](README.md) | Project overview |
| [docs/feature-flags.md](docs/feature-flags.md) | Feature flags reference |
| [benchmarks/README.md](benchmarks/README.md) | Benchmark documentation |
| [API_REFERENCE.md](API_REFERENCE.md) | API documentation |

## Need Help?

- Check existing code for patterns
- Read CLAUDE.md for detailed guidance
- Run `zig build run -- help` for CLI options
- Check docs/ for feature-specific documentation
