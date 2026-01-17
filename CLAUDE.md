# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Quick Reference

```bash
zig build                              # Build the project
zig build test --summary all           # Run tests with detailed output
zig fmt .                              # Format code (run after edits)
zig build run -- --help                # CLI help

# Feature-gated builds
zig build -Denable-ai=true -Denable-gpu=false -Denable-database=true

# Test specific file or filter tests
zig test src/compute/runtime/engine.zig
zig test src/tests/mod.zig --test-filter "pattern"
```

## LLM Instructions

- Keep changes minimal and consistent with existing patterns; avoid breaking public APIs unless requested.
- Preserve feature gating: stub modules must mirror the real API and return `error.*Disabled`.
- Use Zig 0.16 conventions (`std.Io`, `std.ArrayListUnmanaged`, `{t}` formatting, explicit allocators).
- Always clean up resources with `defer`/`errdefer`; use specific error sets (no `anyerror`).
- Run `zig fmt .` after code edits and `zig build test --summary all` when behavior changes.
- Refer to **[TODO.md](TODO.md)** for pending work. All Llama-CPP parity tasks are complete.

## Common Gotchas

| Issue | Solution |
|-------|----------|
| `--test-filter` not working | Use `zig test file.zig --test-filter "pattern"`, NOT `zig build test --test-filter` |
| `std.fs.cwd()` doesn't exist | Use `std.Io.Dir.cwd()` with an `std.Io` context (Zig 0.16) |
| Backup/restore path errors | Paths restricted to `backups/` directory; no `..`, absolute paths, or drive letters |
| Feature disabled errors | Rebuild with `-Denable-<feature>=true` |
| GPU backend conflicts | Enable only one backend: `-Dgpu-cuda=true -Dgpu-vulkan=false` |

## Feature Flags

| Flag | Default | Description |
|------|---------|-------------|
| `-Denable-ai` | true | AI features and connectors |
| `-Denable-llm` | true (if AI) | Local LLM inference |
| `-Denable-explore` | true (if AI) | AI code exploration |
| `-Denable-gpu` | true | GPU acceleration |
| `-Denable-web` | true | Web utilities and HTTP |
| `-Denable-database` | true | Vector database (WDBX) |
| `-Denable-network` | true | Distributed compute |
| `-Denable-profiling` | true | Profiling and metrics |

GPU backends: `-Dgpu-vulkan` (default), `-Dgpu-cuda`, `-Dgpu-metal`, `-Dgpu-webgpu`, `-Dgpu-opengl`, `-Dgpu-opengles`, `-Dgpu-webgl2`, `-Dgpu-stdgpu` (CPU fallback)

## Project Structure

```
abi/
├── src/
│   ├── abi.zig              # Public API entry point
│   ├── core/                # I/O, diagnostics, collections
│   ├── compute/             # Runtime, concurrency, gpu, memory, network, profiling
│   ├── features/            # ai/, database/, gpu/, monitoring/, network/, connectors/
│   ├── framework/           # Lifecycle and orchestration
│   ├── shared/              # logging/, observability/, security/, utils/
│   └── tests/               # Test utilities (proptest)
├── tools/cli/               # CLI implementation
├── benchmarks/              # Benchmark suites
└── docs/                    # Documentation
```

**Module conventions**: `mod.zig` (entry), `stub.zig` (feature-gated placeholder), underscored submodules (`error_handling.zig`)

## Architecture

### Layers

1. **Public API** (`src/abi.zig`) - `abi.init()`, `abi.shutdown()`, `abi.version()`
2. **Framework** (`src/framework/`) - Lifecycle, feature orchestration
3. **Compute** (`src/compute/`) - Work-stealing scheduler, lock-free structures, GPU integration
4. **Features** (`src/features/`) - AI, GPU, database, web, monitoring, network, connectors
5. **Shared** (`src/shared/`) - Platform abstractions, SIMD, crypto, JSON, filesystem

### Key Patterns

**Feature Gating**: Compile-time via `build_options.enable_*`. Disabled features use stub modules returning `error.*Disabled`.

```zig
const impl = if (build_options.enable_feature) @import("real.zig") else @import("stub.zig");
```

**Stub Requirements**: Mirror full API (structs, functions, constants); return `error.<Feature>Disabled`.

**VTable Pattern**: Polymorphic workload execution for CPU/GPU variants (`WorkloadVTable`, `GPUWorkloadVTable`).

**Lifecycle**: Strict init/deinit with `defer`/`errdefer`. Resources cleaned up in reverse initialization order.

**Module Lifecycle**: Use `ModuleLifecycle` (thread-safe) or `SimpleModuleLifecycle` (lock-free) from `src/shared/utils/lifecycle.zig`.

### Concurrency Primitives (`src/compute/`)

- `WorkStealingQueue` - LIFO for owner, FIFO for thieves
- `LockFreeQueue/Stack` - Atomic CAS-based collections
- `PriorityQueue` - Lock-free task scheduling
- `ShardedMap` - Partitioned data for reduced contention
- `Backoff` - Exponential backoff with spin-loop hints

### Runtime Patterns (`src/compute/runtime/`)

- `Future` - Async results with `.then()`, `.catch()`, `.finally()`
- `CancellationToken` - Cooperative cancellation
- `TaskGroup` - Hierarchical task grouping

## Zig 0.16 Conventions

### I/O and File Operations

```zig
// Synchronous file I/O
var io_backend = std.Io.Threaded.init(allocator, .{ .environ = std.process.Environ.empty });
defer io_backend.deinit();
const io = io_backend.io();

const content = try std.Io.Dir.cwd().readFileAlloc(io, path, allocator, .limited(10 * 1024 * 1024));
defer allocator.free(content);
```

### Memory and Formatting

```zig
// Prefer unmanaged containers
var list = std.ArrayListUnmanaged(u8).empty;
try list.append(allocator, item);
list.deinit(allocator);

// Modern format specifiers
std.debug.print("Status: {t}\n", .{status});   // {t} for enum/error (not @tagName)
std.debug.print("Size: {B}\n", .{size});       // {B} for bytes
std.debug.print("Duration: {D}\n", .{dur});    // {D} for durations
```

### Timing and Sleep

```zig
// Timer
var timer = std.time.Timer.start() catch return error.TimerFailed;
const elapsed_ns = timer.read();

// Sleep (use time utilities)
const time_utils = @import("src/shared/utils/time.zig");
time_utils.sleepMs(100);
```

### Style

- 4 spaces, no tabs, lines under 100 chars
- **Types**: PascalCase (`Engine`, `TaskConfig`)
- **Functions/Variables**: camelCase (`createEngine`, `taskId`)
- **Constants**: SCREAMING_SNAKE_CASE (`MAX_TASKS`)
- **Struct Fields**: `allocator` first, then config/state, collections, resources, flags

## Environment Variables

Connector config uses ABI-prefixed vars with fallback: `ABI_OPENAI_API_KEY` → `OPENAI_API_KEY`

| Variable | Default | Description |
|----------|---------|-------------|
| `ABI_OPENAI_API_KEY` / `OPENAI_API_KEY` | - | OpenAI API key |
| `ABI_OPENAI_BASE_URL` | `https://api.openai.com/v1` | OpenAI base URL |
| `ABI_OPENAI_MODE` | - | `responses`, `chat`, or `completions` |
| `ABI_OLLAMA_HOST` / `OLLAMA_HOST` | `http://127.0.0.1:11434` | Ollama host |
| `ABI_OLLAMA_MODEL` | `llama3.2` | Default Ollama model |
| `ABI_HF_API_TOKEN` / `HF_API_TOKEN` | - | HuggingFace token |
| `DISCORD_BOT_TOKEN` | - | Discord bot token |

## CLI Commands

```bash
# Database
zig build run -- db stats
zig build run -- db add --id 1 --embed "text"
zig build run -- db backup --path backup.db

# Agent
zig build run -- agent --persona coder
zig build run -- agent -m "Hello"
zig build run -- agent --list-personas

# LLM
zig build run -- llm info model.gguf
zig build run -- llm chat model.gguf

# GPU
zig build run -- gpu backends
zig build run -- gpu devices

# Training
zig build run -- train run --epochs 10 --batch-size 32
zig build run -- train resume ./checkpoint.ckpt

# Other
zig build run -- explore "fn init" --level thorough
zig build run -- system-info
```

## GPU API

> See [docs/gpu.md](docs/gpu.md) for detailed GPU documentation.

```zig
var gpu = try abi.Gpu.init(allocator, .{ .memory_mode = .automatic });
defer gpu.deinit();

const a = try gpu.createBufferFromSlice(f32, &data_a, .{});
defer gpu.destroyBuffer(a);
```

**Memory modes**: `automatic` (recommended), `explicit`, `unified`
**Operations**: `vectorAdd`, `matrixMultiply`, `reduceSum`, `dotProduct`, `softmax`

## Testing

```bash
zig build test --summary all                    # All tests with details
zig test src/compute/runtime/engine.zig         # Single file
zig test src/tests/mod.zig --test-filter "pat"  # Filter tests
zig build test -Denable-gpu=true                # With features
```

**Hardware-gated tests**: Use `error.SkipZigTest` when feature unavailable.

**Property testing** (`src/tests/proptest.zig`): Generators for `int`, `float`, `bool`, `slice`, `string`, `optional`, `oneOf`.

## Troubleshooting

See [docs/troubleshooting.md](docs/troubleshooting.md) for detailed solutions. Common issues:

- **Feature disabled**: Rebuild with `-Denable-<feature>=true`
- **GPU not detected**: Check `zig build run -- gpu backends` and drivers
- **Timeout errors**: Increase timeout or use `null` for indefinite wait
- **Out of memory**: Use arena allocators, reduce batch sizes, add `defer` cleanup

## References

| Document | Description |
|----------|-------------|
| [docs/intro.md](docs/intro.md) | Architecture overview |
| [API_REFERENCE.md](API_REFERENCE.md) | Public API summary |
| [docs/migration/zig-0.16-migration.md](docs/migration/zig-0.16-migration.md) | Zig 0.16 patterns |
| [TODO.md](TODO.md) | Pending work |
| [ROADMAP.md](ROADMAP.md) | Version milestones |
| [docs/troubleshooting.md](docs/troubleshooting.md) | Common issues |
