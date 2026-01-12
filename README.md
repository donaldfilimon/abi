# ABI Framework

Modern Zig framework for modular AI services, vector search, and systems tooling.

## Highlights

- AI agent runtime, training pipelines, and data structures
- Vector database helpers (WDBX) with unified API
- High-performance compute runtime with work-stealing scheduler
- GPU backends (CUDA, Vulkan, Metal, WebGPU) with feature gating
- Network distributed compute with serialization
- Profiling and metrics collection
- Web utilities (HTTP client/server helpers, weather helper)
- Monitoring (logging, metrics, tracing, profiling)

## Documentation

- Documentation Index: [docs/intro.md](docs/intro.md)
- Concise API summary: [API_REFERENCE.md](API_REFERENCE.md)
- Migration Guide: [docs/migration/zig-0.16-migration.md](docs/migration/zig-0.16-migration.md)

## Requirements

- Zig 0.16.x (minimum 0.16.0; CI uses 0.16.x)

## Build

```bash
zig build
zig build test
zig build -Doptimize=ReleaseFast
zig build -Denable-ai=true -Denable-gpu=false -Denable-web=true -Denable-database=true
```

## Feature Flags

- `-Denable-ai` (default: `true`) - Enable AI features and modules
- `-Denable-gpu` (default: `true`) - Enable GPU acceleration features
- `-Denable-web` (default: `true`) - Enable web utilities and HTTP features
- `-Denable-database` (default: `true`) - Enable database and vector search features
- `-Denable-network` (default: `true`) - Enable distributed network compute
- `-Denable-profiling` (default: `true`) - Enable profiling and metrics collection
- `-Dgpu-cuda` - Enable CUDA GPU backend
- `-Dgpu-vulkan` - Enable Vulkan GPU backend
- `-Dgpu-metal` - Enable Metal GPU backend
- `-Dgpu-webgpu` - Enable WebGPU backend (requires `-Denable-web`)
- `-Dgpu-opengl` - Enable OpenGL backend
- `-Dgpu-opengles` - Enable OpenGL ES backend
- `-Dgpu-webgl2` - Enable WebGL2 backend (requires web/wasm target)

## Development

### Code Formatting

```bash
zig fmt .                    # Format all code
zig fmt --check .           # Check formatting without changes
```

### Testing

```bash
zig build test                                    # Run all tests
zig build test -Denable-gpu=true -Denable-network=true  # Test with features
zig build benchmarks                               # Run performance benchmarks
```

### CLI Usage

```bash
zig build run -- --help                           # Show CLI help
zig build run -- --version                        # Show version info
```

## Quick Example (Zig 0.16)

```zig
const std = @import("std");
const abi = @import("abi");

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;

    var framework = try abi.init(allocator, abi.FrameworkOptions{});
    defer abi.shutdown(&framework);

    std.debug.print("ABI version: {s}\n", .{abi.version()});
}
```

## Compute Engine Example

```zig
const std = @import("std");
const abi = @import("abi");

fn computeTask(_: std.mem.Allocator) !u32 {
    return 42;
}

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;

    var engine = try abi.compute.createDefaultEngine(allocator);
    defer engine.deinit();

    // Using runTask (or its alias runWorkload)
    const result = try abi.compute.runTask(&engine, u32, computeTask, 1000);
    std.debug.print("Result: {d}\n", .{result});
}
```

## GPU Workload Example

```zig
const std = @import("std");
const abi = @import("abi");

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;

    var gpu_workload: u32 = 0;

    const cpu_vtable = abi.compute.WorkloadVTable{
        .execute = struct {
            fn execute(
                ctx: *abi.compute.ExecutionContext,
                user: *anyopaque,
            ) !abi.compute.ResultHandle {
                _ = ctx;
                const ptr: *u32 = @ptrCast(@alignCast(user));
                ptr.* += 1;
                return abi.compute.ResultHandle.fromSlice(&.{});
            }
        }.execute,
    };

    const gpu_vtable = abi.compute.GPUWorkloadVTable{
        .execute = struct {
            fn execute(
                ctx: *abi.compute.ExecutionContext,
                user: *anyopaque,
            ) !abi.compute.ResultHandle {
                _ = ctx;
                const ptr: *u32 = @ptrCast(@alignCast(user));
                ptr.* += 10;
                return abi.compute.ResultHandle.fromSlice(&.{});
            }
        }.execute,
    };

    const item = abi.compute.WorkItem{
        .id = 0,
        .user = &gpu_workload,
        .vtable = &cpu_vtable,
        .priority = 0,
        .hints = .{
            .cpu_affinity = null,
            .estimated_duration_us = 2_000,
            .prefers_gpu = true,
            .requires_gpu = false,
        },
        .gpu_vtable = &gpu_vtable,
    };

    var ctx = abi.compute.ExecutionContext{ .allocator = allocator };
    const result = try abi.compute.runWorkItem(&ctx, &item);
    defer result.deinit();
}
```

## GPU Memory & Pool Helpers

```zig
const std = @import("std");
const abi = @import("abi");

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;

    var pool = abi.gpu.createPool(allocator, 16 * 1024 * 1024);
    defer pool.deinit();

    const flags = abi.gpu.BufferFlags{
        .device_local = true,
        .zero_init = true,
    };
    const buffer = try pool.allocate(4096, flags);
    defer _ = pool.free(buffer);

    try buffer.writeFromHost(&.{ 1, 2, 3, 4 });
    try buffer.copyToDevice();

    buffer.asSlice()[0] = 9;
    try buffer.copyToHost();

    const stats = pool.stats();
    std.debug.print("GPU pool usage: {d:.2}%\n", .{stats.usage_ratio * 100.0});
}
```

## Profiling Example

```zig
var metrics = try abi.compute.MetricsCollector.init(
    gpa.allocator(),
    abi.compute.DEFAULT_METRICS_CONFIG,
    4,
);
defer metrics.deinit();

metrics.recordTaskExecution(0, 1500);
metrics.recordTaskExecution(1, 900);

const summary = metrics.getSummary();
std.debug.print("Total tasks: {d}\n", .{summary.total_tasks});
std.debug.print("Avg execution: {d} us\n", .{summary.avg_execution_ns / 1000});
std.debug.print("Min execution: {d} us\n", .{summary.min_execution_ns / 1000});
std.debug.print("Max execution: {d} us\n", .{summary.max_execution_ns / 1000});
```

## Network Serialization Example

```zig
// Serialize task for network transfer
const payload_type = "matrix_multiply";
const user_data = "serialized_workload_data";

const serialized = try abi.network.serializeTask(
    gpa.allocator(),
    &item,
    payload_type,
    user_data,
);
defer gpa.allocator().free(serialized);

// Deserialize on remote node
const deserialized = try abi.network.deserializeTask(gpa.allocator(), serialized);
defer {
    gpa.allocator().free(deserialized.payload_type);
    gpa.allocator().free(deserialized.user_data);
}
```

## Benchmarking Example

```zig
const results = try abi.compute.runBenchmarks(gpa.allocator());
defer gpa.allocator().free(results);

for (results) |bench| {
    const ops = @as(u64, @intFromFloat(bench.ops_per_sec));
    std.debug.print(
        "{s}: {d} ops/sec ({d} iterations, {d} ns)\n",
        .{ bench.name, ops, bench.iterations, bench.duration_ns },
    );
}
```

## Architecture Overview

- `src/abi.zig`: public API surface and curated re-exports
- `src/root.zig`: root module entrypoint
- `src/framework/`: runtime config, feature orchestration, lifecycle
- `src/features/`: vertical feature stacks (AI, GPU, database, web, monitoring)
- `src/compute/`: compute runtime, memory, and concurrency
- `src/shared/`: shared utilities (logging, observability, platform, utils)

## Project Layout

```
abi/
├── src/                # Core library sources
│   ├── core/           # Core infrastructure
│   ├── compute/        # Compute runtime + memory + concurrency
│   ├── features/       # Feature modules (AI, GPU, web, etc.)
│   ├── framework/      # Framework configuration and runtime
│   ├── shared/         # Shared utilities
├── build.zig           # Build graph + feature flags
└── build.zig.zon        # Zig package metadata
```

## CLI

CLI entrypoint resolution prefers `tools/cli/main.zig` and falls back to
`src/main.zig` if the tools entrypoint is not present.

```bash
zig build run -- --help
zig build run -- --version
```

## Tests

```bash
# Run all tests
zig build test

# Run tests with specific features enabled
zig build test -Denable-gpu=true -Denable-network=true -Denable-profiling=true

# Run tests for specific module
zig test src/compute/runtime/engine.zig

# Run tests matching pattern
zig test --test-filter="engine init"

# Run benchmarks
zig build benchmarks
```

**Test Coverage:**

- Compute engine: Worker threads, work-stealing, result caching
- GPU: Buffer allocation, memory pool, serialization
- Network: Task/result serialization, node registry
- Profiling: Metrics collection, histograms
- Integration: 10+ end-to-end tests with feature gating

## Connector Environment Variables

- `ABI_OPENAI_API_KEY`, `OPENAI_API_KEY`
- `ABI_OPENAI_BASE_URL` (default `https://api.openai.com/v1`)
- `ABI_OPENAI_MODE` (`responses`, `chat`, or `completions`)
- `ABI_HF_API_TOKEN`, `HF_API_TOKEN`, `HUGGING_FACE_HUB_TOKEN`
- `ABI_HF_BASE_URL` (default `https://api-inference.huggingface.co`)
- `DISCORD_BOT_TOKEN` - Discord bot authentication token
- `ABI_LOCAL_SCHEDULER_URL`, `LOCAL_SCHEDULER_URL` (default `http://127.0.0.1:8081`)
- `ABI_LOCAL_SCHEDULER_ENDPOINT` (default `/schedule`)
- `ABI_OLLAMA_HOST`, `OLLAMA_HOST` (default `http://127.0.0.1:11434`)
- `ABI_OLLAMA_MODEL` (default `llama3.2`)

## Contributing

See `CONTRIBUTING.md` for development workflow and style guidelines.
## Contacts

`src/shared/contacts.zig` provides a centralized list of maintainer contacts extracted from the repository markdown files. It can be imported wherever contact information is needed.
