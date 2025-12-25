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

## Requirements
- Zig 0.16.x

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
- `-Denable-network` (default: `false`) - Enable distributed network compute
- `-Denable-profiling` (default: `false`) - Enable profiling and metrics collection
- `-Dgpu-cuda` - Enable CUDA GPU backend
- `-Dgpu-vulkan` - Enable Vulkan GPU backend
- `-Dgpu-metal` - Enable Metal GPU backend
- `-Dgpu-webgpu` - Enable WebGPU backend (requires `-Denable-web`)

## Quick Example
```zig
const std = @import("std");
const abi = @import("abi");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();

    var framework = try abi.init(gpa.allocator(), abi.FrameworkOptions{});
    defer abi.shutdown(&framework);

    std.debug.print("ABI version: {s}\n", .{abi.version()});
}
```

## Compute Engine Example
```zig
const std = @import("std");
const abi = @import("abi");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();

    // Initialize compute engine with 4 workers
    const config = abi.compute.DEFAULT_CONFIG;
    var engine = try abi.compute.Engine.init(gpa.allocator(), config);
    defer engine.deinit();

    // Submit work item
    const vtable = abi.compute.WorkloadVTable{
        .exec = struct {
            fn exec(user: *anyopaque, ctx: *abi.compute.ExecutionContext, a: std.mem.Allocator) !*anyopaque {
                _ = user;
                _ = ctx;
                // Do work here
                const result = try a.create(u64);
                result.* = 42;
                return result;
            }
        }.exec,
        .destroy = struct {
            fn destroy(user: *anyopaque, a: std.mem.Allocator) void {
                a.destroy(@as(*u64, @ptrCast(@alignCast(user))));
            }
        }.destroy,
        .name = "example",
    };

    const item = abi.compute.WorkItem{
        .id = 0,
        .user = undefined,
        .vtable = &vtable,
        .priority = 0.5,
        .hints = abi.compute.DEFAULT_HINTS,
        .gpu_vtable = null,
    };

    const task_id = try engine.submit(item);

    // Wait for result
    while (engine.poll() == null) {
        std.time.sleep(100_000_000); // 100ms
    }

    const result = engine.take(task_id).?;
    defer result.deinit(gpa.allocator());

    std.debug.print("Result: {}\n", .{result.as(u64).*});
}
```

## GPU Workload Example
```zig
// Submit GPU-preferred workload
const gpu_hints = abi.compute.GPUWorkloadHints{
    .prefers_gpu = true,
    .requires_double_precision = false,
    .estimated_memory_bytes = 1024 * 1024,
};

const item = abi.compute.WorkItem{
    .id = 0,
    .user = &gpu_workload,
    .vtable = &cpu_vtable,
    .priority = 0.5,
    .hints = abi.compute.WorkloadHints{
        .cpu_affinity = null,
        .estimated_duration_us = null,
        .prefers_gpu = true,
        .requires_gpu = false,
    },
    .gpu_vtable = &gpu_vtable, // Optional GPU vtable
};

const task_id = try engine.submit(item);
```

## Profiling Example
```zig
// Get metrics summary
if (engine.metrics_collector) |*mc| {
    const summary = mc.getSummary();

    std.debug.print("Total tasks: {}\n", .{summary.total_tasks});
    std.debug.print("Avg execution: {} μs\n", .{summary.avg_execution_ns / 1000});
    std.debug.print("Min execution: {} μs\n", .{summary.min_execution_ns / 1000});
    std.debug.print("Max execution: {} μs\n", .{summary.max_execution_ns / 1000});
    std.debug.print("Throughput: {} ops/sec\n", .{summary.total_tasks * 1_000_000_000 / summary.total_execution_ns});
}
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
// Run performance benchmark
const matrix_bench = abi.compute.MatrixMultBenchmark{
    .matrix_size = 256,
    .iterations = 100,
};

const benchmark = matrix_bench.create(gpa.allocator());
const result = try abi.compute.runBenchmark(gpa.allocator(), benchmark);

abi.compute.printBenchmarkResults(result);
```

## Architecture Overview
- `src/abi.zig`: public API surface and curated re-exports
- `src/root.zig`: root module entrypoint
- `src/framework/`: runtime config, feature orchestration, lifecycle
- `src/features/`: vertical feature stacks (AI, GPU, database, web, monitoring)
- `src/shared/`: shared utilities (logging, observability, platform, utils)
- `src/internal/legacy/`: backward-compat implementations and deprecated modules

## Project Layout
```
abi/
├── src/                # Core library sources
│   ├── core/           # Core infrastructure
│   ├── features/       # Feature modules (AI, GPU, web, etc.)
│   ├── framework/      # Framework configuration and runtime
│   ├── shared/         # Shared utilities
│   └── internal/       # Legacy + experimental modules
│       └── legacy/     # Backward-compat implementations
├── build.zig           # Build graph + feature flags
└── build.zig.zon        # Zig package metadata
```

## CLI
If a CLI entrypoint is present at `tools/cli/main.zig`, it provides a thin
wrapper for embedded usage (help + version). This tree currently omits that
entrypoint; re-add it or update `build.zig` to skip the CLI build step.

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
zig build benchmark
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
- `ABI_LOCAL_SCHEDULER_URL`, `LOCAL_SCHEDULER_URL` (default `http://127.0.0.1:8081`)
- `ABI_LOCAL_SCHEDULER_ENDPOINT` (default `/schedule`)
- `ABI_OLLAMA_HOST`, `OLLAMA_HOST` (default `http://127.0.0.1:11434`)
- `ABI_OLLAMA_MODEL` (default `llama3.2`)

## Contributing
See `CONTRIBUTING.md` for development workflow and style guidelines.
