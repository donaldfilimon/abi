# API Reference (Concise)

> For detailed usage guides, see [Documentation Index](docs/intro.md).
> For coding patterns and conventions, see [CONTRIBUTING.md](CONTRIBUTING.md).
> For comprehensive development guidance, see [CLAUDE.md](CLAUDE.md).

This is a high-level summary of the public ABI API surface. See the source for
implementation details.

## Core Entry Points

- `abi.init(allocator, config_or_options)` -> `Framework`
- `abi.shutdown(framework)`
- `abi.version()` -> `[]const u8`
- `abi.createDefaultFramework(allocator)` -> `Framework`
- `abi.createFramework(allocator, config_or_options)` -> `Framework`

**Example**:
```zig
const std = @import("std");
const abi = @import("abi");

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    var framework = try abi.init(allocator, .{});
    defer abi.shutdown(&framework);

    std.debug.print("ABI v{s} initialized\n", .{abi.version()});
}
```

## Framework Types

- `abi.Framework`
- `abi.FrameworkOptions`
- `abi.RuntimeConfig`
- `abi.Feature` and `abi.features.FeatureTag`

## Feature Namespaces

- `abi.ai` - agent runtime, tools, training pipelines
- `abi.database` - WDBX database and helpers
- `abi.gpu` - GPU backends and vector search helpers
- `abi.web` - HTTP helpers, web utilities
- `abi.monitoring` - logging, metrics, tracing, profiling
- `abi.connectors` - connector interfaces and implementations

## WDBX Convenience API

- `abi.wdbx.createDatabase` / `connectDatabase` / `closeDatabase`
- `abi.wdbx.insertVector` / `searchVectors` / `deleteVector`
- `abi.wdbx.updateVector` / `getVector` / `listVectors`
- `abi.wdbx.getStats` / `optimize` / `backup` / `restore`

**Security Note for backup/restore**:

- Backup and restore operations are restricted to the `backups/` directory only
- Filenames must not contain path traversal sequences (`..`), absolute paths, or Windows drive letters
- Invalid filenames will return `PathValidationError`
- The `backups/` directory is created automatically if it doesn't exist
- This restriction prevents path traversal attacks (see SECURITY.md for details)

## Compute Engine API

- `abi.compute.DistributedComputeEngine` - Main compute runtime (alias: `abi.compute.runtime.DistributedComputeEngine`)
- `abi.compute.createDefaultEngine(allocator)` -> `!Engine` - Create engine with default config
- `abi.compute.createEngine(allocator, config)` -> `!Engine` - Create engine with custom config
- `abi.compute.submitTask(engine, ResultType, task)` -> `!TaskId` - Submit a task for execution
- `abi.compute.waitForResult(engine, ResultType, id, timeout_ms)` -> `!Result` - Wait for task result
- `abi.compute.runTask(engine, ResultType, task, timeout_ms)` -> `!Result` - Submit and wait for result
- `abi.compute.runWorkload(engine, ResultType, workload, timeout_ms)` -> `!Result` - Alias for runTask

**Example**:
```zig
var engine = try abi.compute.createDefaultEngine(allocator);
defer engine.deinit();

fn computeTask(_: std.mem.Allocator) !u32 {
    return 42;
}

// Submit and wait in one call
const result = try abi.compute.runTask(&engine, u32, computeTask, 1000);
std.debug.print("Result: {d}\n", .{result});
```

See [Compute Guide](docs/compute.md) for detailed usage.

**Timeout Semantics**:

- `timeout_ms=0`: Immediately returns `EngineError.Timeout` if result not ready
- `timeout_ms>0`: Waits for the specified timeout (in milliseconds) before returning `EngineError.Timeout`
- `timeout_ms=null`: Waits indefinitely until result is ready

**Breaking Change (0.2.1)**: Prior to version 0.2.1, `timeout_ms=0` returned `ResultNotFound` after one check. This behavior has changed to return `EngineError.Timeout` immediately for clarity. Migration: Use `timeout_ms=1000` for a one-second timeout.

## AI & Agent API

- `abi.ai.Agent` - Conversational agent with history and configuration
- `abi.ai.Agent.init(allocator, config)` -> `!Agent` - Create a new agent
- `abi.ai.Agent.deinit()` - Clean up agent resources
- `abi.ai.Agent.process(input, allocator)` -> `![]u8` - Process input and return response
- `abi.ai.Agent.chat(input, allocator)` -> `![]u8` - Alias for process() providing chat interface
- `abi.ai.train(allocator, config)` -> `!TrainingReport` - Run training pipeline
- `abi.ai.federated.Coordinator` - Federated learning coordinator
- `abi.ai.federated.Coordinator.init(allocator, config, model_size)` -> `!Coordinator`
- `abi.ai.federated.Coordinator.registerNode(node_id)` -> `!void`
- `abi.ai.federated.Coordinator.submitUpdate(update)` -> `!void`
- `abi.ai.federated.Coordinator.aggregate()` -> `![]f32` - Aggregate updates

**Example**:
```zig
var agent = try abi.ai.Agent.init(allocator, .{
    .name = "assistant",
    .temperature = 0.7,
});
defer agent.deinit();

const response = try agent.chat("Hello!", allocator);
defer allocator.free(response);
std.debug.print("Agent: {s}\n", .{response});
```

See [AI Guide](docs/ai.md) for detailed usage.

## Connectors API

- `abi.connectors.openai` - OpenAI API connector
- `abi.connectors.ollama` - Ollama API connector
- `abi.connectors.huggingface` - HuggingFace API connector

Each connector provides:
- `Client.init(allocator, config)` - Initialize client
- `Client.deinit()` - Clean up resources
- Connector-specific methods for inference/chat/completion

## SIMD API

- `abi.simd.vectorAdd(a, b, result)` - SIMD-accelerated vector addition
- `abi.simd.vectorDot(a, b)` -> `f32` - SIMD-accelerated dot product
- `abi.simd.vectorL2Norm(v)` -> `f32` - L2 norm computation
- `abi.simd.cosineSimilarity(a, b)` -> `f32` - Cosine similarity
- `abi.simd.matrixMultiply(a, b, result, m, n, k)` - Blocked matrix multiply with SIMD
- `abi.simd.hasSimdSupport()` -> `bool` - Check SIMD availability
- `abi.simd.getSimdCapabilities()` -> `SimdCapabilities` - Get platform SIMD info

**SimdCapabilities**:
- `.vector_size` - Vector width for SIMD operations
- `.has_simd` - Whether SIMD is available
- `.arch` - Architecture (x86_64, aarch64, wasm, generic)

**Example**:
```zig
const a = [_]f32{ 1, 2, 3, 4 };
const b = [_]f32{ 5, 6, 7, 8 };
var result: [4]f32 = undefined;

abi.simd.vectorAdd(&a, &b, &result);
// result = { 6, 8, 10, 12 }
```

## Benchmark Framework

- `BenchmarkRunner.init(allocator)` - Create runner
- `BenchmarkRunner.run(config, fn, args)` -> `BenchResult` - Run benchmark
- `BenchmarkRunner.exportJson()` - Export results to JSON
- `BenchmarkRunner.printSummaryDebug()` - Print summary
- `compareWithBaseline(allocator, current, baseline, threshold)` -> `[]RegressionResult` - Detect regressions
- `printRegressionSummary(results)` - Print regression analysis

**RegressionResult**:
- `.is_regression` - Performance degraded beyond threshold
- `.is_improvement` - Performance improved beyond threshold
- `.change_percent` - Percentage change from baseline

## Modules

- `src/core` - I/O, diagnostics, collections
- `src/features` - feature modules (AI, GPU, database, web, monitoring, connectors)
- `src/framework` - orchestration runtime and lifecycle management
- `src/shared` - shared utilities and platform helpers
- `src/compute` - compute runtime, memory management, concurrency

**See Also**:
- [Introduction](docs/intro.md) - Architecture overview
- [Framework Guide](docs/framework.md) - Configuration and lifecycle
- [Compute Guide](docs/compute.md) - Task execution
- [AI Guide](docs/ai.md) - LLM connectors and agents
- [GPU Guide](docs/gpu.md) - GPU backends

## Contacts

src/shared/contacts.zig provides a centralized list of maintainer contacts extracted from the repository markdown files. Import this module wherever contact information is needed.
See [TODO.md](TODO.md) for list of pending implementations.
See [TODO.md](TODO.md) and [ROADMAP.md](ROADMAP.md) for the Llama‑CPP parity task list and upcoming milestones.
*See [TODO.md](TODO.md) and [ROADMAP.md](ROADMAP.md) for the Llama‑CPP parity task list and upcoming milestones.*
