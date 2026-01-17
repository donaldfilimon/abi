# API Reference (Concise)

> For detailed usage guides, see [Documentation Index](docs/intro.md).
> For coding patterns and conventions, see [CONTRIBUTING.md](CONTRIBUTING.md).
> For comprehensive development guidance, see [CLAUDE.md](CLAUDE.md).

This is a high-level summary of the public ABI API surface. See the source for
implementation details.

## Core Entry Points

- `abi.init(allocator, config_or_options)` -> `Framework` (backward-compatible)
- `abi.shutdown(framework)` (backward-compatible)
- `abi.version()` -> `[]const u8`
- `abi.Framework.init(allocator, config)` -> `!Framework` (new unified API)
- `abi.Framework.deinit()` (new unified API)

**New Configuration System** (recommended):
```zig
const std = @import("std");
const abi = @import("abi");

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;

    // Unified Config with builder pattern
    const config = abi.Config.init()
        .withAI(true)
        .withGPU(true)
        .withDatabase(true)
        .withNetwork(false);

    var framework = try abi.Framework.init(allocator, config);
    defer framework.deinit();

    std.debug.print("ABI v{s} initialized\n", .{abi.version()});

    // Access feature modules through the framework
    if (framework.ai()) |ai| {
        _ = ai; // Use AI features
    }
    if (framework.gpu()) |gpu| {
        _ = gpu; // Use GPU features
    }
}
```

**Backward-compatible Example**:
```zig
var framework = try abi.init(allocator, .{});
defer abi.shutdown(&framework);
```

## Configuration Types

- `abi.Config` - Unified configuration with builder pattern
  - `.init()` -> `Config` - Create default configuration
  - `.withAI(bool)` -> `Config` - Enable/disable AI features
  - `.withGPU(bool)` -> `Config` - Enable/disable GPU acceleration
  - `.withDatabase(bool)` -> `Config` - Enable/disable vector database
  - `.withNetwork(bool)` -> `Config` - Enable/disable distributed compute
  - `.withObservability(bool)` -> `Config` - Enable/disable metrics/tracing
  - `.withWeb(bool)` -> `Config` - Enable/disable web utilities

## Framework Types

- `abi.Framework` - Main orchestration struct managing feature lifecycles
  - `.init(allocator, config)` -> `!Framework`
  - `.deinit()` - Clean up all resources
  - `.ai()` -> `?*AI` - Access AI module (if enabled)
  - `.gpu()` -> `?*GPU` - Access GPU module (if enabled)
  - `.database()` -> `?*Database` - Access database module (if enabled)
  - `.network()` -> `?*Network` - Access network module (if enabled)
  - `.observability()` -> `?*Observability` - Access observability module
- `abi.FrameworkOptions` (deprecated, use `abi.Config`)
- `abi.RuntimeConfig`
- `abi.Feature` and `abi.features.FeatureTag`

## Feature Namespaces

Top-level domain modules (flat structure):

- `abi.ai` - AI module with sub-features
  - `abi.ai.llm` - Local LLM inference
  - `abi.ai.embeddings` - Vector embeddings
  - `abi.ai.agents` - AI agent runtime
  - `abi.ai.training` - Training pipelines
- `abi.gpu` - GPU backends and unified API
- `abi.database` - WDBX vector database
- `abi.network` - Distributed compute and Raft consensus
- `abi.web` - HTTP helpers, web utilities
- `abi.observability` - Metrics, tracing, profiling (replaces `abi.monitoring`)
- `abi.connectors` - External connectors (OpenAI, Ollama, HuggingFace)

**Note:** `abi.monitoring` is deprecated; use `abi.observability` instead.

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

Flat domain structure (new modular architecture):

- `src/abi.zig` - Public API entry point
- `src/config.zig` - Unified configuration system
- `src/framework.zig` - Framework orchestration and lifecycle management
- `src/runtime/` - Always-on infrastructure (scheduler, memory, concurrency)
- `src/gpu/` - GPU backends and unified API
- `src/ai/` - AI module with sub-features (llm, embeddings, agents, training)
- `src/database/` - WDBX vector database
- `src/network/` - Distributed compute and Raft consensus
- `src/observability/` - Metrics, tracing, profiling
- `src/web/` - HTTP helpers and web utilities
- `src/shared/` - Shared utilities (simd, observability primitives, platform helpers)

**Backward Compatibility**: Re-exports in `abi.zig` maintain API compatibility with the previous `features/` and `compute/` structure.

**See Also**:
- [Introduction](docs/intro.md) - Architecture overview
- [Framework Guide](docs/framework.md) - Configuration and lifecycle
- [Compute Guide](docs/compute.md) - Task execution
- [AI Guide](docs/ai.md) - LLM connectors and agents
- [GPU Guide](docs/gpu.md) - GPU backends

## See Also

- [TODO.md](TODO.md) - Pending implementations
- [ROADMAP.md](ROADMAP.md) - Upcoming milestones
