# API Reference

<p align="center">
  <img src="https://img.shields.io/badge/API-Stable-success?style=for-the-badge" alt="API Stable"/>
  <img src="https://img.shields.io/badge/Version-0.3.0-blue?style=for-the-badge" alt="Version"/>
  <img src="https://img.shields.io/badge/Zig-0.16-F7A41D?style=for-the-badge&logo=zig&logoColor=white" alt="Zig"/>
</p>

<p align="center">
  <a href="docs/intro.md">Documentation Index</a> •
  <a href="CONTRIBUTING.md">Coding Patterns</a> •
  <a href="CLAUDE.md">Development Guide</a>
</p>

---

> **Summary**: This is a high-level summary of the public ABI API surface. See the source for implementation details.

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

| Namespace | Description | Status |
|-----------|-------------|--------|
| `abi.ai` | AI module with sub-features | ![Stable](https://img.shields.io/badge/-Stable-success) |
| `abi.ai.llm` | Local LLM inference | ![Stable](https://img.shields.io/badge/-Stable-success) |
| `abi.ai.embeddings` | Vector embeddings | ![Stable](https://img.shields.io/badge/-Stable-success) |
| `abi.ai.agents` | AI agent runtime | ![Stable](https://img.shields.io/badge/-Stable-success) |
| `abi.ai.training` | Training pipelines | ![Stable](https://img.shields.io/badge/-Stable-success) |
| `abi.gpu` | GPU backends and unified API | ![Stable](https://img.shields.io/badge/-Stable-success) |
| `abi.gpu.device` | Device enumeration and selection | ![Stable](https://img.shields.io/badge/-Stable-success) |
| `abi.gpu.backend_factory` | Backend auto-detection | ![Stable](https://img.shields.io/badge/-Stable-success) |
| `abi.gpu.execution_coordinator` | GPU→SIMD→scalar fallback | ![Stable](https://img.shields.io/badge/-Stable-success) |
| `abi.database` | WDBX vector database | ![Stable](https://img.shields.io/badge/-Stable-success) |
| `abi.network` | Distributed compute and Raft | ![Stable](https://img.shields.io/badge/-Stable-success) |
| `abi.web` | HTTP helpers, web utilities | ![Stable](https://img.shields.io/badge/-Stable-success) |
| `abi.observability` | Metrics, tracing, profiling | ![Stable](https://img.shields.io/badge/-Stable-success) |
| `abi.connectors` | External connectors | ![Stable](https://img.shields.io/badge/-Stable-success) |

> **Note:** `abi.monitoring` is deprecated; use `abi.observability` instead.

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

## Runtime Engine API

- `abi.runtime.DistributedComputeEngine` - Main runtime engine for task execution
- `abi.runtime.createEngine(allocator, config)` -> `!Engine` - Create engine with config
- `abi.runtime.submitTask(engine, ResultType, task)` -> `!TaskId` - Submit a task for execution
- `abi.runtime.waitForResult(engine, ResultType, id, timeout_ms)` -> `!Result` - Wait for task result
- `abi.runtime.runTask(engine, ResultType, task, timeout_ms)` -> `!Result` - Submit and wait for result
- `abi.runtime.runWorkload(engine, ResultType, workload, timeout_ms)` -> `!Result` - Alias for runTask

**Example**:
```zig
var engine = try abi.runtime.createEngine(allocator, .{});
defer engine.deinit();

fn computeTask(_: std.mem.Allocator) !u32 {
    return 42;
}

// Submit and wait in one call
const result = try abi.runtime.runTask(&engine, u32, computeTask, 1000);
std.debug.print("Result: {d}\n", .{result});
```

See [Runtime Guide](docs/compute.md) for detailed usage.

**Timeout Semantics**:

- `timeout_ms=0`: Immediately returns `EngineError.Timeout` if result not ready
- `timeout_ms>0`: Waits for the specified timeout (in milliseconds) before returning `EngineError.Timeout`
- `timeout_ms=null`: Waits indefinitely until result is ready

**Breaking Change (0.2.1)**: Prior to version 0.2.1, `timeout_ms=0` returned `ResultNotFound` after one check. This behavior has changed to return `EngineError.Timeout` immediately for clarity. Migration: Use `timeout_ms=1000` for a one-second timeout.

## GPU API

### Device Enumeration

- `abi.gpu.device.enumerateAllDevices(allocator)` -> `![]Device` - Enumerate all GPU devices
- `abi.gpu.device.enumerateDevicesForBackend(allocator, backend)` -> `![]Device` - Per-backend enumeration
- `abi.gpu.device.selectBestDevice(allocator, criteria)` -> `!?Device` - Select device by criteria
- `abi.gpu.device.DeviceSelectionCriteria` - Selection criteria struct
  - `.prefer_discrete: bool` - Prefer discrete GPUs
  - `.min_memory_gb: u64` - Minimum memory requirement
  - `.required_features: []const DeviceFeature` - Required capabilities

### Backend Auto-Detection

- `abi.gpu.backend_factory.detectAvailableBackends(allocator)` -> `![]Backend` - List available backends
- `abi.gpu.backend_factory.selectBestBackendWithFallback(allocator, options)` -> `!?Backend` - Select with fallback
- `abi.gpu.backend_factory.selectBackendWithFeatures(allocator, options)` -> `!?Backend` - Feature-based selection
- `abi.gpu.backend_factory.SelectionOptions` - Backend selection options
  - `.preferred: ?Backend` - Preferred backend
  - `.fallback_chain: []const Backend` - Fallback order
  - `.required_features: []const BackendFeature` - Required features
  - `.fallback_to_cpu: bool` - Allow CPU fallback

### Execution Coordinator

- `abi.gpu.ExecutionCoordinator.init(allocator, config)` -> `!ExecutionCoordinator` - Initialize coordinator
- `abi.gpu.ExecutionCoordinator.deinit()` - Clean up resources
- `abi.gpu.ExecutionCoordinator.vectorAdd(a, b, result)` -> `!ExecutionMethod` - Auto-select method
- `abi.gpu.ExecutionCoordinator.vectorAddWithMethod(a, b, result, method)` -> `!ExecutionMethod` - Force method
- `abi.gpu.ExecutionMethod` - Execution method enum: `.gpu`, `.simd`, `.scalar`, `.failed`

**Example**:
```zig
const device = abi.gpu.device;
const exec = abi.gpu.execution_coordinator;

// Enumerate devices
const devices = try device.enumerateAllDevices(allocator);
defer allocator.free(devices);

// Select best GPU with 4GB+ memory
const best = try device.selectBestDevice(allocator, .{
    .prefer_discrete = true,
    .min_memory_gb = 4,
});

// Use execution coordinator for automatic fallback
var coordinator = try exec.ExecutionCoordinator.init(allocator, .{});
defer coordinator.deinit();

var result = [_]f32{0} ** 8;
const method = try coordinator.vectorAdd(&input_a, &input_b, &result);
// method is .gpu, .simd, or .scalar depending on availability
```

See [GPU Guide](docs/gpu.md) for detailed usage.

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

Flat domain structure (modular architecture):

| Module | Description | Status |
|--------|-------------|--------|
| `src/abi.zig` | Public API entry point | ![Core](https://img.shields.io/badge/-Core-blue) |
| `src/config.zig` | Unified configuration system | ![Core](https://img.shields.io/badge/-Core-blue) |
| `src/framework.zig` | Framework orchestration | ![Core](https://img.shields.io/badge/-Core-blue) |
| `src/runtime/` | Scheduler, memory, concurrency | ![Core](https://img.shields.io/badge/-Core-blue) |
| `src/gpu/` | GPU backends and unified API | ![Feature](https://img.shields.io/badge/-Feature-green) |
| `src/ai/` | AI module (llm, embeddings, agents, training) | ![Feature](https://img.shields.io/badge/-Feature-green) |
| `src/database/` | WDBX vector database | ![Feature](https://img.shields.io/badge/-Feature-green) |
| `src/network/` | Distributed compute and Raft | ![Feature](https://img.shields.io/badge/-Feature-green) |
| `src/observability/` | Metrics, tracing, profiling | ![Feature](https://img.shields.io/badge/-Feature-green) |
| `src/web/` | HTTP helpers and web utilities | ![Feature](https://img.shields.io/badge/-Feature-green) |
| `src/shared/` | SIMD, platform helpers | ![Shared](https://img.shields.io/badge/-Shared-yellow) |

> **Backward Compatibility**: Re-exports in `abi.zig` maintain API compatibility with the previous `features/` and `compute/` structure.

## See Also

<table>
<tr>
<td>

### Guides
- [Introduction](docs/intro.md) — Architecture overview
- [Framework Guide](docs/framework.md) — Configuration and lifecycle
- [Compute Guide](docs/compute.md) — Task execution
- [AI Guide](docs/ai.md) — LLM connectors and agents
- [GPU Guide](docs/gpu.md) — GPU backends

</td>
<td>

### Project
- [TODO.md](TODO.md) — Pending implementations
- [ROADMAP.md](ROADMAP.md) — Upcoming milestones
- [CONTRIBUTING.md](CONTRIBUTING.md) — Development guidelines
- [CHANGELOG.md](CHANGELOG.md) — Version history

</td>
</tr>
</table>

---

<p align="center">
  <a href="README.md">← Back to README</a> •
  <a href="docs/intro.md">Full Documentation →</a>
</p>
