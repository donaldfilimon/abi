---
title: "API_REFERENCE"
tags: []
---
# API Reference

> **Codebase Status:** Synced with repository as of 2026-02-14.

<p align="center">
  <img src="https://img.shields.io/badge/API-Stable-success?style=for-the-badge" alt="API Stable"/>
  <img src="https://img.shields.io/badge/Version-0.4.0-blue?style=for-the-badge" alt="Version"/>
  <img src="https://img.shields.io/badge/Zig-0.16.0--dev.2535%2Bb5bd49460-F7A41D?style=for-the-badge&logo=zig&logoColor=white" alt="Zig"/>
</p>

<p align="center">
  <a href="README.md">Documentation</a> ·
  <a href="../CONTRIBUTING.md">Contributing</a> ·
  <a href="../CLAUDE.md">Development Guide</a>
</p>

---

> **Summary**: This is the comprehensive API reference for the ABI framework. Each section includes function signatures, parameters, return values, and usage examples.

## Table of Contents

- [Core Entry Points](#core-entry-points)
- [Configuration](#configuration-types)
- [Framework](#framework-types)
- [AI Module](#ai--agent-api)
- [GPU Module](#gpu-api)
- [Database Module](#wdbx-convenience-api)
- [Network Module](#network-api)
- [SIMD Operations](#simd-api)

## Core Entry Points

The main entry point for ABI is through the `abi` namespace. Import it with:

```zig
const abi = @import("abi");
```

### Initialization Functions

| Function | Description |
|----------|-------------|
| `abi.initDefault(allocator)` | Initialize with all default settings |
| `abi.init(allocator, config)` | Initialize with custom configuration |
| `abi.version()` | Get the framework version string |
| `framework.deinit()` | Clean up and release resources |

### Quick Start Example

```zig
const std = @import("std");
const abi = @import("abi");

pub fn main() !void {
    // Set up allocator
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Initialize with defaults - simplest approach
    var fw = try abi.initDefault(allocator);
    defer fw.deinit();

    // Print version
    std.debug.print("ABI Framework v{s}\n", .{abi.version()});

    // Check which features are enabled
    if (fw.isEnabled(.gpu)) {
        std.debug.print("GPU acceleration available\n", .{});
    }
    if (fw.isEnabled(.ai)) {
        std.debug.print("AI features available\n", .{});
    }
}
```

### Custom Configuration Example

```zig
const std = @import("std");
const abi = @import("abi");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Initialize with specific features enabled
    var fw = try abi.init(allocator, .{
        .gpu = .{ .backend = .vulkan },
        .ai = .{
            .llm = .{ .model_path = "./models/llama-7b.gguf" },
            .embeddings = .{ .dimension = 768 },
        },
        .database = .{ .path = "./vectors.db" },
        // network and web are null (disabled)
    });
    defer fw.deinit();

    // Access the AI context
    if (fw.ai) |ai_ctx| {
        const llm = try ai_ctx.getLlm();
        _ = llm; // Use LLM for inference
    }
}
```

### Builder Pattern Example

```zig
var fw = try abi.Framework.builder(allocator)
    .withGpu(.{ .backend = .cuda })
    .withAi(.{
        .llm = .{ .model_path = "./model.gguf" },
        .personas = .{ .default_persona = .abbey, .enable_dynamic_routing = true },
    })
    .withDatabaseDefaults()
    .withObservabilityDefaults()
    .build();
defer fw.deinit();
```

## Configuration Types

### Config Struct

The `abi.Config` struct is the unified configuration for all framework features. Each field being non-null enables that feature.

```zig
pub const Config = struct {
    gpu: ?GpuConfig = null,           // GPU acceleration
    ai: ?AiConfig = null,             // AI features (LLM, embeddings, agents, etc.)
    database: ?DatabaseConfig = null,  // Vector database
    network: ?NetworkConfig = null,    // Distributed networking
    observability: ?ObservabilityConfig = null,  // Metrics/tracing
    web: ?WebConfig = null,           // HTTP utilities
    cloud: ?CloudConfig = null,       // Cloud provider integration
    analytics: ?AnalyticsConfig = null, // Analytics event tracking
    plugins: PluginConfig = .{},      // Plugin system
};
```

### Config Methods

| Method | Description |
|--------|-------------|
| `Config.defaults()` | Create config with all compile-time enabled features |
| `Config.minimal()` | Create config with no features enabled |
| `Config.isEnabled(feature)` | Check if a specific feature is enabled |
| `Config.enabledFeatures(allocator)` | Get list of all enabled features |

### Builder Pattern

```zig
var builder = abi.config.Builder.init(allocator);
const config = builder
    .withDefaults()           // Start with all defaults
    .withGpu(.{ .backend = .vulkan })
    .withAi(.{ .llm = .{} })
    .withDatabaseDefaults()
    .withNetworkDefaults()
    .withObservabilityDefaults()
    .withWebDefaults()
    .withCloudDefaults()
    .withAnalyticsDefaults()
    .build();
```

### Feature Enum

```zig
pub const Feature = enum {
    gpu,          // GPU acceleration
    ai,           // AI core (parent of sub-features below)
    llm,          // Local LLM inference
    embeddings,   // Vector embeddings
    agents,       // AI agent runtime
    training,     // Model training
    database,     // Vector database
    network,      // Distributed compute
    observability,// Metrics and tracing
    web,          // HTTP utilities
    personas,     // Multi-persona AI assistant
    cloud,        // Cloud provider integration
    analytics,    // Analytics event tracking
};

// Check compile-time availability
if (abi.Feature.gpu.isCompileTimeEnabled()) {
    // GPU code was compiled in
}

// Get feature info
const name = abi.Feature.ai.name();              // "ai"
const desc = abi.Feature.ai.description();       // "AI core functionality"
```

> **Note:** `cloud` has its own `-Denable-cloud` build flag (decoupled from web). `analytics` uses `-Denable-analytics`.

## Framework Types

### Framework Struct

The `abi.Framework` struct is the central coordinator for all ABI functionality.

```zig
pub const Framework = struct {
    allocator: std.mem.Allocator,
    io: ?std.Io,               // Optional I/O backend
    config: Config,
    state: State,
    registry: Registry,

    // Feature contexts (null if not enabled)
    gpu: ?*gpu.Context,
    ai: ?*ai.Context,
    database: ?*database.Context,
    network: ?*network.Context,
    observability: ?*observability.Context,
    web: ?*web.Context,
    cloud: ?*cloud.Context,
    analytics: ?*analytics.Context,
    ha: ?ha.HaManager,
    runtime: *runtime.Context,  // Always available

    pub const State = enum {
        uninitialized,
        initializing,
        running,
        stopping,
        stopped,
        failed,
    };
};
```

### Framework Methods

| Method | Description |
|--------|-------------|
| `Framework.init(allocator, config)` | Initialize with configuration |
| `Framework.initDefault(allocator)` | Initialize with default config |
| `Framework.initMinimal(allocator)` | Initialize with no features |
| `Framework.builder(allocator)` | Create a FrameworkBuilder |
| `deinit()` | Clean up all resources |
| `isRunning()` | Check if framework is in running state |
| `isEnabled(feature)` | Check if a feature is enabled |
| `getState()` | Get current lifecycle state |
| `getGpu()` | Get GPU context (error if not enabled) |
| `getAi()` | Get AI context (error if not enabled) |
| `getDatabase()` | Get database context (error if not enabled) |
| `getNetwork()` | Get network context (error if not enabled) |
| `getObservability()` | Get observability context (error if not enabled) |
| `getWeb()` | Get web context (error if not enabled) |
| `getCloud()` | Get cloud context (error if not enabled) |
| `getAnalytics()` | Get analytics context (error if not enabled) |
| `getRuntime()` | Get runtime context (always available) |
| `getRegistry()` | Get feature registry |

### Framework Example

```zig
var fw = try abi.Framework.init(allocator, .{
    .gpu = .{ .backend = .vulkan },
    .ai = .{ .llm = .{ .model_path = "./model.gguf" } },
});
defer fw.deinit();

// Check state
if (fw.isRunning()) {
    // Access features safely
    const gpu_ctx = try fw.getGpu();
    const ai_ctx = try fw.getAi();

    // Use the contexts...
    _ = gpu_ctx;
    _ = ai_ctx;
}

// Runtime is always available
const runtime = fw.getRuntime();
_ = runtime;
```

### FrameworkBuilder

```zig
var builder = abi.Framework.builder(allocator);
var fw = try builder
    .withDefaults()
    .withGpu(.{ .backend = .cuda })
    .withAi(.{ .llm = .{}, .personas = .{} })
    .withDatabaseDefaults()
    .withIo(io)  // Optional: provide I/O backend
    .build();
defer fw.deinit();
```

## Feature Namespaces

Top-level domain modules (flat structure):

| Namespace | Description | Status |
|-----------|-------------|--------|
| `abi.ai` | AI module with sub-features | ![Stable](https://img.shields.io/badge/-Stable-success) |
| `abi.ai.llm` | Local LLM inference | ![Stable](https://img.shields.io/badge/-Stable-success) |
| `abi.ai.embeddings` | Vector embeddings | ![Stable](https://img.shields.io/badge/-Stable-success) |
| `abi.ai.agents` | AI agent runtime | ![Stable](https://img.shields.io/badge/-Stable-success) |
| `abi.ai.training` | Training pipelines | ![Stable](https://img.shields.io/badge/-Stable-success) |
| `abi.ai.personas` | Multi-persona AI assistant | ![Stable](https://img.shields.io/badge/-Stable-success) |
| `abi.gpu` | GPU backends and unified API | ![Stable](https://img.shields.io/badge/-Stable-success) |
| `abi.gpu.device` | Device enumeration and selection | ![Stable](https://img.shields.io/badge/-Stable-success) |
| `abi.gpu.backend_factory` | Backend auto-detection | ![Stable](https://img.shields.io/badge/-Stable-success) |
| `abi.gpu.execution_coordinator` | GPU→SIMD→scalar fallback | ![Stable](https://img.shields.io/badge/-Stable-success) |
| `abi.database` | WDBX vector database | ![Stable](https://img.shields.io/badge/-Stable-success) |
| `abi.network` | Distributed compute and Raft | ![Stable](https://img.shields.io/badge/-Stable-success) |
| `abi.web` | HTTP helpers, web utilities | ![Stable](https://img.shields.io/badge/-Stable-success) |
| `abi.observability` | Metrics, tracing, profiling | ![Stable](https://img.shields.io/badge/-Stable-success) |
| `abi.cloud` | Cloud function adapters | ![Stable](https://img.shields.io/badge/-Stable-success) |
| `abi.analytics` | Event tracking and experiments | ![Stable](https://img.shields.io/badge/-Stable-success) |
| `abi.connectors` | External connectors | ![Stable](https://img.shields.io/badge/-Stable-success) |

> **Note:** `abi.monitoring` is deprecated; use `abi.observability` instead.

## Database API (WDBX)

The database module provides a high-performance vector database with HNSW indexing.

### Core Functions

| Function | Description |
|----------|-------------|
| `abi.database.open(allocator, path)` | Open or create a database |
| `abi.database.connect(allocator, path)` | Connect to existing database |
| `abi.database.close(handle)` | Close database and release resources |
| `abi.database.insert(handle, id, vector, metadata)` | Insert a vector |
| `abi.database.search(handle, allocator, query, top_k)` | Search for similar vectors |
| `abi.database.remove(handle, id)` | Delete a vector by ID |
| `abi.database.update(handle, id, vector)` | Update an existing vector |
| `abi.database.get(handle, id)` | Get a specific vector by ID |
| `abi.database.list(handle, allocator, limit)` | List vectors |
| `abi.database.stats(handle)` | Get database statistics |
| `abi.database.optimize(handle)` | Optimize the index |
| `abi.database.backup(handle, path)` | Create a backup |
| `abi.database.restore(handle, path)` | Restore from backup |

### Database Example

```zig
const db = abi.database;

// Open database
var handle = try db.open(allocator, "vectors.db");
defer db.close(&handle);

// Insert vectors with metadata
try db.insert(&handle, 1, &[_]f32{ 0.1, 0.2, 0.3, 0.4 }, "document_1");
try db.insert(&handle, 2, &[_]f32{ 0.5, 0.6, 0.7, 0.8 }, "document_2");
try db.insert(&handle, 3, &[_]f32{ 0.2, 0.3, 0.4, 0.5 }, "document_3");

// Search for similar vectors
const query = [_]f32{ 0.15, 0.25, 0.35, 0.45 };
const results = try db.search(&handle, allocator, &query, 5);
defer allocator.free(results);

for (results) |result| {
    std.debug.print("ID: {}, Score: {d:.4}\n", .{ result.id, result.score });
}

// Get statistics
const stats = db.stats(&handle);
std.debug.print("Total vectors: {}\n", .{stats.total_vectors});
```

### Context-based API

When using the Framework, access the database through the Context:

```zig
var fw = try abi.Framework.init(allocator, .{
    .database = .{ .path = "./vectors.db" },
});
defer fw.deinit();

const db_ctx = try fw.getDatabase();

// Insert and search through context
try db_ctx.insertVector(1, &vector, "metadata");
const results = try db_ctx.searchVectors(&query, 10);
```

### Advanced Features

#### Hybrid Search (Vector + Text)

```zig
var engine = try db.HybridSearchEngine.init(allocator, .{
    .vector_weight = 0.7,
    .text_weight = 0.3,
    .fusion = .rrf,  // Reciprocal Rank Fusion
});
defer engine.deinit();

const results = try engine.search(query_vector, "search text", 10);
```

#### Metadata Filtering

```zig
var filter = db.FilterBuilder.init()
    .eq("category", .{ .string = "science" })
    .gte("year", .{ .int = 2020 })
    .build();

var filtered = try db.FilteredSearch.init(allocator, &handle, filter);
defer filtered.deinit();

const results = try filtered.search(&query, 10);
```

#### Batch Operations

```zig
var batch = try db.BatchProcessor.init(allocator, .{
    .batch_size = 1000,
    .parallel = true,
});
defer batch.deinit();

for (vectors) |v| {
    try batch.add(v.id, v.data, v.metadata);
}
try batch.flush(&handle);
```

### Security Note for backup/restore

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

See the [Architecture Guide](docs/content/architecture.html) for detailed usage.

**Timeout Semantics**:

- `timeout_ms=0`: Immediately returns `EngineError.Timeout` if result not ready
- `timeout_ms>0`: Waits for the specified timeout (in milliseconds) before returning `EngineError.Timeout`
- `timeout_ms=null`: Waits indefinitely until result is ready

**Breaking Change (0.2.1)**: Prior to version 0.2.1, `timeout_ms=0` returned `ResultNotFound` after one check. This behavior has changed to return `EngineError.Timeout` immediately for clarity. Migration: Use `timeout_ms=1000` for a one-second timeout.

## GPU API

### Supported Backends

| Backend | Platform | Build Flag | Description |
|---------|----------|------------|-------------|
| `auto` | All | `-Dgpu-backend=auto` | Auto-detect best available |
| `metal` | macOS | `-Dgpu-backend=metal` | Apple Metal + Accelerate |
| `cuda` | Linux/Windows | `-Dgpu-backend=cuda` | NVIDIA CUDA |
| `vulkan` | All | `-Dgpu-backend=vulkan` | Vulkan compute |
| `stdgpu` | All | `-Dgpu-backend=stdgpu` | Software fallback |
| `webgpu` | WASM | `-Dgpu-backend=webgpu` | WebGPU for browsers |
| `opengl` | All | `-Dgpu-backend=opengl` | OpenGL compute |
| `fpga` | Specialized | `-Dgpu-backend=fpga` | FPGA acceleration |
| `none` | All | `-Dgpu-backend=none` | Disable GPU |

Multiple backends: `-Dgpu-backend=cuda,vulkan`

### Device Enumeration

```zig
const device = abi.gpu.device;

// Enumerate all devices
const devices = try device.enumerateAllDevices(allocator);
defer allocator.free(devices);

// Select best GPU with 4GB+ memory
const best = try device.selectBestDevice(allocator, .{
    .prefer_discrete = true,
    .min_memory_gb = 4,
});
```

### Backend Factory

```zig
const factory = abi.gpu.backend_factory;

// Detect available backends
const backends = try factory.detectAvailableBackends(allocator);

// Select with fallback chain
const backend = try factory.selectBestBackendWithFallback(allocator, .{
    .preferred = .cuda,
    .fallback_chain = &.{ .vulkan, .metal, .stdgpu },
    .fallback_to_cpu = true,
});
```

### Execution Coordinator (Auto-Fallback)

```zig
var coordinator = try abi.gpu.ExecutionCoordinator.init(allocator, .{});
defer coordinator.deinit();

var result: [8]f32 = undefined;
const method = try coordinator.vectorAdd(&input_a, &input_b, &result);
// method is .gpu, .simd, or .scalar depending on availability
```

### Metal Backend (macOS)

Apple Silicon optimizations with Accelerate framework and unified memory:

```zig
const metal = abi.gpu.backends.metal;

// Accelerate framework (AMX-accelerated)
if (metal.hasAccelerate()) {
    const acc = metal.accelerate;
    acc.vblas.sgemm(M, N, K, 1.0, a_ptr, lda, b_ptr, ldb, 0.0, c_ptr, ldc);
    acc.vdsp.vadd(a_ptr, b_ptr, c_ptr, n);
    acc.neural.softmax(input, output, size);
}

// Unified memory (zero-copy CPU/GPU sharing)
if (metal.hasUnifiedMemory()) {
    var umm = try metal.unified_memory.UnifiedMemoryManager.init(allocator);
    defer umm.deinit();

    var tensor = try umm.allocateTensor(f32, &.{1024, 768}, .shared);
    defer umm.freeTensor(&tensor);

    try umm.synchronize(&tensor, .cpu_to_gpu);
}
```

### CUDA Backend

```zig
const cuda = abi.gpu.backends.cuda;

// Initialize with allocator
try cuda.init(allocator);
defer cuda.deinit();

// Memory operations
const device_mem = try cuda.allocateDeviceMemory(size);
defer cuda.freeDeviceMemory(device_mem);

try cuda.memcpyHostToDevice(device_mem, host_ptr, size);
try cuda.memcpyDeviceToHost(host_ptr, device_mem, size);

// Kernel execution
const kernel = try cuda.compileKernel(allocator, source);
defer cuda.destroyKernel(allocator, kernel);
try cuda.launchKernel(allocator, kernel, config, args);
```

See the [GPU Guide](docs/api/gpu.md) for detailed usage.

## Network API

The network module provides distributed compute capabilities with Raft consensus.

### Core Types

| Type | Description |
|------|-------------|
| `abi.network.NetworkConfig` | Network configuration |
| `abi.network.NetworkState` | Current network state |
| `abi.network.Context` | Network context for framework integration |

### Network Example

```zig
var fw = try abi.Framework.init(allocator, .{
    .network = .{
        .node_id = "node-1",
        .listen_address = "0.0.0.0:8080",
        .peers = &[_][]const u8{ "node-2:8080", "node-3:8080" },
    },
});
defer fw.deinit();

const net_ctx = try fw.getNetwork();
_ = net_ctx;

// Network features available through context
```

See the [Network Guide](docs/api/network.md) for detailed usage.

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

See the [AI Guide](docs/api/ai.md) for detailed usage.

## Streaming API

Real-time token streaming for LLM inference:

### Server

```bash
# Start streaming server
zig build run -- llm serve -m ./model.gguf --preload

# With authentication
zig build run -- llm serve -m ./model.gguf -a 0.0.0.0:8000 --auth-token my-secret
```

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/v1/chat/completions` | OpenAI-compatible chat completions (SSE) |
| POST | `/api/stream` | Custom ABI streaming (SSE) |
| WS | `/api/stream/ws` | WebSocket with cancellation |
| GET | `/v1/models` | OpenAI-compatible model listing |
| GET | `/metrics` | Server metrics |
| POST | `/admin/reload` | Hot-reload model |
| GET | `/health` | Health check |

> **Note:** `/v1/chat/completions` and `/v1/models` require `enable_openai_compat` in server config. `/api/stream/ws` requires `enable_websocket`. `/health` and `/metrics` can optionally bypass authentication via `allow_health_without_auth`.

### Client Example

```zig
const streaming = abi.ai.streaming;

var client = try streaming.Client.init(allocator, .{
    .endpoint = "http://localhost:8080",
    .auth_token = "my-secret",
});
defer client.deinit();

// Stream completion
var stream = try client.streamCompletion(.{
    .prompt = "Explain quantum computing",
    .max_tokens = 500,
});
defer stream.deinit();

while (try stream.next()) |token| {
    std.debug.print("{s}", .{token});
}
```

### Circuit Breaker (Error Recovery)

```zig
const recovery = abi.ai.streaming.recovery;

var breaker = recovery.CircuitBreaker.init(.{
    .failure_threshold = 5,
    .reset_timeout_ms = 30000,
    .half_open_requests = 3,
});

// Automatic retry with exponential backoff
const result = try recovery.withRetry(allocator, operation, .{
    .max_retries = 3,
    .initial_delay_ms = 100,
    .max_delay_ms = 5000,
    .backoff_multiplier = 2.0,
});
```

## Personas API

The Multi-Persona AI Assistant routes queries through specialized personas.

### Personas

| Persona | Type | Description |
|---------|------|-------------|
| **Abi** | Router/Moderator | Content moderation, sentiment analysis, routing |
| **Abbey** | Empathetic Polymath | Supportive responses with emotional intelligence |
| **Aviva** | Direct Expert | Concise, factual, technically accurate responses |
| **Assistant** | General | General-purpose helpful assistant |
| **Coder** | Specialist | Programming and code-focused assistant |
| **Writer** | Specialist | Creative writing and content generation |
| **Analyst** | Specialist | Data analysis and research |
| **Companion** | Conversational | Friendly, supportive chat |
| **Docs** | Specialist | Technical documentation helper |
| **Reviewer** | Specialist | Code and logic reviewer |
| **Minimal** | Specialist | Minimal, direct responses |
| **Ralph** | Specialist | Iterative agent loop specialist |

### Core Types

- `abi.ai.personas.MultiPersonaSystem` - Main orchestrator managing persona routing
- `abi.ai.personas.PersonaType` - Enum of all persona types (Abbey/Aviva/Abi plus generic modes)
- `abi.ai.personas.PersonaRequest` - Request structure with content, optional forced persona, and context
- `abi.ai.personas.PersonaResponse` - Response with content, persona, confidence, and optional metadata
- `abi.ai.personas.RoutingScore` - Simple score tuple `{ persona_type, score }` for load balancing

### Orchestrator API

- `MultiPersonaSystem.init(allocator, config)` -> `!*MultiPersonaSystem` - Initialize base system
- `MultiPersonaSystem.initWithDefaults(allocator, config)` -> `!*MultiPersonaSystem` - Initialize with Abbey/Aviva/Abi plus metrics/load balancer
- `MultiPersonaSystem.deinit()` - Clean up resources
- `MultiPersonaSystem.process(request)` -> `!PersonaResponse` - Auto-route message
- `MultiPersonaSystem.processWithPersona(type, request)` -> `!PersonaResponse` - Force persona
- `MultiPersonaSystem.registerPersona(type, interface)` -> `!void` - Register a persona implementation
- `MultiPersonaSystem.getPersona(type)` -> `?PersonaInterface` - Access specific persona
- `MultiPersonaSystem.enableMetrics(collector?)` -> `!void` - Enable metrics (creates owned collector if null)
- `MultiPersonaSystem.enableLoadBalancer(config)` -> `!void` - Enable load balancing
- `MultiPersonaSystem.enableHealthChecks(config)` -> `!void` - Enable health checks
- `MultiPersonaSystem.getMetrics()` -> `?*PersonaMetrics` - Access metrics if enabled
- `MultiPersonaSystem.getHealthChecker()` -> `?*HealthChecker` - Access health if enabled

### Abi (Router) Components

- `abi.ai.personas.abi.sentiment.SentimentAnalyzer` - Analyze message sentiment
- `abi.ai.personas.abi.policy.PolicyChecker` - Check content policy
- `abi.ai.personas.abi.rules.RulesEngine` - Evaluate routing rules

### Abbey (Empathetic) Components

- `abi.ai.personas.abbey.emotion.EmotionProcessor` - Detect emotions
- `abi.ai.personas.abbey.empathy.EmpathyInjector` - Inject empathetic responses
- `abi.ai.personas.abbey.reasoning.ReasoningEngine` - Generate reasoning chains

### Aviva (Expert) Components

- `abi.ai.personas.aviva.classifier.QueryClassifier` - Classify query types
- `abi.ai.personas.aviva.knowledge.KnowledgeRetriever` - Retrieve knowledge
- `abi.ai.personas.aviva.code.CodeGenerator` - Generate code blocks
- `abi.ai.personas.aviva.facts.FactChecker` - Verify factual claims

### HTTP API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/chat` | Auto-routing chat |
| POST | `/api/v1/chat/abbey` | Force Abbey persona |
| POST | `/api/v1/chat/aviva` | Force Aviva persona |
| GET | `/api/v1/personas` | List personas and status |
| GET | `/api/v1/personas/metrics` | Get aggregated metrics |
| GET | `/api/v1/personas/health` | Health check all personas |
| WS | `/api/v1/chat/stream` | WebSocket streaming |

### Example

```zig
const personas = abi.ai.personas;

// Initialize orchestrator with defaults (Abbey/Aviva/Abi registered)
var orchestrator = try personas.MultiPersonaSystem.initWithDefaults(allocator, .{
    .default_persona = .abbey,
    .enable_dynamic_routing = true,
    .routing_confidence_threshold = 0.6,
});
defer orchestrator.deinit();

// Auto-route based on content
const response = try orchestrator.process(.{
    .content = "How do I implement a linked list?",
});

std.debug.print("Persona: {t}\n", .{response.persona});
std.debug.print("Response: {s}\n", .{response.content});

// Force specific persona
const abbey_response = try orchestrator.processWithPersona(.abbey, .{
    .content = "I'm struggling with this concept...",
});
```

See [AI Guide](docs/api/ai.md) for persona capabilities and routing.

## Connectors API

8 LLM provider connectors + Discord REST client:

- `abi.connectors.openai` - OpenAI API connector
- `abi.connectors.anthropic` - Anthropic (Claude) API connector
- `abi.connectors.ollama` - Ollama API connector (local)
- `abi.connectors.huggingface` - HuggingFace API connector
- `abi.connectors.mistral` - Mistral API connector
- `abi.connectors.cohere` - Cohere API connector
- `abi.connectors.lm_studio` - LM Studio connector (local, OpenAI-compatible)
- `abi.connectors.vllm` - vLLM connector (local, OpenAI-compatible)
- `abi.connectors.discord` - Discord REST client

Each connector provides:
- `Client.init(allocator, config)` - Initialize client
- `Client.deinit()` - Clean up resources
- `Client.isAvailable()` - Zero-allocation env var check
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
| `src/core/config/mod.zig` | Unified configuration system | ![Core](https://img.shields.io/badge/-Core-blue) |
| `src/core/framework.zig` | Framework orchestration | ![Core](https://img.shields.io/badge/-Core-blue) |
| `src/services/platform/` | Platform detection and CPU probing | ![Core](https://img.shields.io/badge/-Core-blue) |
| `src/core/registry/` | Feature registry | ![Core](https://img.shields.io/badge/-Core-blue) |
| `src/services/runtime/` | Scheduler, memory, concurrency | ![Core](https://img.shields.io/badge/-Core-blue) |
| `src/services/shared/` | Logging, security, utils | ![Shared](https://img.shields.io/badge/-Shared-yellow) |
| `src/features/gpu/` | GPU backends and unified API | ![Feature](https://img.shields.io/badge/-Feature-green) |
| `src/features/ai/` | AI module (llm, embeddings, agents, training) | ![Feature](https://img.shields.io/badge/-Feature-green) |
| `src/features/database/` | WDBX vector database | ![Feature](https://img.shields.io/badge/-Feature-green) |
| `src/features/network/` | Distributed compute and Raft | ![Feature](https://img.shields.io/badge/-Feature-green) |
| `src/features/observability/` | Metrics, tracing, profiling | ![Feature](https://img.shields.io/badge/-Feature-green) |
| `src/features/web/` | HTTP helpers and web utilities | ![Feature](https://img.shields.io/badge/-Feature-green) |
| `src/features/cloud/` | Cloud function adapters | ![Feature](https://img.shields.io/badge/-Feature-green) |
| `src/features/analytics/` | Event tracking and experiments | ![Feature](https://img.shields.io/badge/-Feature-green) |
| `src/services/connectors/` | External API connectors | ![Feature](https://img.shields.io/badge/-Feature-green) |
| `src/services/ha/` | High availability | ![Feature](https://img.shields.io/badge/-Feature-green) |
| `src/services/tasks/` | Task management | ![Feature](https://img.shields.io/badge/-Feature-green) |
| `src/services/tests/` | Test infrastructure | ![Shared](https://img.shields.io/badge/-Shared-yellow) |

> **Backward Compatibility**: Re-exports in `src/abi.zig` maintain API compatibility with the previous module layout.

## See Also

<table>
<tr>
<td>

### Guides
- [Documentation](docs/README.md) -- Documentation site source
- [API Index](docs/api/index.md) -- Full API module listing
- [Framework Reference](docs/api/framework.md) -- Configuration and lifecycle
- [AI Reference](docs/api/ai.md) -- LLM connectors and agents
- [GPU Reference](docs/api/gpu.md) -- GPU backends

</td>
<td>

### Project
- [Roadmap](roadmap.md) -- Upcoming milestones
- [CONTRIBUTING.md](../CONTRIBUTING.md) -- Development guidelines
- [CHANGELOG.md](../CHANGELOG.md) -- Version history

</td>
</tr>
</table>

---

<p align="center">
  <a href="README.md">← Back to README</a> •
  <a href="docs/README.md">Full Documentation →</a>
</p>
