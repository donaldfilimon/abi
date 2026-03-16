---
title: ABI Framework API Reference
purpose: Comprehensive API documentation auto-generated from source code
last_updated: 2026-03-16
target_zig_version: 0.16.0-dev.2905+5d71e3051
---

# ABI Framework API Reference

> Comprehensive API documentation auto-generated from source code.

---

## Quick Links
| Module | Category | Description | Build Flag |
| --- | --- | --- | --- |
| [app](app.md) | Core Framework | Framework Orchestration Layer | `always-on` |
| [config](config.md) | Core Framework | Configuration Module | `always-on` |
| [errors](errors.md) | Core Framework | Composable Error Hierarchy | `always-on` |
| [registry](registry.md) | Core Framework | Feature Registry System | `always-on` |
| [benchmarks](benchmarks.md) | Compute & Runtime | Benchmarks Module | `feat_benchmarks` |
| [gpu](gpu.md) | Compute & Runtime | GPU Module - Hardware Acceleration API | `feat_gpu` |
| [runtime](runtime.md) | Compute & Runtime | Runtime Module - Always-on Core Infrastructure | `always-on` |
| [ai](ai.md) | AI & Machine Learning | AI feature facade. | `feat_ai` |
| [cache](cache.md) | Data & Storage | Cache Module | `feat_cache` |
| [database](database.md) | Data & Storage | Database Feature Module | `feat_database` |
| [search](search.md) | Data & Storage | Search Module | `feat_search` |
| [storage](storage.md) | Data & Storage | Storage Module | `feat_storage` |
| [acp](acp.md) | Infrastructure | ACP (Agent Communication Protocol) Service | `always-on` |
| [cloud](cloud.md) | Infrastructure | Cloud Functions Module | `feat_cloud` |
| [gateway](gateway.md) | Infrastructure | Gateway Module | `feat_gateway` |
| [ha](ha.md) | Infrastructure | High Availability Module | `always-on` |
| [mcp](mcp.md) | Infrastructure | MCP (Model Context Protocol) Service | `always-on` |
| [messaging](messaging.md) | Infrastructure | Messaging Module | `feat_messaging` |
| [mobile](mobile.md) | Infrastructure | Mobile Module | `feat_mobile` |
| [network](network.md) | Infrastructure | Network Module | `feat_network` |
| [observability](observability.md) | Infrastructure | Observability Module | `feat_profiling` |
| [pages](pages.md) | Infrastructure | Pages Module | `feat_pages` |
| [web](web.md) | Infrastructure | Web Module - HTTP Client and Web Utilities | `feat_web` |
| [analytics](analytics.md) | Utilities | Analytics Module | `feat_analytics` |
| [auth](auth.md) | Utilities | Auth Module | `feat_auth` |
| [compute](compute.md) | Utilities | Omni-Compute Module | `feat_compute` |
| [connectors](connectors.md) | Utilities | Connector configuration loaders and auth helpers. | `always-on` |
| [desktop](desktop.md) | Utilities | Desktop Integration | `feat_desktop` |
| [documents](documents.md) | Utilities | Native Documents Parser Module | `feat_documents` |
| [foundation](foundation.md) | Utilities | Shared Utilities Module | `always-on` |
| [lsp](lsp.md) | Utilities | LSP (ZLS) service module. | `always-on` |
| [platform](platform.md) | Utilities | Platform Detection and Abstraction | `always-on` |
| [tasks](tasks.md) | Utilities | Task Management Module | `always-on` |

---

## Core Framework

### [app](app.md)

Framework Orchestration Layer

This module provides the central orchestration for the ABI framework, managing
the lifecycle of all feature modules, coordinating initialization and shutdown,
and maintaining runtime state.

## Overview

The `Framework` struct is the primary entry point for using ABI. It:

- Initializes and manages feature contexts (GPU, AI, Database, etc.)
- Maintains a feature registry for runtime feature management
- Provides typed access to enabled features
- Handles graceful shutdown and resource cleanup

## Initialization Patterns

### Default Initialization

```zig
const abi = @import("abi");

var fw = try abi.App.initDefault(allocator);
defer fw.deinit();

// All compile-time enabled features are now available
```

### Custom Configuration

```zig
var fw = try abi.App.init(allocator, .{
.gpu = .{ .backend = .vulkan },
.ai = .{ .llm = .{ .model_path = "./model.gguf" } },
.database = .{ .path = "./data" },
});
defer fw.deinit();
```

### Builder Pattern

```zig
var fw = try abi.App.builder(allocator)
.with(.gpu, abi.config.GpuConfig{ .backend = .vulkan })
.withDefault(.ai)
.withDefault(.database)
.build();
defer fw.deinit();
```

## Feature Access

```zig
// Check if a feature is enabled
if (fw.isEnabled(.gpu)) {
// Get the feature context
const gpu_ctx = try fw.get(.gpu);
// Use GPU features...
}

// Runtime context is always available
const runtime = fw.getRuntime();
```

## State Management

The framework transitions through the following states:
- `uninitialized`: Initial state before `init()`
- `initializing`: During feature initialization
- `running`: Normal operation state
- `stopping`: During shutdown
- `stopped`: After `deinit()` completes
- `failed`: If initialization fails

**Source:** [`src/core/framework.zig`](../../src/core/framework.zig)

### [config](config.md)

Configuration Module

Re-exports all configuration types from domain-specific files.
Import this module for access to all configuration types.

Use `ConfigLoader` (see `loader.zig`) to load config from environment variables
(e.g. `ABI_GPU_BACKEND`, `ABI_LLM_MODEL_PATH`). Use `Config.Builder` for fluent construction.

**Source:** [`src/core/config/mod.zig`](../../src/core/config/mod.zig)

### [errors](errors.md)

Composable Error Hierarchy

Defines the framework's error taxonomy as composable error sets.
Feature modules can import and extend these base categories.
`FrameworkError` composes lifecycle, feature, config, and allocator errors.

**Source:** [`src/core/errors.zig`](../../src/core/errors.zig)

### [registry](registry.md)

Feature Registry System

Provides a unified interface for feature registration and lifecycle management
supporting three registration modes:

- **Comptime-only**: Zero overhead, features resolved at compile time
- **Runtime-toggle**: Compiled in but can be enabled/disabled at runtime
- **Dynamic**: Features loaded from shared libraries at runtime (future)

## Usage

```zig
const registry = @import("registry");

var reg = registry.Registry.init(allocator);
defer reg.deinit();

// Register features
try reg.registerComptime(.gpu);
try reg.registerRuntimeToggle(.ai, ai_mod.Context, &ai_config);

// Query features
if (reg.isEnabled(.gpu)) {
// Use GPU...
}
```

**Source:** [`src/core/registry/mod.zig`](../../src/core/registry/mod.zig)

## Compute & Runtime

### [benchmarks](benchmarks.md)

Benchmarks Module

Performance benchmarking and timing utilities. Provides a Context for
running benchmark suites, recording results, and exporting metrics.

**Source:** [`src/features/benchmarks/mod.zig`](../../src/features/benchmarks/mod.zig) | **Flag:** `-Dfeat_benchmarks`

### [gpu](gpu.md)

GPU Module - Hardware Acceleration API

This module provides a unified interface for GPU compute operations across
multiple backends including CUDA, Vulkan, Metal, WebGPU, OpenGL, and std.gpu.

## Overview

The GPU module abstracts away backend differences, allowing you to write
portable GPU code that runs on any supported hardware. Key features include:

- **Backend Auto-detection**: Automatically selects the best available backend
- **Unified Buffer API**: Cross-platform memory management
- **Kernel DSL**: Write portable kernels that compile to any backend
- **Execution Coordinator**: Automatic fallback from GPU to SIMD to scalar
- **Multi-device Support**: Manage multiple GPUs with peer-to-peer transfers
- **Profiling**: Built-in timing and occupancy analysis

## Available Backends

| Backend | Platform | Build Flag |
|---------|----------|------------|
| CUDA | NVIDIA GPUs | `-Dgpu-backend=cuda` |
| Vulkan | Cross-platform | `-Dgpu-backend=vulkan` |
| Metal | Apple devices | `-Dgpu-backend=metal` |
| WebGPU | Web/Native | `-Dgpu-backend=webgpu` |
| OpenGL | Legacy support | `-Dgpu-backend=opengl` |
| std.gpu | Zig native | `-Dgpu-backend=stdgpu` |

## Public API

These exports form the stable interface:
- `Gpu` - Main unified GPU context
- `GpuConfig` - Configuration for GPU initialization
- `UnifiedBuffer` - Cross-backend buffer type
- `Device`, `DeviceType` - Device discovery and selection
- `KernelBuilder`, `KernelIR` - DSL for custom kernels
- `Backend`, `BackendAvailability` - Backend detection

## Quick Start

```zig
const abi = @import("abi");

// Initialize framework with GPU
var fw = try abi.App.init(allocator, .{
.gpu = .{ .backend = .auto },  // Auto-detect best backend
});
defer fw.deinit();

// Get GPU context
const gpu_ctx = try fw.get(.gpu);
const gpu = gpu_ctx.get(.gpu);

// Create buffers
var a = try gpu.createBufferFromSlice(f32, &[_]f32{ 1, 2, 3, 4 }, .{});
var b = try gpu.createBufferFromSlice(f32, &[_]f32{ 5, 6, 7, 8 }, .{});
var result = try gpu.createBuffer(f32, 4, .{});
defer {
gpu.destroyBuffer(&a);
gpu.destroyBuffer(&b);
gpu.destroyBuffer(&result);
}

// Perform vector addition
_ = try gpu.vectorAdd(&a, &b, &result);
```

## Standalone Usage

```zig
const gpu = abi.gpu;

var g = try gpu.Gpu.init(allocator, .{
.preferred_backend = .vulkan,
.allow_fallback = true,
});
defer g.deinit();

// Check device capabilities
const health = try g.getHealth();
std.debug.print("Backend: {t}\n", .{health.backend});
std.debug.print("Memory: {} MB\n", .{health.memory_total / (1024 * 1024)});
```

## Custom Kernels

```zig
const kernel = gpu.KernelBuilder.init()
.name("my_kernel")
.addParam(.{ .name = "input", .type = .buffer_f32 })
.addParam(.{ .name = "output", .type = .buffer_f32 })
.setBody(
\\output[gid] = input[gid] * 2.0;
)
.build();

// Compile for all backends
const sources = try gpu.compileAll(kernel);
```

## Internal (do not depend on)

These may change without notice:
- Direct backend module imports (cuda_loader, vulkan_*, etc.)
- Lifecycle management internals (gpu_lifecycle, cuda_backend_init_lock)
- Backend-specific initialization functions (initCudaComponents, etc.)

**Source:** [`src/features/gpu/mod.zig`](../../src/features/gpu/mod.zig) | **Flag:** `-Dfeat_gpu`

### [runtime](runtime.md)

Runtime Module - Always-on Core Infrastructure

This module provides the foundational runtime infrastructure that is always
available regardless of which features are enabled. It includes:

- Task scheduling and execution engine
- Concurrency primitives (futures, task groups, cancellation)
- Memory management utilities

## Module Organization

```
runtime/
├── mod.zig          # This file - unified entry point
├── engine/          # Task execution engine
├── scheduling/      # Futures, cancellation, task groups
├── concurrency/     # Lock-free data structures
└── memory/          # Memory pools and allocators
```

## Usage

```zig
const runtime = @import("runtime");

// Create runtime context
var ctx = try runtime.Context.init(allocator);
defer ctx.deinit();

// Use task groups for parallel work
var group = try ctx.createTaskGroup(.{});
defer group.deinit();
```

**Source:** [`src/services/runtime/mod.zig`](../../src/services/runtime/mod.zig)

## AI & Machine Learning

### [ai](ai.md)

AI feature facade.

This top-level module presents the canonical `abi.ai` surface for framework
code, tests, and external callers. Compatibility aliases delegate here while
the stub-facing contract stays aligned with `stub.zig`.

**Source:** [`src/features/ai/mod.zig`](../../src/features/ai/mod.zig) | **Flag:** `-Dfeat_ai`

## Data & Storage

### [cache](cache.md)

Cache Module

In-memory caching with LRU/LFU/FIFO/Random eviction, TTL support,
and thread-safe concurrent access via RwLock.

Architecture:
- SwissMap-backed storage for O(1) key lookup
- 4 eviction strategies: LRU (doubly-linked list), LFU (frequency buckets),
FIFO (queue), Random (RNG selection)
- Lazy TTL expiration on get() + size-triggered sweep
- RwLock for read-heavy concurrency (multiple readers, single writer)
- Cache owns all keys/values (copies on put, caller borrows on get)

**Source:** [`src/features/cache/mod.zig`](../../src/features/cache/mod.zig) | **Flag:** `-Dfeat_cache`

### [database](database.md)

Database Feature Module

Canonical public entrypoint for vector database operations.
Delegates to the core unified database engine (`core/database/mod.zig`).

This module re-exports the full public API surface so that callers get
identical types and functions regardless of whether `feat_database` selects
`mod.zig` (this file) or `stub.zig` (no-op).

**Source:** [`src/features/database/mod.zig`](../../src/features/database/mod.zig) | **Flag:** `-Dfeat_database`

### [search](search.md)

Search Module

Full-text search with inverted index, BM25 scoring, tokenization,
and snippet generation.

Architecture:
- Named indexes (SwissMap of name → InvertedIndex)
- Inverted index: term → PostingList (doc_id, term_freq, positions)
- BM25 scoring: IDF × TF component with configurable k1, b
- Tokenizer: lowercase, stop word removal
- Snippet: window with highest match density

**Source:** [`src/features/search/mod.zig`](../../src/features/search/mod.zig) | **Flag:** `-Dfeat_search`

### [storage](storage.md)

Storage Module

Unified file/object storage with vtable-based backend abstraction.

Architecture:
- Backend vtable: put/get/delete/list/exists/deinit function pointers
- Memory backend: StringHashMap-based in-memory storage
- Local backend: Planned (requires I/O backend init)
- S3/GCS: Planned (requires HTTP client)

Security: path traversal validation on all keys.

**Source:** [`src/features/storage/mod.zig`](../../src/features/storage/mod.zig) | **Flag:** `-Dfeat_storage`

## Infrastructure

### [acp](acp.md)

ACP (Agent Communication Protocol) Service

Provides an HTTP server implementing the Agent Communication Protocol
for agent-to-agent communication. Exposes an agent card at
`/.well-known/agent.json` and task management endpoints.

## Usage
```bash
abi acp serve --port 8080
curl http://localhost:8080/.well-known/agent.json
```

**Source:** [`src/services/acp/mod.zig`](../../src/services/acp/mod.zig)

### [cloud](cloud.md)

Cloud Functions Module

Provides unified adapters for deploying ABI applications as serverless
functions across major cloud providers: AWS Lambda, Google Cloud Functions,
and Azure Functions.

## Features

- **Unified Event Model**: Common `CloudEvent` struct that normalizes events
across all providers
- **Unified Response Model**: Common `CloudResponse` struct for consistent
response handling
- **Provider-Specific Adapters**: Optimized parsing and formatting for each
cloud provider
- **Context Extraction**: Access provider-specific context and metadata

## Quick Start

```zig
const std = @import("std");
const abi = @import("abi");
const cloud = abi.cloud;

/// Your function handler - same code works on all providers
fn handler(event: *cloud.CloudEvent, allocator: std.mem.Allocator) !cloud.CloudResponse {
// Access request data uniformly
const body = event.body orelse "{}";

// Return a JSON response
return try cloud.CloudResponse.json(allocator,
\\{"message": "Hello from the cloud!"}
);
}

pub fn main(init: std.process.Init) !void {
const arena = init.arena.allocator();

// Deploy to AWS Lambda
try cloud.aws_lambda.runHandler(arena, handler);

// Or Google Cloud Functions
// try cloud.gcp_functions.runHandler(arena, handler, 8080);

// Or Azure Functions
// try cloud.azure_functions.runHandler(arena, handler);
}
```

## Deployment

See the deployment templates in `deploy/` for provider-specific configurations:
- `deploy/aws/template.yaml` - AWS SAM template
- `deploy/gcp/cloudfunctions.yaml` - GCP configuration
- `deploy/azure/function.json` - Azure Functions configuration

**Source:** [`src/features/cloud/mod.zig`](../../src/features/cloud/mod.zig) | **Flag:** `-Dfeat_cloud`

### [gateway](gateway.md)

Gateway Module

API gateway with radix-tree route matching, 3 rate limiting algorithms
(token bucket, sliding window, fixed window), circuit breaker state
machine, middleware chain, and latency tracking.

Architecture:
- Radix tree for O(path_segments) route matching with params and wildcards
- Per-route rate limiters (token bucket, sliding/fixed window)
- Per-upstream circuit breakers (closed → open → half_open → closed)
- Latency histogram with 7 buckets for p50/p99 estimation
- RwLock for concurrent route lookups

**Source:** [`src/features/gateway/mod.zig`](../../src/features/gateway/mod.zig) | **Flag:** `-Dfeat_gateway`

### [ha](ha.md)

High Availability Module

Provides comprehensive high-availability features for production deployments:
- Multi-region replication
- Automated backup orchestration
- Point-in-time recovery (PITR)
- Health monitoring and automatic failover

## Quick Start

```zig
const ha = @import("ha");

var manager = ha.HaManager.init(allocator, .{
.replication_factor = 3,
.backup_interval_hours = 6,
.enable_pitr = true,
});
defer manager.deinit();

// Start HA services
try manager.start();
```

**Source:** [`src/services/ha/mod.zig`](../../src/services/ha/mod.zig)

### [mcp](mcp.md)

MCP (Model Context Protocol) Service

Provides a JSON-RPC 2.0 server over stdio for exposing ABI framework
tools to MCP-compatible AI clients (Claude Desktop, Cursor, etc.).

## Usage
```bash
abi mcp serve                          # Start MCP server (stdio)
echo '{"jsonrpc":"2.0","method":"initialize","id":1,"params":{}}' | abi mcp serve
```

## Exposed Tools
- `db_*` — Database tools
- `zls_*` — ZLS LSP tools (hover, completion, definition, etc.)

**Source:** [`src/services/mcp/mod.zig`](../../src/services/mcp/mod.zig)

### [messaging](messaging.md)

Messaging Module

Topic-based pub/sub messaging with MQTT-style pattern matching,
synchronous delivery, dead letter queue, and backpressure.

Architecture:
- Topic registry with subscriber lists
- Pattern matching: `events.*` (single-level), `events.#` (multi-level)
- Bounded message queues with backpressure (returns ChannelFull)
- Dead letter queue for failed deliveries
- Synchronous delivery (publish blocks until all subscribers notified)
- RwLock for concurrent topic lookups

**Source:** [`src/features/messaging/mod.zig`](../../src/features/messaging/mod.zig) | **Flag:** `-Dfeat_messaging`

### [mobile](mobile.md)

Mobile Module

Platform lifecycle, sensors, notifications, permissions, and device info.
Provides simulated mobile platform behavior for development and testing.

**Source:** [`src/features/mobile/mod.zig`](../../src/features/mobile/mod.zig) | **Flag:** `-Dfeat_mobile`

### [network](network.md)

Network Module

Distributed compute network with node discovery, Raft consensus,
and distributed task coordination.

## Features
- Node registry and discovery
- Raft consensus for leader election
- Task scheduling and load balancing
- Connection pooling and retry logic
- Circuit breakers for fault tolerance
- Rate limiting

## Usage

```zig
const network = @import("network");

// Initialize the network module
try network.init(allocator);
defer network.deinit();

// Get the node registry
const registry = try network.defaultRegistry();
try registry.register("node-a", "127.0.0.1:9000");
```

**Source:** [`src/features/network/mod.zig`](../../src/features/network/mod.zig) | **Flag:** `-Dfeat_network`

### [observability](observability.md)

Observability Module

Unified observability with metrics, tracing, and profiling.

## Features
- Metrics collection and export (Prometheus, OpenTelemetry, StatsD)
- Distributed tracing
- Performance profiling
- Circuit breakers and error aggregation
- Alerting rules and notifications

**Source:** [`src/features/observability/mod.zig`](../../src/features/observability/mod.zig) | **Flag:** `-Dfeat_profiling`

### [pages](pages.md)

Pages Module

Dashboard/UI pages with radix-tree URL routing, template rendering,
and support for both static and dynamic content.

Architecture:
- Radix tree for O(path_segments) page matching with params and wildcards
- Single-pass {{variable}} template substitution
- Per-page flags for auth requirement and cache TTL
- RwLock for concurrent page lookups

**Source:** [`src/features/observability/pages/mod.zig`](../../src/features/observability/pages/mod.zig) | **Flag:** `-Dfeat_pages`

### [web](web.md)

Web Module - HTTP Client and Web Utilities

This module provides HTTP client functionality, weather API integration,
and persona API handlers for the ABI framework. It wraps Zig's standard
library HTTP client with convenient utilities for common web operations.

## Features

- **HTTP Client**: Synchronous HTTP client with configurable options
- GET and POST requests with JSON support
- Configurable timeouts, redirects, and response size limits
- Thread-safe global client with mutex protection

- **Weather Client**: Integration with Open-Meteo weather API
- Coordinate-based weather forecasts
- Location validation and URL building

- **Persona API**: HTTP handlers and routes for AI persona system
- Chat request/response handlers
- REST API routes with OpenAPI documentation
- Health check and metrics endpoints

## Usage Example

```zig
const web = @import("abi").web;

// Initialize the web module
try web.init(allocator);
defer web.deinit();

// Make an HTTP GET request
const response = try web.get(allocator, "https://api.example.com/data");
defer web.freeResponse(allocator, response);

if (web.isSuccessStatus(response.status)) {
// Parse JSON response
var parsed = try web.parseJsonValue(allocator, response);
defer parsed.deinit();
// Use parsed.value...
}
```

## Using the Context API

For Framework integration, use the Context struct:

```zig
const cfg = config_module.WebConfig{};
var ctx = try web.Context.init(allocator, cfg);
defer ctx.deinit();

const response = try ctx.get("https://api.example.com/data");
defer ctx.freeResponse(response);
```

## POST Request with JSON

```zig
const body = "{\"message\": \"hello\"}";
const response = try web.postJson(allocator, "https://api.example.com/chat", body);
defer web.freeResponse(allocator, response);
```

## Request Options

```zig
const response = try web.getWithOptions(allocator, url, .{
.max_response_bytes = 10 * 1024 * 1024,  // 10MB limit
.user_agent = "my-app/1.0",
.follow_redirects = true,
.redirect_limit = 5,
.extra_headers = &.{
.{ .name = "Authorization", .value = "Bearer token" },
},
});
```

## Error Handling

The module uses `HttpError` for HTTP-specific errors:
- `InvalidUrl`: URL parsing failed
- `InvalidRequest`: Request configuration is invalid
- `RequestFailed`: HTTP request failed
- `ConnectionFailed`: Network connection failed
- `ResponseTooLarge`: Response exceeds max_response_bytes
- `Timeout`: Request timed out
- `ReadFailed`: Error reading response body

## Feature Flag

This module is controlled by `-Dfeat-web=true` (default: enabled).
When disabled, all operations return `error.WebDisabled`.

## Thread Safety

The global `init()`/`deinit()` functions use mutex protection for
thread-safe access to the default client. The `Context` struct should
be used per-thread or with external synchronization.

**Source:** [`src/features/web/mod.zig`](../../src/features/web/mod.zig) | **Flag:** `-Dfeat_web`

## Utilities

### [analytics](analytics.md)

Analytics Module

Business event tracking, session analytics, and funnel instrumentation.
Unlike observability (system metrics, tracing, profiling), analytics
focuses on user-facing events and product usage patterns.

## Features
- Custom event tracking with typed properties
- Session lifecycle management
- Funnel step recording
- A/B experiment assignment tracking
- Thread-safe event buffer with configurable flush

**Source:** [`src/features/analytics/mod.zig`](../../src/features/analytics/mod.zig) | **Flag:** `-Dfeat_analytics`

### [auth](auth.md)

Auth Module

Authentication and security infrastructure for the ABI framework.
Re-exports the full security infrastructure from `services/shared/security/`.

When the `auth` feature is enabled, all security sub-modules are available:
- `abi.auth.jwt` — JSON Web Tokens (HMAC-SHA256/384/512)
- `abi.auth.api_keys` — API key management with secure hashing
- `abi.auth.rbac` — Role-based access control
- `abi.auth.session` — Session management
- `abi.auth.password` — Secure password hashing (Argon2id, PBKDF2, scrypt)
- `abi.auth.cors` — Cross-Origin Resource Sharing
- `abi.auth.rate_limit` — Token bucket, sliding window, leaky bucket
- `abi.auth.encryption` — AES-256-GCM, ChaCha20-Poly1305
- `abi.auth.tls` / `abi.auth.mtls` — Transport security
- `abi.auth.certificates` — X.509 certificate management
- `abi.auth.secrets` — Encrypted credential storage
- `abi.auth.audit` — Tamper-evident security event logging
- `abi.auth.validation` — Input sanitization
- `abi.auth.ip_filter` — IP allow/deny lists
- `abi.auth.headers` — Security headers middleware

**Source:** [`src/features/auth/mod.zig`](../../src/features/auth/mod.zig) | **Flag:** `-Dfeat_auth`

### [compute](compute.md)

Omni-Compute Module

Provides the distributed mesh networking, multi-GPU orchestration,
and tensor sharing protocols.

**Source:** [`src/features/compute/mod.zig`](../../src/features/compute/mod.zig) | **Flag:** `-Dfeat_compute`

### [connectors](connectors.md)

Connector configuration loaders and auth helpers.

This module provides unified access to various AI service connectors including:

- **OpenAI**: GPT models via the Chat Completions API
- **Anthropic**: Claude models via the Messages API
- **Ollama**: Local LLM inference server
- **HuggingFace**: Hosted inference API
- **Mistral**: Mistral AI models with OpenAI-compatible API
- **Cohere**: Chat, embeddings, and reranking
- **LM Studio**: Local LLM inference with OpenAI-compatible API
- **vLLM**: High-throughput local LLM serving with OpenAI-compatible API
- **MLX**: Apple Silicon-optimized inference via mlx-lm server
- **Discord**: Bot integration for Discord

## Usage

Each connector can be loaded from environment variables:

```zig
const connectors = @import("abi").connectors;

// Load and create clients
if (try connectors.tryLoadOpenAI(allocator)) |config| {
var client = try connectors.openai.Client.init(allocator, config);
defer client.deinit();
// Use client...
}
```

## Security

All connectors securely wipe API keys from memory using `std.crypto.secureZero`
before freeing to prevent memory forensics attacks.

**Source:** [`src/services/connectors/mod.zig`](../../src/services/connectors/mod.zig)

### [desktop](desktop.md)

Desktop Integration

Provides native UI extensions and integrations for the host OS.

**Source:** [`src/features/desktop/mod.zig`](../../src/features/desktop/mod.zig) | **Flag:** `-Dfeat_desktop`

### [documents](documents.md)

Native Documents Parser Module

Provides zero-dependency parsers for complex file formats like
HTML, DOM trees, and PDF binaries.

**Source:** [`src/features/documents/mod.zig`](../../src/features/documents/mod.zig) | **Flag:** `-Dfeat_documents`

### [foundation](foundation.md)

Shared Utilities Module

Common utilities, helpers, and cross-cutting concerns used throughout the ABI framework.
This module consolidates logging, SIMD operations, platform utilities, and security.

# Overview

The shared module provides foundational building blocks that are used across all ABI
framework components. It is organized into several categories:

- **Core Utilities**: Error handling, logging, time, I/O operations
- **Security**: Authentication, authorization, encryption, secrets management
- **Performance**: SIMD operations, memory management, binary serialization
- **Networking**: HTTP client, network utilities, encoding/decoding

# Usage

Import the shared module and access components directly:

```zig
// From external modules (CLI, tests):
const shared = @import("abi").foundation;
// From within the abi module:
// const shared = @import("../../services/shared/mod.zig");

// Logging
shared.log.info("Application started", .{});

// SIMD operations
const dot = shared.vectorDot(a, b);

// Security
var jwt_manager = shared.security.JwtManager.init(allocator, secret, .{});
```

# Security Components

The security sub-module provides comprehensive security features:

| Component | Description |
|-----------|-------------|
| `api_keys` | API key generation, validation, rotation |
| `jwt` | JSON Web Token creation and verification |
| `rbac` | Role-based access control |
| `tls` | TLS/SSL connection management |
| `secrets` | Encrypted secrets storage |
| `rate_limit` | Request rate limiting |
| `encryption` | Data encryption at rest |
| `audit` | Security audit logging |

# Thread Safety

Most components in this module are thread-safe when used with proper synchronization.
Security components like `JwtManager`, `RateLimiter`, and `SecretsManager` include
internal mutex protection for concurrent access.

**Source:** [`src/services/shared/mod.zig`](../../src/services/shared/mod.zig)

### [lsp](lsp.md)

LSP (ZLS) service module.

**Source:** [`src/services/lsp/mod.zig`](../../src/services/lsp/mod.zig)

### [platform](platform.md)

Platform Detection and Abstraction

Provides OS, architecture, and capability detection for cross-platform code.
This module consolidates all platform-specific detection and abstraction logic.

## Usage

```zig
const platform = @import("abi").platform;

const info = platform.getPlatformInfo();
std.debug.print("OS: {t}, Arch: {t}, Cores: {d}\n", .{
info.os,
info.arch,
info.max_threads,
});

if (platform.supportsThreading()) {
// Use multi-threaded code path
}
```

**Source:** [`src/services/platform/mod.zig`](../../src/services/platform/mod.zig)

### [tasks](tasks.md)

Task Management Module

Provides unified task tracking for personal tasks, project roadmap
items, and distributed compute jobs.

## Usage

```zig
const tasks = @import("tasks");

var manager = try tasks.Manager.init(allocator, .{});
defer manager.deinit();

const id = try manager.add("Fix bug", .{ .priority = .high });
try manager.complete(id);
```

**Source:** [`src/services/tasks/mod.zig`](../../src/services/tasks/mod.zig)



---

*Generated automatically by `zig build gendocs`*


## Workflow Contract
- Canonical repo workflow: [AGENTS.md](../../AGENTS.md)
- Active execution tracker: [tasks/todo.md](../../tasks/todo.md)
- Correction log: [tasks/lessons.md](../../tasks/lessons.md)

## Zig Validation
Use `zig build full-check` on supported hosts. On Darwin 25+ / 26+, use `zig fmt --check ...` plus `./tools/scripts/run_build.sh <step>`. For docs generation, use `zig build gendocs` or `./tools/scripts/run_build.sh gendocs` on Darwin.
