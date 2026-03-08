# ABI Framework API Reference

> Comprehensive API documentation auto-generated from source code.

---

## Quick Links
| Module | Category | Description | Build Flag |
| --- | --- | --- | --- |
| [config](config.md) | Core Framework | Unified configuration system. | `always-on` |
| [errors](errors.md) | Core Framework | Composable error hierarchy for framework operations. | `always-on` |
| [feature_catalog](feature_catalog.md) | Core Framework | Canonical feature catalog for ABI. | `always-on` |
| [framework](framework.md) | Core Framework | Framework orchestration with builder pattern. | `always-on` |
| [registry](registry.md) | Core Framework | Plugin registry for feature management. | `always-on` |
| [runtime](runtime.md) | Compute & Runtime | Runtime Module - Always-on Core Infrastructure | `always-on` |
| [simd](simd.md) | Compute & Runtime | SIMD vector operations | `always-on` |
| [acp](acp.md) | Infrastructure | ACP (Agent Communication Protocol) Service | `always-on` |
| [ha](ha.md) | Infrastructure | High Availability Module | `always-on` |
| [mcp](mcp.md) | Infrastructure | MCP (Model Context Protocol) Service | `always-on` |
| [connectors](connectors.md) | Utilities | Connector configuration loaders and auth helpers. | `always-on` |
| [lsp](lsp.md) | Utilities | LSP (ZLS) service module. | `always-on` |
| [platform](platform.md) | Utilities | Platform Detection and Abstraction | `always-on` |
| [shared](shared.md) | Utilities | Shared Utilities Module | `always-on` |
| [tasks](tasks.md) | Utilities | Task Management Module | `always-on` |

---

## Core Framework

### [config](config.md)

Unified configuration system.

**Source:** [`src/core/config/mod.zig`](../../src/core/config/mod.zig)

### [errors](errors.md)

Composable error hierarchy for framework operations.

**Source:** [`src/core/errors.zig`](../../src/core/errors.zig)

### [feature_catalog](feature_catalog.md)

Canonical feature catalog for ABI.

Centralizes feature descriptions, compile-time flag mappings, parent-child
relationships, and real/stub module paths used by parity checks.

**Source:** [`src/core/feature_catalog.zig`](../../src/core/feature_catalog.zig)

### [framework](framework.md)

Framework orchestration with builder pattern.

**Source:** [`src/core/framework.zig`](../../src/core/framework.zig)

### [registry](registry.md)

Plugin registry for feature management.

**Source:** [`src/core/registry/mod.zig`](../../src/core/registry/mod.zig)

## Compute & Runtime

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
const runtime = @import("runtime/mod.zig");

// Create runtime context
var ctx = try runtime.Context.init(allocator);
defer ctx.deinit();

// Use task groups for parallel work
var group = try ctx.createTaskGroup(.{});
defer group.deinit();
```

**Source:** [`src/services/runtime/mod.zig`](../../src/services/runtime/mod.zig)

### [simd](simd.md)

SIMD vector operations

Re-exports from focused submodules. Every public symbol from the
original monolithic simd.zig is available here.

**Source:** [`src/services/shared/simd/mod.zig`](../../src/services/shared/simd/mod.zig)

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

### [ha](ha.md)

High Availability Module

Provides comprehensive high-availability features for production deployments:
- Multi-region replication
- Automated backup orchestration
- Point-in-time recovery (PITR)
- Health monitoring and automatic failover

## Quick Start

```zig
const ha = @import("ha/mod.zig");

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
- `db_query` — Vector similarity search
- `db_insert` — Insert vectors with metadata
- `db_stats` — Database statistics
- `db_list` — List stored vectors
- `db_delete` — Delete a vector by ID
- `zls_*` — ZLS LSP tools (hover, completion, definition, etc.)

**Source:** [`src/services/mcp/mod.zig`](../../src/services/mcp/mod.zig)

## Utilities

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

### [shared](shared.md)

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
const shared = @import("shared");

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

### [tasks](tasks.md)

Task Management Module

Provides unified task tracking for personal tasks, project roadmap
items, and distributed compute jobs.

## Usage

```zig
const tasks = @import("tasks/mod.zig");

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
Use the `$zig-master` Codex skill for ABI Zig validation, docs generation, and build-wiring changes.
