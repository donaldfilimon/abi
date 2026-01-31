# abi API Reference

> Main framework entry point and public API

**Source:** [`src/abi.zig`](../../src/abi.zig)

---

ABI Framework - Main Library Interface

A modern Zig 0.16 framework for modular AI services, vector search,
and high-performance compute. This is the primary entry point for all
ABI functionality.

## Features

- **AI Module**: Local LLM inference, embeddings, agents, training pipelines
- **GPU Acceleration**: Multi-backend support (CUDA, Vulkan, Metal, WebGPU)
- **Vector Database**: WDBX with HNSW/IVF-PQ indexing
- **Distributed Compute**: Raft consensus, task distribution
- **Observability**: Metrics, tracing, and profiling

## Quick Start

```zig
const std = @import("std");
const abi = @import("abi");

pub fn main() !void {
var gpa = std.heap.GeneralPurposeAllocator(.{}){};
defer _ = gpa.deinit();
const allocator = gpa.allocator();

// Minimal initialization with defaults
var fw = try abi.initDefault(allocator);
defer fw.deinit();

// Check framework version
std.debug.print("ABI v{s}\n", .{abi.version()});
}
```

## Builder Pattern

For more control over which features are enabled:

```zig
var fw = try abi.Framework.builder(allocator)
.withGpu(.{ .backend = .vulkan })
.withAi(.{ .llm = .{ .model_path = "./models/llama.gguf" } })
.withDatabase(.{ .path = "./data" })
.build();
defer fw.deinit();

// Access enabled features
if (fw.isEnabled(.gpu)) {
const gpu_ctx = try fw.getGpu();
// Use GPU features...
}
```

## Feature Modules

Access feature modules through the namespace exports:
- `abi.ai` - AI capabilities (LLM, embeddings, agents, training)
- `abi.gpu` - GPU acceleration and compute
- `abi.database` - Vector database operations
- `abi.network` - Distributed networking
- `abi.observability` - Metrics and tracing
- `abi.web` - HTTP utilities

---

## API

### `pub const config`

<sup>**type**</sup>

Unified configuration system.

### `pub const framework`

<sup>**type**</sup>

Framework orchestration with builder pattern.

### `pub const registry`

<sup>**type**</sup>

Plugin registry for feature management.

### `pub const runtime`

<sup>**type**</sup>

Runtime infrastructure (always available).

### `pub const platform`

<sup>**type**</sup>

Platform detection and abstraction.

### `pub const shared`

<sup>**type**</sup>

Shared utilities.

### `pub const gpu`

<sup>**const**</sup>

GPU acceleration.

### `pub const ai`

<sup>**const**</sup>

AI capabilities (modular sub-features).

### `pub const database`

<sup>**const**</sup>

Vector database.

### `pub const network`

<sup>**const**</sup>

Distributed network.

### `pub const observability`

<sup>**const**</sup>

Observability (metrics, tracing, profiling).

### `pub const systemInfo`

<sup>**const**</sup>

Convenience alias for system information utilities.

### `pub const web`

<sup>**const**</sup>

Web utilities.

### `pub const cloud`

<sup>**const**</sup>

Cloud function adapters.

### `pub const ha`

<sup>**type**</sup>

High availability (replication, backup, PITR).

### `pub const tasks`

<sup>**type**</sup>

Task management system.

### `pub const core`

<sup>**type**</sup>

Core utilities (legacy).

### `pub const connectors`

<sup>**type**</sup>

Connectors (legacy).

### `pub const monitoring`

<sup>**const**</sup>

Monitoring (legacy - use observability).

### `pub const wdbx`

<sup>**const**</sup>

WDBX compatibility namespace.

### `pub fn init(allocator: std.mem.Allocator, config_or_options: anytype) !Framework`

<sup>**fn**</sup>

Initialize the ABI framework with the given configuration.

This is a flexible initialization function that accepts multiple configuration types
for backward compatibility. For new code, prefer using `initDefault` or `initWithConfig`.

## Parameters

- `allocator`: Memory allocator for framework resources. The framework stores this
allocator and uses it for all internal allocations. The caller retains ownership
and must ensure the allocator outlives the framework.
- `config_or_options`: Configuration for the framework. Accepts:
- `Config`: The new unified configuration struct
- `FrameworkOptions`: Legacy options struct (deprecated)
- `void`: Uses default configuration
- Struct literal: Coerced to `Config`

## Returns

A fully initialized `Framework` instance, or an error if initialization fails.

## Errors

- `ConfigError.FeatureDisabled`: A feature is enabled in config but disabled at compile time
- `error.OutOfMemory`: Memory allocation failed
- `error.FeatureInitFailed`: A feature module failed to initialize

## Example

```zig
// With explicit Config
var fw = try abi.init(allocator, abi.Config.defaults());
defer fw.deinit();

// With struct literal
var fw = try abi.init(allocator, .{
.gpu = .{ .backend = .vulkan },
});
defer fw.deinit();
```

### `pub fn initDefault(allocator: std.mem.Allocator) !Framework`

<sup>**fn**</sup>

Initialize the ABI framework with default configuration.

This is the simplest way to initialize the framework. It enables all features
that are available at compile time with their default settings.

## Parameters

- `allocator`: Memory allocator for framework resources

## Returns

A fully initialized `Framework` instance with default configuration.

## Example

```zig
var gpa = std.heap.GeneralPurposeAllocator(.{}){};
defer _ = gpa.deinit();

var fw = try abi.initDefault(gpa.allocator());
defer fw.deinit();

// Framework is ready to use with all default features
std.debug.print("Version: {s}\n", .{abi.version()});
```

### `pub fn initWithConfig(allocator: std.mem.Allocator, cfg: anytype) !Framework`

<sup>**fn**</sup>

Initialize the ABI framework with custom configuration.

Use this function when you need fine-grained control over which features
are enabled and their configuration. Accepts multiple configuration formats
for backward compatibility.

## Parameters

- `allocator`: Memory allocator for framework resources
- `cfg`: Configuration, one of:
- `Config`: New unified configuration struct (recommended)
- `FrameworkOptions`: Legacy options (deprecated)
- Struct literal: Coerced to `Config`
- Empty struct `{}`: Uses default configuration

## Returns

A fully initialized `Framework` instance with the specified configuration.

## Example

```zig
// Enable only specific features
var fw = try abi.initWithConfig(allocator, .{
.gpu = .{ .backend = .cuda },
.ai = .{
.llm = .{ .model_path = "./models/llama-7b.gguf" },
},
.database = .{ .path = "./vector_db" },
});
defer fw.deinit();
```

### `pub fn shutdown(fw: *Framework) void`

<sup>**fn**</sup>

Shutdown and clean up the framework.

This is a convenience wrapper around `Framework.deinit()`. It releases all
resources held by the framework, including feature contexts, the registry,
and internal state.

After calling this function, the framework instance should not be used.

## Parameters

- `fw`: Pointer to the framework instance to shut down

## Example

```zig
var fw = try abi.initDefault(allocator);
// ... use the framework ...
abi.shutdown(&fw);  // Clean up resources
```

## Note

Using `defer fw.deinit()` directly is equivalent and often preferred:
```zig
var fw = try abi.initDefault(allocator);
defer fw.deinit();  // Automatically clean up on scope exit
```

### `pub fn version() []const u8`

<sup>**fn**</sup>

Get the ABI framework version string.

Returns the semantic version of the ABI framework as defined at build time.
This can be used for logging, compatibility checks, or displaying version
information to users.

## Returns

A compile-time constant string containing the version (e.g., "0.1.0").

## Example

```zig
std.debug.print("Running ABI Framework v{s}\n", .{abi.version()});
```

### `pub fn createDefaultFramework(allocator: std.mem.Allocator) !Framework`

<sup>**fn**</sup>

Create a framework with default configuration (legacy compatibility).

### `pub fn createFramework(allocator: std.mem.Allocator, config_or_options: anytype) !Framework`

<sup>**fn**</sup>

Create a framework with custom configuration (legacy compatibility).

---

*Generated automatically by `zig build gendocs`*
