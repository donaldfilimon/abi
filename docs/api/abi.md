# abi API Reference

> Main framework entry point and public API

**Source:** [`src/abi.zig`](../../src/abi.zig)

---

ABI Framework - Main Library Interface

A modern Zig 0.16 framework for modular AI services, vector search,
and high-performance compute.

## Quick Start

```zig
const std = @import("std");
const abi = @import("abi");

pub fn main() !void {
var gpa = std.heap.GeneralPurposeAllocator(.{}){};
defer _ = gpa.deinit();
const allocator = gpa.allocator();

// Minimal initialization
var fw = try abi.init(allocator);
defer fw.deinit();

// Or use the builder pattern
var fw2 = try abi.Framework.builder(allocator)
.withGpu(.{ .backend = .vulkan })
.withAi(.{ .llm = .{} })
.build();
defer fw2.deinit();
}
```

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

Initialize the ABI framework.
When called with just an allocator, uses default configuration.
When called with allocator and config, uses the provided configuration.

### `pub fn initDefault(allocator: std.mem.Allocator) !Framework`

<sup>**fn**</sup>

Initialize the ABI framework with default configuration.
Convenience function for simple initialization.

### `pub fn initWithConfig(allocator: std.mem.Allocator, cfg: anytype) !Framework`

<sup>**fn**</sup>

Initialize the ABI framework with custom configuration.
Accepts Config, FrameworkOptions (legacy), or struct literal.

### `pub fn shutdown(fw: *Framework) void`

<sup>**fn**</sup>

Shutdown the framework (convenience wrapper).

### `pub fn version() []const u8`

<sup>**fn**</sup>

Get framework version.

### `pub fn createDefaultFramework(allocator: std.mem.Allocator) !Framework`

<sup>**fn**</sup>

Create a framework with default configuration (legacy compatibility).

### `pub fn createFramework(allocator: std.mem.Allocator, config_or_options: anytype) !Framework`

<sup>**fn**</sup>

Create a framework with custom configuration (legacy compatibility).

---

*Generated automatically by `zig build gendocs`*
