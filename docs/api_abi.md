# abi API Reference

**Source:** `src/abi.zig`

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
### `pub const config`

 Unified configuration system.

### `pub const framework`

 Framework orchestration with builder pattern.

### `pub const registry`

 Plugin registry for feature management.

### `pub const runtime`

 Runtime infrastructure (always available).

### `pub const gpu`

 GPU acceleration.

### `pub const ai`

 AI capabilities (modular sub-features).

### `pub const database`

 Vector database.

### `pub const network`

 Distributed network.

### `pub const observability`

 Observability (metrics, tracing, profiling).

### `pub const web`

 Web utilities.

### `pub const ha`

 High availability (replication, backup, PITR).

### `pub const tasks`

 Task management system.

### `pub const core`

 Core utilities (legacy).

### `pub const features`

 Features module (legacy - use direct imports above).

### `pub const connectors`

 Connectors (legacy).

### `pub const monitoring`

 Monitoring (legacy - use observability).

### `pub const wdbx`

 WDBX compatibility namespace.

### `pub fn init(allocator: std.mem.Allocator, config_or_options: anytype) !Framework`

 Initialize the ABI framework.
 When called with just an allocator, uses default configuration.
 When called with allocator and config, uses the provided configuration.

### `pub fn initDefault(allocator: std.mem.Allocator) !Framework`

 Initialize the ABI framework with default configuration.
 Convenience function for simple initialization.

### `pub fn initWithConfig(allocator: std.mem.Allocator, cfg: anytype) !Framework`

 Initialize the ABI framework with custom configuration.
 Accepts Config, FrameworkOptions (legacy), or struct literal.

### `pub fn shutdown(fw: *Framework) void`

 Shutdown the framework (convenience wrapper).

### `pub fn version() []const u8`

 Get framework version.

### `pub fn createDefaultFramework(allocator: std.mem.Allocator) !Framework`

 Create a framework with default configuration (legacy compatibility).

### `pub fn createFramework(allocator: std.mem.Allocator, config_or_options: anytype) !Framework`

 Create a framework with custom configuration (legacy compatibility).

