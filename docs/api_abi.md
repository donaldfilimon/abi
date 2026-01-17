# abi API Reference

**Source:** `src/abi.zig`

 ABI Framework - Main Library Interface
 High level entrypoints and re-exports for the modernized runtime.
### `pub const core`

 Core utilities and fundamental types

### `pub const features`

 Feature modules grouped for discoverability

### `pub const ai`

 Individual feature namespaces re-exported at the root for ergonomic imports.

### `pub const framework`

 Framework orchestration layer that coordinates features and plugins.

### `pub const wdbx`

 Compatibility namespace for the WDBX tooling.

### `pub fn init(allocator: std.mem.Allocator, config_or_options: anytype) !Framework`

 Initialise the ABI framework and return the orchestration handle.

### `pub fn shutdown(instance: *Framework) void`

 Convenience wrapper around `Framework.deinit`.

### `pub fn version() []const u8`

 Get framework version information.

### `pub fn createDefaultFramework(allocator: std.mem.Allocator) !Framework`

 Create a framework with default configuration.

### `pub fn createFramework(allocator: std.mem.Allocator, config_or_options: anytype) !Framework`

 Create a framework with custom configuration.

