//! Public API root for the ABI framework. Re-exports all public modules at the
//! top level. Consumers and the MCP handler graph import via `@import("abi")`.
//! This file defines the library's module topology but does not re-export every
//! leaf function — use the modules directly for fine-grained access.
//! Version is declared per build.zig; no runtime version constant is exposed
//! (consumers may read `abi.MCP.serverInfo.version` at connect time).
pub const interfaces = @import("interfaces.zig");
pub const foundation = @import("foundation/mod.zig");
pub const features = @import("features/mod.zig");
pub const registry = @import("core/registry.zig");
pub const config = @import("core/config.zig");
pub const connectors = @import("connectors/mod.zig");
pub const memory = @import("core/memory.zig");
pub const scheduler = @import("core/scheduler.zig");
pub const plugins = @import("plugins/plugin_manager.zig");

// Re-export the most common plugin operations at the top level for convenience
pub const runPlugin = plugins.PluginManager.run;

test {
    const std = @import("std");
    _ = @import("foundation/mod.zig");
    _ = @import("features/mod.zig");
    _ = @import("connectors/mod.zig");
    _ = @import("core/config.zig");
    _ = @import("core/memory.zig");
    _ = @import("core/scheduler.zig");
    _ = @import("core/registry.zig");
    _ = @import("plugins/plugin_manager.zig");
    std.testing.refAllDecls(@This());
}
