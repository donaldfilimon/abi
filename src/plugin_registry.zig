//! Generated plugin registry. DO NOT EDIT.
const Registry = @import("core/registry.zig").Registry;

pub fn registerPlugins(registry: *Registry) !void {
    try registry.registerPlugin(.{ .name = "example-plugin", .version = "0.1.0", .description = "Minimal example plugin used by registry generation tests.", .target_feature = "plugins", .entry_point = "mod.zig" });
    try registry.registerPlugin(.{ .name = "example-wdbx-plugin", .version = "0.1.0", .description = "Example WDBX plugin used by multi-plugin registry contract tests.", .target_feature = "wdbx", .entry_point = "mod.zig" });
}
