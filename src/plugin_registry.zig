//! Generated plugin registry. DO NOT EDIT.
const Registry = @import("core/registry.zig").Registry;

pub fn registerPlugins(registry: *Registry) !void {
    try registry.register("example-plugin", "Minimal example plugin used by registry generation tests.");
}
