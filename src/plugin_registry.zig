//! Generated plugin registry. DO NOT EDIT.
const Registry = @import("registry.zig").Registry;

pub fn registerPlugins(registry: *Registry) !void {
    try registry.register("example-plugin", "plugin information");
}
