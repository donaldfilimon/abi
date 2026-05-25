const std = @import("std");
const abi = @import("registry");

test "generated plugin registry exposes complete manifest metadata" {
    var registry = abi.registry.Registry.init(std.testing.allocator);
    defer registry.deinit();

    try registry.loadPlugins();

    const plugin = registry.getPlugin("example-plugin") orelse return error.MissingPlugin;
    try std.testing.expectEqualStrings("example-plugin", plugin.name);
    try std.testing.expectEqualStrings("0.1.0", plugin.version);
    try std.testing.expectEqualStrings("plugins", plugin.target_feature);
    try std.testing.expectEqualStrings("mod.zig", plugin.entry_point);
    try std.testing.expect(std.mem.indexOf(u8, plugin.description, "Minimal example plugin") != null);
}
