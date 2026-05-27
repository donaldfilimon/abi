const std = @import("std");
const abi = @import("abi");

fn expectPlugin(
    registry: *abi.registry.Registry,
    name: []const u8,
    version: []const u8,
    description: []const u8,
    target_feature: []const u8,
    entry_point: []const u8,
) !void {
    const plugin = registry.getPlugin(name) orelse return error.MissingPlugin;
    try std.testing.expectEqualStrings(name, plugin.name);
    try std.testing.expectEqualStrings(version, plugin.version);
    try std.testing.expectEqualStrings(description, plugin.description);
    try std.testing.expectEqualStrings(target_feature, plugin.target_feature);
    try std.testing.expectEqualStrings(entry_point, plugin.entry_point);
}

fn containsName(names: []const []const u8, needle: []const u8) bool {
    for (names) |name| {
        if (std.mem.eql(u8, name, needle)) return true;
    }
    return false;
}

test "generated plugin registry exposes complete manifest metadata" {
    var registry = abi.registry.Registry.init(std.testing.allocator);
    defer registry.deinit();

    try registry.loadPlugins();

    try std.testing.expectEqual(@as(usize, 2), registry.pluginCount());
    try expectPlugin(
        &registry,
        "example-plugin",
        "0.1.0",
        "Minimal example plugin used by registry generation tests.",
        "plugins",
        "mod.zig",
    );
    try expectPlugin(
        &registry,
        "example-wdbx-plugin",
        "0.1.0",
        "Example WDBX plugin used by multi-plugin registry contract tests.",
        "wdbx",
        "mod.zig",
    );

    var names: std.ArrayListUnmanaged([]const u8) = .empty;
    defer names.deinit(std.testing.allocator);
    try registry.appendPluginNames(std.testing.allocator, &names);
    try std.testing.expectEqual(@as(usize, 2), names.items.len);
    try std.testing.expect(containsName(names.items, "example-plugin"));
    try std.testing.expect(containsName(names.items, "example-wdbx-plugin"));

    const plugins = try registry.snapshotPlugins(std.testing.allocator);
    defer abi.registry.Registry.freePluginSnapshot(std.testing.allocator, plugins);
    try std.testing.expectEqual(@as(usize, 2), plugins.len);

    const formatted = try registry.formatPluginList(std.testing.allocator);
    defer std.testing.allocator.free(formatted);
    try std.testing.expect(std.mem.indexOf(u8, formatted, "Installed Plugins (2):") != null);
    try std.testing.expect(std.mem.indexOf(u8, formatted, "example-plugin v0.1.0 [plugins] (mod.zig)") != null);
    try std.testing.expect(std.mem.indexOf(u8, formatted, "example-wdbx-plugin v0.1.0 [wdbx] (mod.zig)") != null);
}
