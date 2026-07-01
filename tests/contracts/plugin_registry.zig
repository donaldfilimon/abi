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

    try std.testing.expectEqual(@as(usize, 16), registry.pluginCount());
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
    try expectPlugin(
        &registry,
        "telemetry-exporter",
        "0.1.0",
        "Example telemetry plugin: formats a telemetry event line for the feat-telemetry observability path.",
        "telemetry",
        "mod.zig",
    );
    const feature_plugins = [_]struct { name: []const u8, feature: []const u8 }{
        .{ .name = "ai-plugin", .feature = "ai" },
        .{ .name = "gpu-plugin", .feature = "gpu" },
        .{ .name = "accelerator-plugin", .feature = "accelerator" },
        .{ .name = "shader-plugin", .feature = "shader" },
        .{ .name = "mlir-plugin", .feature = "mlir" },
        .{ .name = "os-control-plugin", .feature = "os-control" },
        .{ .name = "hash-plugin", .feature = "hash" },
        .{ .name = "tui-plugin", .feature = "tui" },
        .{ .name = "nn-plugin", .feature = "nn" },
        .{ .name = "metrics-plugin", .feature = "metrics" },
        .{ .name = "sea-plugin", .feature = "sea" },
        .{ .name = "mobile-plugin", .feature = "mobile" },
        .{ .name = "foundationmodels-plugin", .feature = "foundationmodels" },
    };
    inline for (feature_plugins) |fp| {
        var desc_buf: [128]u8 = undefined;
        const desc = try std.fmt.bufPrint(&desc_buf, "Example reference plugin targeting the feat-{s} gate.", .{fp.feature});
        try expectPlugin(&registry, fp.name, "0.1.0", desc, fp.feature, "mod.zig");
    }

    var names: std.ArrayListUnmanaged([]const u8) = .empty;
    defer names.deinit(std.testing.allocator);
    try registry.appendPluginNames(std.testing.allocator, &names);
    try std.testing.expectEqual(@as(usize, 16), names.items.len);
    try std.testing.expect(containsName(names.items, "example-plugin"));
    try std.testing.expect(containsName(names.items, "example-wdbx-plugin"));
    try std.testing.expect(containsName(names.items, "telemetry-exporter"));
    inline for (feature_plugins) |fp| {
        try std.testing.expect(containsName(names.items, fp.name));
    }

    const plugins = try registry.snapshotPlugins(std.testing.allocator);
    defer abi.registry.Registry.freePluginSnapshot(std.testing.allocator, plugins);
    try std.testing.expectEqual(@as(usize, 16), plugins.len);

    const formatted = try registry.formatPluginList(std.testing.allocator);
    defer std.testing.allocator.free(formatted);
    try std.testing.expect(std.mem.indexOf(u8, formatted, "Installed Plugins (16):") != null);
    try std.testing.expect(std.mem.indexOf(u8, formatted, "example-plugin v0.1.0 [plugins] (mod.zig)") != null);
    try std.testing.expect(std.mem.indexOf(u8, formatted, "example-wdbx-plugin v0.1.0 [wdbx] (mod.zig)") != null);
    try std.testing.expect(std.mem.indexOf(u8, formatted, "telemetry-exporter v0.1.0 [telemetry] (mod.zig)") != null);
    inline for (feature_plugins) |fp| {
        var line_buf: [128]u8 = undefined;
        const line = try std.fmt.bufPrint(&line_buf, "{s} v0.1.0 [{s}] (mod.zig)", .{ fp.name, fp.feature });
        try std.testing.expect(std.mem.indexOf(u8, formatted, line) != null);
    }
}
