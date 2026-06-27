const std = @import("std");
const Registry = @import("../../core/registry.zig").Registry;
const usage_mod = @import("../usage.zig");
const abi = @import("../../root.zig");

fn loadBundledPlugin(pm: *abi.plugins.PluginManager, path: []const u8) void {
    _ = pm.loadPlugin(path) catch |err| switch (err) {
        error.AlreadyLoaded => {},
        else => std.log.warn("failed to load bundled plugin path={s} err={s}", .{ path, @errorName(err) }),
    };
}

pub fn handlePlugin(allocator: std.mem.Allocator, args: []const []const u8) !u8 {
    if (args.len < 3) return usage_mod.usageError("usage: abi plugin list | run <name> [input]");

    const sub = args[2];

    if (std.mem.eql(u8, sub, "list")) {
        var registry = Registry.init(allocator);
        defer registry.deinit();
        try registry.loadPlugins();

        const rendered = try registry.formatPluginList(allocator);
        defer allocator.free(rendered);
        std.debug.print("{s}", .{rendered});
        return 0;
    }

    if (std.mem.eql(u8, sub, "run")) {
        if (args.len < 4) return usage_mod.usageError("usage: abi plugin run <name> [input]");

        const name = args[3];
        var input: []const u8 = "";
        if (args.len > 4) {
            input = try std.mem.join(allocator, " ", args[4..]);
        }
        defer if (input.len > 0) allocator.free(input);

        var pm = abi.plugins.PluginManager.init(allocator);
        defer pm.deinit();

        // Load the two known bundled plugins so run can find them.
        loadBundledPlugin(&pm, "src/plugins/example-plugin");
        loadBundledPlugin(&pm, "src/plugins/example-wdbx-plugin");

        const output = try pm.run(allocator, name, input);
        defer allocator.free(output);

        std.debug.print("{s}\n", .{output});
        return 0;
    }

    return usage_mod.usageError("usage: abi plugin list | run <name> [input]");
}

test "plugin dispatch rejects malformed grammar with exit code 2" {
    const allocator = std.testing.allocator;
    // Missing subcommand, unknown subcommand, and `run` without a name all reject
    // with usage (exit 2) before the plugin registry is loaded.
    try std.testing.expectEqual(@as(u8, 2), try handlePlugin(allocator, &.{ "abi", "plugin" }));
    try std.testing.expectEqual(@as(u8, 2), try handlePlugin(allocator, &.{ "abi", "plugin", "bogus" }));
    try std.testing.expectEqual(@as(u8, 2), try handlePlugin(allocator, &.{ "abi", "plugin", "run" }));
}

test {
    std.testing.refAllDecls(@This());
}
