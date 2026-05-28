const std = @import("std");
const Registry = @import("../../core/registry.zig").Registry;
const usage_mod = @import("../usage.zig");
const abi = @import("../../root.zig");

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
        _ = pm.loadPlugin("src/plugins/example-plugin") catch {};
        _ = pm.loadPlugin("src/plugins/example-wdbx-plugin") catch {};

        const output = try pm.run(allocator, name, input);
        defer allocator.free(output);

        std.debug.print("{s}\n", .{output});
        return 0;
    }

    return usage_mod.usageError("usage: abi plugin list | run <name> [input]");
}
