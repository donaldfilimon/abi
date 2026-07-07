const std = @import("std");
const Registry = @import("../../core/registry.zig").Registry;
const usage_mod = @import("../usage.zig");
const abi = @import("../../root.zig");

/// `abi plugin list | run <name> [input]`: list the registered plugins or run a
/// named bundled plugin against optional input. `run` loads the shared bundled
/// plugin set (symmetric with the MCP surface) before dispatching. Returns the
/// process exit code.
pub fn handlePlugin(allocator: std.mem.Allocator, args: []const []const u8) !u8 {
    if (args.len < 3) return usage_mod.usageError("usage: abi plugin list | run <name> [input]");

    const sub = args[2];
    if (usage_mod.isHelpToken(sub)) return usage_mod.printCommandHelp("plugin");

    if (std.mem.eql(u8, sub, "list")) {
        if (args.len == 4 and usage_mod.isHelpToken(args[3])) return pluginListHelp();
        if (args.len != 3) return usage_mod.usageError("usage: abi plugin list");
        return handlePluginList(allocator);
    }

    if (std.mem.eql(u8, sub, "run")) {
        if (args.len == 4 and usage_mod.isHelpToken(args[3])) return pluginRunHelp();
        if (args.len < 4) return usage_mod.usageError("usage: abi plugin run <name> [input]");

        const name = args[3];
        var input: []const u8 = "";
        var joined_input: ?[]u8 = null;
        if (args.len > 4) {
            joined_input = try std.mem.join(allocator, " ", args[4..]);
            input = joined_input.?;
        }
        defer if (joined_input) |owned| allocator.free(owned);
        return handlePluginRun(allocator, name, input);
    }

    return usage_mod.usageError("usage: abi plugin list | run <name> [input]");
}

pub fn handlePluginList(allocator: std.mem.Allocator) !u8 {
    var registry = Registry.init(allocator);
    defer registry.deinit();
    try registry.loadPlugins();

    const rendered = try registry.formatPluginList(allocator);
    defer allocator.free(rendered);
    std.debug.print("{s}", .{rendered});
    return 0;
}

pub fn handlePluginRun(allocator: std.mem.Allocator, name: []const u8, input: []const u8) !u8 {
    var pm = abi.plugins.PluginManager.init(allocator);
    defer pm.deinit();

    // Load the known bundled plugins so run can find them. The list and the
    // tolerant load behavior are shared with the MCP surface so a plugin
    // runnable here is also runnable over MCP.
    abi.plugins.loadBundled(&pm);

    const output = try pm.run(allocator, name, input);
    defer allocator.free(output);

    std.debug.print("{s}\n", .{output});
    return 0;
}

fn pluginListHelp() u8 {
    std.debug.print(
        \\usage: abi plugin list
        \\
        \\Print the generated plugin registry with each installed plugin module.
        \\
    , .{});
    return 0;
}

fn pluginRunHelp() u8 {
    std.debug.print(
        \\usage: abi plugin run <name> [input]
        \\
        \\Run a bundled plugin by registry name with optional text input.
        \\
    , .{});
    return 0;
}

test "plugin dispatch rejects malformed grammar with exit code 2" {
    const allocator = std.testing.allocator;
    // Missing subcommand, unknown subcommand, and `run` without a name all reject
    // with usage (exit 2) before the plugin registry is loaded.
    try std.testing.expectEqual(@as(u8, 2), try handlePlugin(allocator, &.{ "abi", "plugin" }));
    try std.testing.expectEqual(@as(u8, 2), try handlePlugin(allocator, &.{ "abi", "plugin", "bogus" }));
    try std.testing.expectEqual(@as(u8, 2), try handlePlugin(allocator, &.{ "abi", "plugin", "run" }));
}

test "plugin handler help returns success before loading registry" {
    const allocator = std.testing.allocator;
    try std.testing.expectEqual(@as(u8, 0), try handlePlugin(allocator, &.{ "abi", "plugin", "--help" }));
    try std.testing.expectEqual(@as(u8, 0), try handlePlugin(allocator, &.{ "abi", "plugin", "list", "--help" }));
    try std.testing.expectEqual(@as(u8, 0), try handlePlugin(allocator, &.{ "abi", "plugin", "run", "-h" }));
}

test {
    std.testing.refAllDecls(@This());
}
