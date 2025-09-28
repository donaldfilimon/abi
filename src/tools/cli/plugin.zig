const std = @import("std");
const plugins = @import("plugins");
const common = @import("common.zig");

pub const command = common.Command{
    .name = "plugin",
    .summary = "Manage ABI plugin registry",
    .usage = "abi plugin <list|load|info|call> [args...]",
    .details =
    "  list            Show registered plugins\n" ++
    "  load <path>     Load plugin from path\n" ++
    "  info <name>     Display plugin information\n" ++
    "  call <name> <function> Invoke plugin function\n",
    .run = run,
};

pub fn run(ctx: *common.Context, args: [][:0]u8) !void {
    const allocator = ctx.allocator;
    if (args.len < 3) {
        std.debug.print("Usage: {s}\n{s}", .{ command.usage, command.details orelse "" });
        return;
    }

    const sub = args[2];
    if (std.mem.eql(u8, sub, "list")) {
        var registry = try plugins.createRegistry(allocator);
        defer registry.deinit();

        std.debug.print("Plugin Registry:\n", .{});
        std.debug.print("  Status: Active\n", .{});
        std.debug.print("  Loaded Plugins: 0\n", .{});
        std.debug.print("  Available Plugins: Check plugin directory\n", .{});
        return;
    }

    if (std.mem.eql(u8, sub, "load")) {
        if (args.len < 4) {
            std.debug.print("plugin load requires <path>\n", .{});
            return;
        }

        const path = args[3];
        var registry = try plugins.createRegistry(allocator);
        defer registry.deinit();

        try registry.loadPlugin(path);
        std.debug.print("Plugin loaded from: {s}\n", .{path});
        return;
    }

    if (std.mem.eql(u8, sub, "info")) {
        if (args.len < 4) {
            std.debug.print("plugin info requires <name>\n", .{});
            return;
        }

        const plugin_name = args[3];
        var registry = try plugins.createRegistry(allocator);
        defer registry.deinit();

        std.debug.print("Plugin '{s}' info:\n", .{plugin_name});
        std.debug.print("  Status: Not loaded (plugin system in development)\n", .{});
        std.debug.print("  Type: Unknown\n", .{});
        std.debug.print("  Version: N/A\n", .{});
        return;
    }

    if (std.mem.eql(u8, sub, "call")) {
        if (args.len < 5) {
            std.debug.print("plugin call requires <name> <function> [args...]\n", .{});
            return;
        }

        const name = args[3];
        const function = args[4];

        var registry = try plugins.createRegistry(allocator);
        defer registry.deinit();

        if (registry.getPlugin(name)) |_| {
            std.debug.print("Calling {s}.{s}()...\n", .{ name, function });
            std.debug.print("Plugin '{s}' calling function '{s}':\n", .{ args[3], args[4] });
            std.debug.print("  Result: Plugin system in development\n", .{});
            std.debug.print("  Status: Function not executed\n", .{});
        } else {
            std.debug.print("Plugin '{s}' not found\n", .{name});
        }
        return;
    }

    std.debug.print("Unknown plugin subcommand: {s}\n", .{sub});
}
