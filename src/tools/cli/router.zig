const std = @import("std");
const common = @import("common.zig");
const registry = @import("registry.zig");

pub fn run(ctx: *common.Context, args: [][:0]u8) !void {
    if (args.len <= 1) {
        try printGlobalHelp();
        return;
    }

    const first = args[1];
    if (common.isHelpToken(first)) {
        try printGlobalHelp();
        return;
    }
    if (std.mem.eql(u8, first, "--version") or std.mem.eql(u8, first, "-v") or std.mem.eql(u8, first, "version")) {
        printVersion();
        return;
    }

    if (registry.find(first)) |command| {
        try command.run(ctx, args);
        return;
    }

    std.debug.print("Unknown command: {s}\n", .{first});
    std.debug.print("Use 'abi --help' for available commands.\n", .{});
    std.process.exit(1);
}

pub fn printGlobalHelp() !void {
    std.debug.print("{s} {s}\n", .{ common.CLI_NAME, common.CLI_VERSION });
    std.debug.print("Usage: abi <command> [options]\n\n", .{});
    std.debug.print("Available commands:\n", .{});

    // Print commands directly
    const commands = registry.all();
    var longest: usize = 0;
    for (commands) |cmd| {
        longest = @max(longest, cmd.name.len);
        for (cmd.aliases) |alias| {
            longest = @max(longest, alias.len);
        }
    }

    for (commands) |cmd| {
        std.debug.print("  {s}  {s}\n", .{ cmd.name, cmd.summary });
    }

    std.debug.print("\nUse 'abi <command> --help' for detailed information.\n", .{});
}
pub fn printVersion() void {
    std.debug.print("{s} {s}\n", .{ common.CLI_NAME, common.CLI_VERSION });
}
