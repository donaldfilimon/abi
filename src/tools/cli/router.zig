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

    // TODO: Format summary when registry.formatSummary is working
    std.debug.print("Available commands will be listed here\n", .{});

    std.debug.print("Use 'abi <command> --help' for detailed information.\n", .{});
}

pub fn printVersion() void {
    std.debug.print("{s} {s}\n", .{ common.CLI_NAME, common.CLI_VERSION });
}
