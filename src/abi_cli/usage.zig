const std = @import("std");

pub const Command = struct {
    name: []const u8,
    usage: []const u8,
    summary: []const u8,
};

pub const commands = [_]Command{
    .{ .name = "help", .usage = "abi help [command]", .summary = "Show top-level or command-specific help" },
    .{ .name = "train", .usage = "abi train <input>", .summary = "Run the AI pipeline compatibility command" },
    .{ .name = "agent", .usage = "abi agent <plan|train|tui|os> ...", .summary = "Run safe agent planning, WDBX-backed local training, TUI, or OS dry-runs" },
    .{ .name = "backends", .usage = "abi backends", .summary = "Show GPU, accelerator, shader, and MLIR backend status" },
    .{ .name = "plugin", .usage = "abi plugin list", .summary = "Inspect installed plugins" },
    .{ .name = "tui", .usage = "abi tui", .summary = "Render a minimal terminal dashboard" },
};

pub fn printUsage() void {
    std.debug.print("Usage: abi <command> [args...] [--tui]\n\nCommands:\n", .{});
    for (commands) |command| {
        std.debug.print("  {s:<8} {s}\n", .{ command.name, command.summary });
    }
    std.debug.print("\nRun `abi help <command>` for details.\n", .{});
}

pub fn printCommandHelp(name: []const u8) u8 {
    for (commands) |command| {
        if (std.mem.eql(u8, command.name, name)) {
            std.debug.print("{s}\n\n{s}\n", .{ command.usage, command.summary });
            return 0;
        }
    }
    std.debug.print("error: unknown command '{s}'\n\n", .{name});
    printUsage();
    return 2;
}

pub fn usageError(message: []const u8) u8 {
    std.debug.print("error: {s}\n", .{message});
    return 2;
}
