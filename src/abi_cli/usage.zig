const std = @import("std");

pub const Command = struct {
    name: []const u8,
    usage: []const u8,
    summary: []const u8,
};

pub const commands = [_]Command{
    .{ .name = "help", .usage = "abi help [command]", .summary = "Show top-level or command-specific help" },
    .{ .name = "complete", .usage = "abi complete <input>", .summary = "Run local model completion with WDBX metadata" },
    .{ .name = "train", .usage = "abi train <input>", .summary = "Run the AI pipeline compatibility command" },
    .{ .name = "agent", .usage = "abi agent <plan|train|tui|os> ...", .summary = "Run safe agent planning, WDBX-backed local training, TUI, or OS dry-runs" },
    .{ .name = "backends", .usage = "abi backends", .summary = "Show GPU, accelerator, shader, and MLIR backend status" },
    .{ .name = "plugin", .usage = "abi plugin list | run <name> [input]", .summary = "Inspect or execute installed plugins" },
    .{ .name = "auth", .usage = "abi auth <signin|logout|status>", .summary = "Manage authentication for external services" },
    .{ .name = "twilio", .usage = "abi twilio simulate <input>", .summary = "Run a local Twilio voice-agent support simulation" },
    .{ .name = "tui", .usage = "abi tui", .summary = "Render the diagnostics dashboard" },
    .{ .name = "dashboard", .usage = "abi dashboard", .summary = "Render the diagnostics dashboard" },
};

pub fn printUsage() void {
    std.debug.print("Usage: abi <command> [args...]\n       abi --tui\n\nCommands:\n", .{});
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
