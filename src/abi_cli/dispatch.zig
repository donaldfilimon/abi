const std = @import("std");
const usage_mod = @import("usage.zig");
const handlers = @import("handlers/mod.zig");

const CommandHandler = *const fn (std.Io, std.mem.Allocator, []const []const u8) anyerror!u8;

const DispatchEntry = struct {
    name: []const u8,
    handler: CommandHandler,
};

fn handleTrainWrapper(io: std.Io, alloc: std.mem.Allocator, args: []const []const u8) anyerror!u8 {
    _ = io;
    if (args.len != 3) return usage_mod.usageError("usage: abi train <input>");
    return handlers.handleTrain(alloc, args[2]);
}

fn handleCompleteWrapper(io: std.Io, alloc: std.mem.Allocator, args: []const []const u8) anyerror!u8 {
    _ = io;
    if (args.len != 3) return usage_mod.usageError("usage: abi complete <input>");
    return handlers.handleComplete(alloc, args[2]);
}

fn handleBackendsWrapper(io: std.Io, alloc: std.mem.Allocator, args: []const []const u8) anyerror!u8 {
    _ = io;
    _ = alloc;
    if (args.len != 2) return usage_mod.usageError("usage: abi backends");
    return handlers.handleBackends();
}

fn handlePluginWrapper(io: std.Io, alloc: std.mem.Allocator, args: []const []const u8) anyerror!u8 {
    _ = io;
    return handlers.handlePlugin(alloc, args);
}

fn handleTwilioWrapper(io: std.Io, alloc: std.mem.Allocator, args: []const []const u8) anyerror!u8 {
    _ = io;
    return handlers.handleTwilio(alloc, args);
}

fn handleTuiWrapper(io: std.Io, alloc: std.mem.Allocator, args: []const []const u8) anyerror!u8 {
    _ = io;
    if (args.len != 2) return usage_mod.usageError("usage: abi tui");
    return handlers.handleDashboard(alloc);
}

fn handleDashboardWrapper(io: std.Io, alloc: std.mem.Allocator, args: []const []const u8) anyerror!u8 {
    _ = io;
    if (args.len != 2) return usage_mod.usageError("usage: abi dashboard");
    return handlers.handleDashboard(alloc);
}

const dispatch_table = [_]DispatchEntry{
    .{ .name = "train", .handler = handleTrainWrapper },
    .{ .name = "complete", .handler = handleCompleteWrapper },
    .{ .name = "agent", .handler = handlers.handleAgent },
    .{ .name = "backends", .handler = handleBackendsWrapper },
    .{ .name = "plugin", .handler = handlePluginWrapper },
    .{ .name = "auth", .handler = handlers.handleAuth },
    .{ .name = "twilio", .handler = handleTwilioWrapper },
    .{ .name = "tui", .handler = handleTuiWrapper },
    .{ .name = "dashboard", .handler = handleDashboardWrapper },
    .{ .name = "wdbx", .handler = handlers.handleWdbx },
};

pub fn runCli(io: std.Io, allocator: std.mem.Allocator, args: []const []const u8) !u8 {
    if (args.len < 2) {
        usage_mod.printUsage();
        return 0;
    }

    const cmd = args[1];

    if (std.mem.eql(u8, cmd, "help") or std.mem.eql(u8, cmd, "--help") or std.mem.eql(u8, cmd, "-h")) {
        if (args.len >= 3) return usage_mod.printCommandHelp(args[2]);
        usage_mod.printUsage();
        return 0;
    }

    for (dispatch_table) |entry| {
        if (std.mem.eql(u8, cmd, entry.name)) {
            return entry.handler(io, allocator, args);
        }
    }

    std.debug.print("error: unknown command '{s}'\n\n", .{cmd});
    usage_mod.printUsage();
    return 2;
}
