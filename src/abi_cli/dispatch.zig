const std = @import("std");
const usage_mod = @import("usage.zig");
const handlers = @import("handlers.zig");

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
    } else if (std.mem.eql(u8, cmd, "train")) {
        if (args.len != 3) return usage_mod.usageError("usage: abi train <input>");
        return handlers.handleTrain(allocator, args[2]);
    } else if (std.mem.eql(u8, cmd, "agent")) {
        return handlers.handleAgent(io, allocator, args);
    } else if (std.mem.eql(u8, cmd, "backends")) {
        if (args.len != 2) return usage_mod.usageError("usage: abi backends");
        return handlers.handleBackends();
    } else if (std.mem.eql(u8, cmd, "plugin")) {
        return handlers.handlePlugin(allocator, args);
    } else if (std.mem.eql(u8, cmd, "auth")) {
        return handlers.handleAuth(io, allocator, args);
    } else if (std.mem.eql(u8, cmd, "twilio")) {
        return handlers.handleTwilio(allocator, args);
    } else if (std.mem.eql(u8, cmd, "tui")) {
        if (args.len != 2) return usage_mod.usageError("usage: abi tui");
        return handlers.renderTui(allocator);
    } else {
        std.debug.print("error: unknown command '{s}'\n\n", .{cmd});
        usage_mod.printUsage();
        return 2;
    }
}
