//! AI agent command.

const std = @import("std");
const abi = @import("abi");
const utils = @import("../utils/mod.zig");

/// Run the agent command with the provided arguments.
pub fn run(allocator: std.mem.Allocator, args: []const [:0]const u8) !void {
    const agent_mod = abi.ai.agent;

    var name: []const u8 = "cli-agent";
    var message: ?[]const u8 = null;

    var i: usize = 0;
    while (i < args.len) {
        const arg = args[i];
        i += 1;

        if (std.mem.eql(u8, arg, "--name")) {
            if (i < args.len) {
                name = args[i];
                i += 1;
            }
            continue;
        }

        if (utils.args.matchesAny(arg, &[_][]const u8{ "--message", "-m" })) {
            if (i < args.len) {
                message = args[i];
                i += 1;
            }
            continue;
        }
    }

    var agent = try agent_mod.Agent.init(allocator, .{ .name = name });
    defer agent.deinit();

    if (message) |msg| {
        const response = try agent.process(msg, allocator);
        defer allocator.free(response);
        std.debug.print("User: {s}\n", .{msg});
        std.debug.print("Agent: {s}\n", .{response});
        return;
    }

    try runInteractive(allocator, &agent);
}

fn runInteractive(allocator: std.mem.Allocator, agent: *abi.ai.agent.Agent) !void {
    std.debug.print("Interactive mode. Type 'exit' to quit.\n", .{});
    var io_backend = std.Io.Threaded.init(allocator, .{
        .environ = std.process.Environ.empty,
    });
    defer io_backend.deinit();

    const io = io_backend.io();
    var stdin_file = std.Io.File.stdin();
    var buffer: [4096]u8 = undefined;
    var reader = stdin_file.reader(io, &buffer);

    while (true) {
        std.debug.print("> ", .{});
        const line_opt = reader.interface.takeDelimiter('\n') catch |err| switch (err) {
            error.ReadFailed => return err,
            error.StreamTooLong => {
                std.debug.print("Input too long. Try a shorter line.\n", .{});
                continue;
            },
        };
        const line = line_opt orelse break;
        const trimmed = std.mem.trim(u8, line, " \t\r\n");
        if (trimmed.len == 0) continue;
        if (std.mem.eql(u8, trimmed, "exit") or std.mem.eql(u8, trimmed, "quit")) {
            break;
        }
        const response = try agent.process(trimmed, allocator);
        defer allocator.free(response);
        std.debug.print("Agent: {s}\n", .{response});
    }
}
