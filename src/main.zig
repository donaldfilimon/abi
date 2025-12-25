const std = @import("std");
const abi = @import("abi");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var framework = try abi.init(allocator, abi.FrameworkOptions{});
    defer abi.shutdown(&framework);

    const args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, args);

    if (args.len <= 1) {
        printHelp();
        return;
    }

    const command = args[1];
    if (std.mem.eql(u8, command, "help") or std.mem.eql(u8, command, "--help") or
        std.mem.eql(u8, command, "-h"))
    {
        printHelp();
        return;
    }

    if (std.mem.eql(u8, command, "version") or std.mem.eql(u8, command, "--version") or
        std.mem.eql(u8, command, "-v"))
    {
        std.debug.print("ABI Framework v{s}\n", .{abi.version()});
        return;
    }

    if (std.mem.eql(u8, command, "db")) {
        try runDb(allocator, args[2..]);
        return;
    }

    if (std.mem.eql(u8, command, "agent")) {
        try runAgent(allocator, args[2..]);
        return;
    }

    if (std.mem.eql(u8, command, "system-info")) {
        try runSystemInfo(&framework);
        return;
    }

    std.debug.print("Unknown command: {s}\nUse 'help' for usage.\n", .{command});
    std.process.exit(1);
}

fn printHelp() void {
    const help_text =
        "Usage: abi <command> [options]\n\n" ++
        "Commands:\n" ++
        "  db <subcommand>   Database operations (add, query, stats, serve)\n" ++
        "  agent [--message]  Run AI agent (interactive or one-shot)\n" ++
        "  system-info       Show system and GPU information\n" ++
        "  version           Show framework version\n" ++
        "  help              Show this help message\n\n" ++
        "Run 'abi db help' for database specific commands.\n";

    std.debug.print("{s}", .{help_text});
}

fn runSystemInfo(framework: *abi.Framework) !void {
    const platform = abi.platform.platform;
    const info = platform.PlatformInfo.detect();

    std.debug.print("System Info\n", .{});
    std.debug.print("  OS: {s}\n", .{@tagName(info.os)});
    std.debug.print("  Arch: {s}\n", .{@tagName(info.arch)});
    std.debug.print("  CPU Threads: {d}\n", .{info.max_threads});
    std.debug.print("  ABI Version: {s}\n", .{abi.version()});

    std.debug.print("\nFeature Matrix:\n", .{});
    inline for (std.enums.values(abi.Feature)) |tag| {
        const status = if (framework.isFeatureEnabled(tag)) "enabled" else "disabled";
        std.debug.print("  {s}: {s}\n", .{ @tagName(tag), status });
    }
}

fn runDb(allocator: std.mem.Allocator, args: []const [:0]u8) !void {
    try abi.database.cli.run(allocator, args);
}

fn runAgent(allocator: std.mem.Allocator, args: []const [:0]u8) !void {
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

        if (std.mem.eql(u8, arg, "--message") or std.mem.eql(u8, arg, "-m")) {
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

    try runAgentInteractive(allocator, &agent);
}

fn runAgentInteractive(allocator: std.mem.Allocator, agent: *abi.ai.agent.Agent) !void {
    std.debug.print("Interactive mode. Type 'exit' to quit.\n", .{});
    var io_backend = std.Io.Threaded.init(allocator);
    defer io_backend.deinit();

    const io = io_backend.io();
    var stdin_file = std.fs.File.stdin();
    var buffer: [4096]u8 = undefined;
    var reader = stdin_file.reader(io, &buffer);

    while (true) {
        std.debug.print("> ", .{});
        const line_opt = reader.takeDelimiter('\n') catch |err| switch (err) {
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
