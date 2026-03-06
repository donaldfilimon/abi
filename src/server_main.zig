//! ABI Server — Entry Point
//!
//! Starts the Abi framework REST API server with CLI argument parsing.
//! Usage: abi-server [--port PORT] [--host HOST] [--no-auth]

const std = @import("std");
const root = @import("root.zig");
const abi = @import("abi");

// Use the internal CLI DSL utilities.
const cli_utils = @import("cli_utils");
const ArgParser = cli_utils.args.ArgParser;
const output = cli_utils.output;

pub fn main(init: std.process.Init) !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Parse CLI arguments using the internal ArgParser.
    const proc_args = try init.args.toSlice(allocator);
    defer allocator.free(proc_args);

    var parser = ArgParser.init(allocator, proc_args);
    _ = parser.next(); // skip program name

    var config = root.server.Config{};

    while (parser.hasMore()) {
        if (parser.wantsHelp()) {
            printUsage();
            return;
        } else if (parser.matches(&[_][]const u8{ "--version", "-v" })) {
            output.println("abi-server v{s}", .{root.version()});
            return;
        } else if (parser.consumeOption(&[_][]const u8{"--port"})) |port_str| {
            config.port = std.fmt.parseInt(u16, port_str, 10) catch {
                output.printError("Invalid port: {s}", .{port_str});
                return;
            };
        } else if (parser.consumeOption(&[_][]const u8{"--host"})) |host| {
            config.host = host;
        } else if (parser.consumeFlag(&[_][]const u8{"--no-auth"})) {
            config.enable_auth = false;
        } else {
            const unknown = parser.next().?;
            output.printError("Unknown argument: {s}", .{unknown});
            printUsage();
            std.process.exit(1);
        }
    }

    output.print(
        \\
        \\  ╔══════════════════════════════════════╗
        \\  ║   ABI Framework v{s}              ║
        \\  ║   Abbey · Aviva · Abi                ║
        \\  ╚══════════════════════════════════════╝
        \\
        \\
    , .{root.version()});

    output.printInfo("Starting server on {s}:{d}...", .{ config.host, config.port });
    output.printInfo("Auth: {s}", .{if (config.enable_auth) "enabled" else "disabled"});

    var srv = root.Server.init(allocator, config);
    defer srv.deinit();

    try srv.start();
}

fn printUsage() void {
    output.print(
        \\Usage: abi-server [OPTIONS]
        \\
        \\Options:
        \\  --port PORT      Server port (default: 8080)
        \\  --host HOST      Bind address (default: 0.0.0.0)
        \\  --no-auth        Disable API key authentication
        \\  --version, -v    Show version
        \\  --help, -h       Show this help
        \\
    , .{});
}
