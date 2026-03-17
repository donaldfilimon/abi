//! ABI Server — Entry Point
//!
//! Starts the Abi framework REST API server with CLI argument parsing.
//! Usage: abi-server [--port PORT] [--host HOST] [--no-auth]

const std = @import("std");
const root = @import("root");

pub fn main(init: std.process.Init) !void {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const io = init.io;
    const arena = init.arena.allocator();

    // Parse CLI arguments.
    const args = try init.minimal.args.toSlice(arena);

    var config = root.server.Config{};
    var show_help = false;
    var show_version = false;

    var i: usize = 1; // skip program name
    while (i < args.len) : (i += 1) {
        const arg = args[i];
        if (std.mem.eql(u8, arg, "--help") or std.mem.eql(u8, arg, "-h")) {
            show_help = true;
        } else if (std.mem.eql(u8, arg, "--version") or std.mem.eql(u8, arg, "-v")) {
            show_version = true;
        } else if (std.mem.eql(u8, arg, "--port")) {
            i += 1;
            if (i >= args.len) {
                std.debug.print("Error: --port requires a value\n", .{});
                return;
            }
            config.port = std.fmt.parseInt(u16, args[i], 10) catch {
                std.debug.print("Error: Invalid port: {s}\n", .{args[i]});
                return;
            };
        } else if (std.mem.eql(u8, arg, "--host")) {
            i += 1;
            if (i >= args.len) {
                std.debug.print("Error: --host requires a value\n", .{});
                return;
            }
            config.host = args[i];
        } else if (std.mem.eql(u8, arg, "--no-auth")) {
            config.enable_auth = false;
        } else {
            std.debug.print("Error: Unknown argument: {s}\n", .{arg});
            printUsage();
            return;
        }
    }

    if (show_help) {
        printUsage();
        return;
    }

    if (show_version) {
        var stdout_buffer: [256]u8 = undefined;
        var stdout_writer: std.Io.File.Writer = .init(.stdout(), io, &stdout_buffer);
        const stdout = &stdout_writer.interface;
        try stdout.print("abi-server v{s}\n", .{root.version()});
        try stdout.flush();
        return;
    }

    // Warn if --no-auth is used with a non-localhost bind address.
    if (!config.enable_auth and !std.mem.eql(u8, config.host, "127.0.0.1") and !std.mem.eql(u8, config.host, "localhost")) {
        std.debug.print(
            \\WARNING: --no-auth is enabled with bind address '{s}'.
            \\  This exposes the API without authentication to the network.
            \\  Consider using --host 127.0.0.1 for local-only access.
            \\
        , .{config.host});
    }

    std.debug.print(
        \\
        \\  ABI Framework v{s}
        \\  Abbey / Aviva / Abi
        \\
        \\  Starting server on {s}:{d}...
        \\  Auth: {s}
        \\
    , .{ root.version(), config.host, config.port, if (config.enable_auth) "enabled" else "disabled" });

    var srv = root.Server.init(allocator, config);
    defer srv.deinit();

    try srv.start();
}

fn printUsage() void {
    std.debug.print(
        \\Usage: abi-server [OPTIONS]
        \\
        \\Options:
        \\  --port PORT      Server port (default: 8080)
        \\  --host HOST      Bind address (default: 127.0.0.1)
        \\  --no-auth        Disable API key authentication
        \\  --version, -v    Show version
        \\  --help, -h       Show this help
        \\
    , .{});
}
