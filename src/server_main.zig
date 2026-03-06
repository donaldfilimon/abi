//! ABI Server — Entry Point
//!
//! Starts the Abi framework REST API server with CLI argument parsing.
//! Usage: abi-server [--port PORT] [--host HOST] [--no-auth]

const std = @import("std");
const root = @import("root.zig");

pub fn main(init: std.process.Init) !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Parse CLI arguments.
    var args = try std.process.Args.Iterator.initAllocator(init.minimal.args, allocator);
    defer args.deinit();

    var config = root.server.Config{};

    _ = args.next(); // skip program name
    while (args.next()) |arg| {
        if (std.mem.eql(u8, arg, "--port")) {
            if (args.next()) |port_str| {
                config.port = std.fmt.parseInt(u16, port_str, 10) catch {
                    std.debug.print("Invalid port: {s}\n", .{port_str});
                    return;
                };
            }
        } else if (std.mem.eql(u8, arg, "--host")) {
            if (args.next()) |host| {
                config.host = host;
            }
        } else if (std.mem.eql(u8, arg, "--no-auth")) {
            config.enable_auth = false;
        } else if (std.mem.eql(u8, arg, "--help") or std.mem.eql(u8, arg, "-h")) {
            printUsage();
            return;
        } else if (std.mem.eql(u8, arg, "--version") or std.mem.eql(u8, arg, "-v")) {
            std.debug.print("abi-server v{s}\n", .{root.version()});
            return;
        }
    }

    std.debug.print(
        \\
        \\  ╔══════════════════════════════════════╗
        \\  ║   ABI Framework v{s}              ║
        \\  ║   Abbey · Aviva · Abi                ║
        \\  ╚══════════════════════════════════════╝
        \\
        \\  Starting server on {s}:{d}...
        \\  Auth: {s}
        \\
    , .{
        root.version(),
        config.host,
        config.port,
        if (config.enable_auth) "enabled" else "disabled",
    });

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
        \\  --host HOST      Bind address (default: 0.0.0.0)
        \\  --no-auth        Disable API key authentication
        \\  --version, -v    Show version
        \\  --help, -h       Show this help
        \\
    , .{});
}
