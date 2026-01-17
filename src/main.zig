//! Main CLI entrypoint for ABI Framework
//!
//! Provides command-line interface to access framework functionality.

const std = @import("std");
// Import the ABI module directly when building this file without a custom build script.
// Using the file name ensures the import works even when the project build does not
// provide a named module.
const abi = @import("abi.zig");

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;

    var args_iter = std.process.Args.Iterator.initAllocator(init.minimal.args, allocator) catch {
        printHelp();
        return;
    };
    defer args_iter.deinit();

    _ = args_iter.skip();
    const command = args_iter.next() orelse {
        printHelp();
        return;
    };

    if (std.mem.eql(u8, command, "help") or std.mem.eql(u8, command, "--help")) {
        printHelp();
        return;
    }

    if (std.mem.eql(u8, command, "version")) {
        std.debug.print("ABI Framework v{s}\n", .{abi.version()});
        return;
    }

    if (std.mem.eql(u8, command, "info")) {
        try printFrameworkInfo(allocator);
        return;
    }

    std.debug.print("Unknown command: {s}\n", .{command});
    printHelp();
}

fn printHelp() void {
    const help_text =
        \\ABI Framework CLI
        \\
        \\Usage:
        \\  abi <command> [options]
        \\
        \\Commands:
        \\  help, --help     Show this help message
        \\  version          Show framework version
        \\  info             Show framework information and available features
        \\
        \\For the full CLI (including tui), build the tools/cli entrypoint.
        \\For more advanced usage, see the examples/ directory.
    ;
    std.debug.print("{s}\n", .{help_text});
}

fn printFrameworkInfo(allocator: std.mem.Allocator) !void {
    std.debug.print("=== ABI Framework Information ===\n", .{});
    std.debug.print("Version: {s}\n", .{abi.version()});
    std.debug.print("SIMD Support: {s}\n", .{if (abi.hasSimdSupport()) "Yes" else "No"});

    var framework = abi.init(allocator, abi.FrameworkOptions{
        .enable_gpu = true,
        .enable_ai = true,
        .enable_database = true,
        .enable_web = true,
        .enable_network = true,
        .enable_profiling = true,
    }) catch |err| {
        std.debug.print("Framework initialization failed: {t}\n", .{err});
        std.debug.print("Running with minimal features...\n", .{});

        var minimal_framework = try abi.init(allocator, abi.FrameworkOptions{
            .enable_gpu = false,
            .enable_ai = false,
            .enable_database = false,
            .enable_web = false,
            .enable_network = false,
            .enable_profiling = false,
        });
        defer abi.shutdown(&minimal_framework);

        std.debug.print("Minimal framework initialized successfully\n", .{});
        return;
    };
    defer abi.shutdown(&framework);

    std.debug.print("Framework initialized successfully\n", .{});

    if (abi.database.isEnabled()) {
        std.debug.print("Database: Available\n", .{});
    } else {
        std.debug.print("Database: Not available (enable with -Denable-database=true)\n", .{});
    }

    if (abi.gpu.moduleEnabled()) {
        std.debug.print("GPU: Available\n", .{});
    } else {
        std.debug.print("GPU: Not available (enable with -Denable-gpu=true)\n", .{});
    }

    if (abi.ai.isEnabled()) {
        std.debug.print("AI: Available\n", .{});
    } else {
        std.debug.print("AI: Not available (enable with -Denable-ai=true)\n", .{});
    }

    if (abi.web.isEnabled()) {
        std.debug.print("Web: Available\n", .{});
    } else {
        std.debug.print("Web: Not available (enable with -Denable-web=true)\n", .{});
    }

    if (abi.network.isEnabled()) {
        std.debug.print("Network: Available\n", .{});
    } else {
        std.debug.print("Network: Not available (enable with -Denable-network=true)\n", .{});
    }
}
