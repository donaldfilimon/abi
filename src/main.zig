//! Main CLI entrypoint for ABI Framework
//!
//! Provides command-line interface to access framework functionality.

const std = @import("std");
const abi = @import("abi");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const args = try std.process.argsAlloc(allocator);
    defer allocator.free(args);

    if (args.len < 2) {
        printHelp();
        return;
    }

    const command = std.mem.sliceTo(args[1], 0);
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
        \\For more advanced usage, see the examples/ directory.
    ;
    std.debug.print("{s}\n", .{help_text});
}

fn printFrameworkInfo(allocator: std.mem.Allocator) !void {
    std.debug.print("=== ABI Framework Information ===\n", .{});
    std.debug.print("Version: {s}\n", .{abi.version()});
    std.debug.print("SIMD Support: {s}\n", .{if (abi.hasSimdSupport()) "Yes" else "No"});

    // Try to initialize with all features enabled
    var framework = abi.init(allocator, abi.FrameworkOptions{
        .enable_gpu = true,
        .enable_ai = true,
        .enable_database = true,
        .enable_web = true,
        .enable_network = true,
        .enable_profiling = true,
    }) catch |err| {
        std.debug.print("Framework initialization failed: {s}\n", .{@errorName(err)});
        std.debug.print("Running with minimal features...\n", .{});

        // Try minimal initialization
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

    std.debug.print("Framework: Initialized successfully\n", .{});
    std.debug.print("GPU Module: {s}\n", .{if (abi.gpu.moduleEnabled()) "Enabled" else "Disabled"});
    std.debug.print("AI Module: {s}\n", .{if (abi.ai.isEnabled()) "Enabled" else "Disabled"});
    std.debug.print("Database Module: {s}\n", .{if (abi.database.isEnabled()) "Enabled" else "Disabled"});
    std.debug.print("Web Module: {s}\n", .{if (abi.web.isEnabled()) "Enabled" else "Disabled"});
    std.debug.print("Network Module: {s}\n", .{if (abi.network.isEnabled()) "Enabled" else "Disabled"});
}
