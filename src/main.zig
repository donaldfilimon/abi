//! Main CLI entrypoint for ABI Framework
//!
//! Provides command-line interface to access framework functionality.

const std = @import("std");
// Import the ABI module directly when building this file without a custom build script.
// Using the file name ensures the import works even when the project build does not
// provide a named module.
const abi = @import("abi.zig");
// Shared I/O backend helper (Zig 0.16)
// Note: This intentionally references ABI's shared module so `zig run src/main.zig`
// works without requiring the build system to define an "io" package/module.
const IoBackend = abi.shared.io.IoBackend;

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, args);

    // Skip program name
    const command = if (args.len > 1) args[1] else {
        printHelp();
        return;
    };

    if (std.mem.eql(u8, command, "help") or std.mem.eql(u8, command, "--help") or std.mem.eql(u8, command, "-h")) {
        printHelp();
        return;
    }

    if (std.mem.eql(u8, command, "version") or std.mem.eql(u8, command, "--version") or std.mem.eql(u8, command, "-v")) {
        std.debug.print("ABI Framework v{s}\n", .{abi.version()});
        return;
    }

    if (std.mem.eql(u8, command, "info") or std.mem.eql(u8, command, "system-info")) {
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
        \\  help, --help, -h     Show this help message
        \\  version, --version  Show framework version
        \\  info, system-info   Show framework information and available features
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

    // Initialise the shared I/O backend (Zig 0.16)
    var io_backend = try IoBackend.init(allocator);
    defer io_backend.deinit();

    // Build a fully‑featured framework using the builder pattern.
    var builder = abi.Framework.builder(allocator);
    var framework = builder
        .withGpuDefaults()
        .withAiDefaults()
        .withDatabaseDefaults()
        .withWebDefaults()
        .withNetworkDefaults()
        .withObservabilityDefaults()
        .withIo(io_backend.io)
        .build() catch |err| {
        std.debug.print("Framework initialization failed: {t}\n", .{err});
        std.debug.print("Running with minimal features...\n", .{});

        // Minimal framework – builder without any feature defaults.
        var minimal_framework = try abi.Framework.builder(allocator).build();
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
