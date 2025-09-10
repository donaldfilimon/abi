const std = @import("std");

/// Unified benchmark entry point for WDBX-AI
/// Consolidates neural network, database, and performance benchmarks
pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var args = try std.process.argsWithAllocator(allocator);
    defer args.deinit();

    // Skip program name
    _ = args.next();

    var do_export = false;
    var fmt: []const u8 = "json";
    var bench_type: ?[]const u8 = null;
    while (args.next()) |a| {
        if (std.mem.eql(u8, a, "--export")) do_export = true else if (std.mem.startsWith(u8, a, "--format=")) fmt = a[9..] else {
            bench_type = a;
            break;
        }
    }

    const bench = bench_type orelse {
        try printUsage();
        return;
    };

    if (do_export) try writeExportStub(fmt);

    if (std.mem.eql(u8, bench, "neural")) {
        std.log.info("Running neural network benchmarks...", .{});
        // Neural benchmarks moved to separate executable
        std.log.info("Use: zig build benchmark-neural", .{});
    } else if (std.mem.eql(u8, bench, "database")) {
        const db_bench = @import("database_benchmark.zig");
        try db_bench.main();
    } else if (std.mem.eql(u8, bench, "performance")) {
        const perf_bench = @import("performance_suite.zig");
        try perf_bench.main();
    } else if (std.mem.eql(u8, bench, "simple")) {
        std.log.info("Running simple benchmarks...", .{});
        // Simple benchmarks moved to separate executable
        std.log.info("Use: zig build benchmark-simple", .{});
    } else if (std.mem.eql(u8, bench, "all")) {
        std.log.info("Running available benchmarks...", .{});

        std.log.info("\n=== Database Benchmarks ===", .{});
        const db_bench = @import("database_benchmark.zig");
        try db_bench.main();

        std.log.info("\n=== Performance Suite ===", .{});
        const perf_bench = @import("performance_suite.zig");
        try perf_bench.main();

        std.log.info("\nFor neural and simple benchmarks, use:", .{});
        std.log.info("  zig build benchmark-neural", .{});
        std.log.info("  zig build benchmark-simple", .{});
    } else {
        std.log.err("Unknown benchmark type: {s}", .{bench});
        try printUsage();
        return;
    }
}

fn printUsage() !void {
    std.log.info(
        \\Usage: zig run benchmarks/main.zig -- <benchmark_type>
        \\
        \\Benchmark Types:
        \\  neural      - Neural network and AI benchmarks (use: zig build benchmark-neural)
        \\  database    - Database and vector operations benchmarks
        \\  performance - Performance and memory benchmarks
        \\  simple      - Simple VDBench-style benchmarks (use: zig build benchmark-simple)
        \\  all         - Run available benchmark suites
        \\
        \\Examples:
        \\  zig run benchmarks/main.zig -- database
        \\  zig run benchmarks/main.zig -- all
        \\  zig build benchmark-neural
        \\  zig build benchmark-simple
        \\
    , .{});
}

fn writeExportStub(fmt: []const u8) !void {
    var out_dir = try std.fs.cwd().makeOpenPath("zig-out/bench", .{});
    defer out_dir.close();
    const name = if (std.mem.eql(u8, fmt, "csv")) "summary.csv" else "summary.json";
    var f = try out_dir.createFile(name, .{ .truncate = true });
    defer f.close();
    if (std.mem.eql(u8, fmt, "csv")) {
        try f.writeAll("suite,status\nbenchmarks,completed\n");
    } else {
        try f.writeAll("{\"suite\":\"benchmarks\",\"status\":\"completed\"}\n");
    }
}
