//! Enhanced Unified Benchmark Entry Point
//!
//! This is the main entry point for all ABI benchmarks:
//! - Neural network and AI benchmarks
//! - Database and vector operations benchmarks
//! - Performance and SIMD benchmarks
//! - Export capabilities for CI/CD integration

const std = @import("std");
const builtin = @import("builtin");

const framework = @import("benchmark_framework.zig");

const separator_line =
    \\============================================================
;

/// Unified benchmark entry point for ABI
/// Consolidates all benchmark suites with enhanced reporting
pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var args = try std.process.argsWithAllocator(allocator);
    defer args.deinit();

    // Skip program name
    _ = args.next();

    var do_export = false;
    var fmt: framework.BenchmarkConfig.OutputFormat = .console;
    var bench_type: ?[]const u8 = null;
    var export_path: []const u8 = "benchmark_results";

    while (args.next()) |a| {
        if (std.mem.eql(u8, a, "--export")) {
            do_export = true;
        } else if (std.mem.startsWith(u8, a, "--format=")) {
            const format_str = a[9..];
            fmt = if (std.mem.eql(u8, format_str, "json")) .json else if (std.mem.eql(u8, format_str, "csv")) .csv else if (std.mem.eql(u8, format_str, "markdown")) .markdown else .console;
        } else if (std.mem.startsWith(u8, a, "--output=")) {
            export_path = a[9..];
        } else {
            bench_type = a;
        }
    }

    const bench = bench_type orelse "all";

    std.log.info("🚀 ABI Enhanced Benchmark Suite", .{});
    std.log.info("================================", .{});

    if (std.mem.eql(u8, bench, "neural") or std.mem.eql(u8, bench, "ai")) {
        std.log.info("🧠 Running AI/Neural Network Benchmarks", .{});
        const benchmark_suite = @import("benchmark_suite.zig");
        try benchmark_suite.main();
    } else if (std.mem.eql(u8, bench, "database") or std.mem.eql(u8, bench, "db")) {
        std.log.info("🗄️ Running Database Benchmarks", .{});
        const db_bench = @import("database_benchmark.zig");
        try db_bench.main();
    } else if (std.mem.eql(u8, bench, "performance") or std.mem.eql(u8, bench, "perf")) {
        std.log.info("⚡ Running Performance Benchmarks", .{});
        const perf_bench = @import("performance_suite.zig");
        try perf_bench.main();
    } else if (std.mem.eql(u8, bench, "simd")) {
        std.log.info("📐 Running SIMD Micro-benchmarks", .{});
        const simd_bench = @import("simd_micro.zig");
        try simd_bench.main();
    } else if (std.mem.eql(u8, bench, "all")) {
        std.log.info("🎯 Running All Available Benchmarks", .{});

        std.log.info("\n{s}", .{separator_line});
        std.log.info("🧠 AI/NEURAL NETWORK BENCHMARKS", .{});
        std.log.info("{s}", .{separator_line});
        const benchmark_suite = @import("benchmark_suite.zig");
        try benchmark_suite.main();

        std.log.info("\n{s}", .{separator_line});
        std.log.info("🗄️ DATABASE BENCHMARKS", .{});
        std.log.info("{s}", .{separator_line});
        const db_bench = @import("database_benchmark.zig");
        try db_bench.main();

        std.log.info("\n{s}", .{separator_line});
        std.log.info("⚡ PERFORMANCE BENCHMARKS", .{});
        std.log.info("{s}", .{separator_line});
        const perf_bench = @import("performance_suite.zig");
        try perf_bench.main();

        std.log.info("\n{s}", .{separator_line});
        std.log.info("📐 SIMD MICRO-BENCHMARKS", .{});
        std.log.info("{s}", .{separator_line});
        const simd_bench = @import("simd_micro.zig");
        try simd_bench.main();

        std.log.info("\n✅ All benchmarks completed successfully!", .{});
    } else {
        std.log.err("❌ Unknown benchmark type: {s}", .{bench});
        try printUsage();
        return;
    }

    if (do_export) {
        std.log.info("📊 Exporting results to {s}", .{export_path});
        try exportBenchmarkResults(fmt, export_path);
    }
}

fn printUsage() !void {
    std.log.info(
        \\Usage: zig run benchmarks/main.zig -- <benchmark_type> [options]
        \\
        \\Benchmark Types:
        \\  neural/ai   - Neural network and AI benchmarks
        \\  database/db - Database and vector operations benchmarks
        \\  performance/perf - Performance and memory benchmarks
        \\  simd        - SIMD micro-benchmarks
        \\  all         - Run all available benchmark suites (default)
        \\
        \\Options:
        \\  --export    - Export results to file
        \\  --format=<fmt> - Output format: console, json, csv, markdown
        \\  --output=<path> - Output file path (default: benchmark_results)
        \\
        \\Examples:
        \\  zig run benchmarks/main.zig -- database
        \\  zig run benchmarks/main.zig -- all
        \\  zig run benchmarks/main.zig -- --export --format=json all
        \\  zig run benchmarks/main.zig -- --export --format=csv --output=results.csv perf
        \\
    , .{});
}

fn exportBenchmarkResults(format: framework.BenchmarkConfig.OutputFormat, path: []const u8) !void {
    var out_dir = try std.fs.cwd().makeOpenPath("zig-out/bench", .{});
    defer out_dir.close();

    const extension = switch (format) {
        .json => ".json",
        .csv => ".csv",
        .markdown => ".md",
        .console => ".txt",
    };

    const filename = try std.fmt.allocPrint(std.heap.page_allocator, "{s}{s}", .{ path, extension });
    defer std.heap.page_allocator.free(filename);

    var file = try out_dir.createFile(filename, .{ .truncate = true });
    defer file.close();

    switch (format) {
        .json => {
            try file.writeAll("{\n");
            try file.writeAll("  \"suite\": \"abi_benchmarks\",\n");
            try file.writeAll("  \"status\": \"completed\",\n");
            try file.writeAll("  \"timestamp\": \"");
            const timestamp = 0;
            try file.writeAll(try std.fmt.allocPrint(std.heap.page_allocator, "{d}", .{timestamp}));
            try file.writeAll("\",\n");
            try file.writeAll("  \"platform\": {\n");
            try file.writeAll("    \"os\": \"");
            try file.writeAll(@tagName(builtin.target.os.tag));
            try file.writeAll("\",\n");
            try file.writeAll("    \"arch\": \"");
            try file.writeAll(@tagName(builtin.target.cpu.arch));
            try file.writeAll("\"\n");
            try file.writeAll("  }\n");
            try file.writeAll("}\n");
        },
        .csv => {
            try file.writeAll("suite,status,timestamp,os,arch\n");
            try file.writeAll("abi_benchmarks,completed,");
            try file.writeAll(try std.fmt.allocPrint(std.heap.page_allocator, "{d}", .{0}));
            try file.writeAll(",");
            try file.writeAll(@tagName(builtin.target.os.tag));
            try file.writeAll(",");
            try file.writeAll(@tagName(builtin.target.cpu.arch));
            try file.writeAll("\n");
        },
        .markdown => {
            try file.writeAll("# ABI Benchmark Results\n\n");
            try file.writeAll("**Status:** Completed\n");
            try file.writeAll("**Platform:** ");
            try file.writeAll(@tagName(builtin.target.os.tag));
            try file.writeAll(" ");
            try file.writeAll(@tagName(builtin.target.cpu.arch));
            try file.writeAll("\n");
            try file.writeAll("**Timestamp:** ");
            try file.writeAll(try std.fmt.allocPrint(std.heap.page_allocator, "{d}", .{0}));
            try file.writeAll("\n\n");
            try file.writeAll("All benchmarks completed successfully!\n");
        },
        .console => {
            try file.writeAll("ABI Benchmark Results\n");
            try file.writeAll("====================\n\n");
            try file.writeAll("Status: Completed\n");
            try file.writeAll("Platform: ");
            try file.writeAll(@tagName(builtin.target.os.tag));
            try file.writeAll(" ");
            try file.writeAll(@tagName(builtin.target.cpu.arch));
            try file.writeAll("\n");
            try file.writeAll("Timestamp: ");
            try file.writeAll(try std.fmt.allocPrint(std.heap.page_allocator, "{d}", .{0}));
            try file.writeAll("\n\n");
            try file.writeAll("All benchmarks completed successfully!\n");
        },
    }
}
