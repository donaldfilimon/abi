//! Benchmark output formatting and utility types.
//!
//! Contains JSON output, header/footer printing, and benchmark helper
//! functions used by the suite runners.

const std = @import("std");
const abi = @import("abi");
const utils = @import("../../utils/mod.zig");
const cli_io = utils.io_backend;
const mod = @import("mod.zig");

/// Intermediate benchmark result from benchmarkOp / benchmarkAllocOp.
pub const BenchmarkResult = struct {
    ops_per_sec: f64,
    mean_ns: f64,
    p99_ns: u64,
    iterations: u64,
};

pub fn benchmarkOp(
    allocator: std.mem.Allocator,
    comptime op: anytype,
    args: anytype,
) !BenchmarkResult {
    _ = allocator;
    const warmup: u64 = 100;
    const iterations: u64 = 10000;

    // Warmup
    var w: u64 = 0;
    while (w < warmup) : (w += 1) {
        const result = @call(.auto, op, args);
        std.mem.doNotOptimizeAway(&result);
    }

    // Benchmark
    const timer = abi.shared.time.Timer.start() catch return BenchmarkResult{
        .ops_per_sec = 0,
        .mean_ns = 0,
        .p99_ns = 0,
        .iterations = 0,
    };

    var i: u64 = 0;
    while (i < iterations) : (i += 1) {
        const result = @call(.auto, op, args);
        std.mem.doNotOptimizeAway(&result);
    }

    var t = timer;
    const elapsed_ns = t.read();
    const mean_ns = @as(f64, @floatFromInt(elapsed_ns)) / @as(f64, @floatFromInt(iterations));
    const ops_per_sec = if (mean_ns > 0) 1_000_000_000.0 / mean_ns else 0;

    return .{
        .ops_per_sec = ops_per_sec,
        .mean_ns = mean_ns,
        .p99_ns = @intFromFloat(mean_ns * 1.5),
        .iterations = iterations,
    };
}

pub fn benchmarkAllocOp(allocator: std.mem.Allocator, size: usize) !BenchmarkResult {
    const warmup: u64 = 100;
    const iterations: u64 = 10000;

    // Warmup
    var w: u64 = 0;
    while (w < warmup) : (w += 1) {
        const buf = try allocator.alloc(u8, size);
        allocator.free(buf);
    }

    // Benchmark
    const timer = abi.shared.time.Timer.start() catch return BenchmarkResult{
        .ops_per_sec = 0,
        .mean_ns = 0,
        .p99_ns = 0,
        .iterations = 0,
    };

    var i: u64 = 0;
    while (i < iterations) : (i += 1) {
        const buf = try allocator.alloc(u8, size);
        allocator.free(buf);
    }

    var t = timer;
    const elapsed_ns = t.read();
    const mean_ns = @as(f64, @floatFromInt(elapsed_ns)) / @as(f64, @floatFromInt(iterations));
    const ops_per_sec = if (mean_ns > 0) 1_000_000_000.0 / mean_ns else 0;

    return .{
        .ops_per_sec = ops_per_sec,
        .mean_ns = mean_ns,
        .p99_ns = @intFromFloat(mean_ns * 1.5),
        .iterations = iterations,
    };
}

// =============================================================================
// Output Formatting
// =============================================================================

pub fn outputJson(allocator: std.mem.Allocator, results: []const mod.BenchResult, duration_sec: f64, output_file: ?[]const u8) !void {
    var json_buf: std.ArrayListUnmanaged(u8) = .empty;
    defer json_buf.deinit(allocator);

    try json_buf.appendSlice(allocator, "{\n  \"duration_sec\": ");
    var dur_buf: [32]u8 = undefined;
    const dur_str = std.fmt.bufPrint(&dur_buf, "{d:.2}", .{duration_sec}) catch "0";
    try json_buf.appendSlice(allocator, dur_str);
    try json_buf.appendSlice(allocator, ",\n  \"benchmarks\": [\n");

    for (results, 0..) |result, idx| {
        if (idx > 0) try json_buf.appendSlice(allocator, ",\n");
        try json_buf.appendSlice(allocator, "    {\"name\": \"");
        try json_buf.appendSlice(allocator, result.name);
        try json_buf.appendSlice(allocator, "\", \"category\": \"");
        try json_buf.appendSlice(allocator, result.category);
        try json_buf.appendSlice(allocator, "\", \"ops_per_sec\": ");

        var ops_buf: [32]u8 = undefined;
        const ops_str = std.fmt.bufPrint(&ops_buf, "{d:.2}", .{result.ops_per_sec}) catch "0";
        try json_buf.appendSlice(allocator, ops_str);

        try json_buf.appendSlice(allocator, ", \"mean_ns\": ");
        var mean_buf: [32]u8 = undefined;
        const mean_str = std.fmt.bufPrint(&mean_buf, "{d:.2}", .{result.mean_ns}) catch "0";
        try json_buf.appendSlice(allocator, mean_str);

        try json_buf.appendSlice(allocator, ", \"iterations\": ");
        var iter_buf: [32]u8 = undefined;
        const iter_str = std.fmt.bufPrint(&iter_buf, "{d}", .{result.iterations}) catch "0";
        try json_buf.appendSlice(allocator, iter_str);

        try json_buf.appendSlice(allocator, "}");
    }

    try json_buf.appendSlice(allocator, "\n  ]\n}\n");

    if (output_file) |path| {
        // Write to file
        var io_backend = cli_io.initIoBackend(allocator);
        defer io_backend.deinit();
        const io = io_backend.io();

        const file = std.Io.Dir.cwd().createFile(io, path, .{ .truncate = true }) catch |err| {
            std.debug.print("Error creating output file: {t}\n", .{err});
            return;
        };
        defer file.close(io);

        // Use writeStreamingAll for Zig 0.16 compatibility
        file.writeStreamingAll(io, json_buf.items) catch |err| {
            std.debug.print("Error writing to file: {t}\n", .{err});
            return;
        };
        std.debug.print("Results written to: {s}\n", .{path});
    } else {
        std.debug.print("{s}", .{json_buf.items});
    }
}

pub fn printHeader() void {
    const header =
        \\
        \\╔════════════════════════════════════════════════════════════════════════════╗
        \\║                     ABI FRAMEWORK BENCHMARK SUITE                          ║
        \\╚════════════════════════════════════════════════════════════════════════════╝
        \\
    ;
    std.debug.print("{s}", .{header});
}

pub fn printFooter(duration_sec: f64) void {
    std.debug.print("================================================================================\n", .{});
    std.debug.print(" BENCHMARK COMPLETE - Total time: {d:.2}s\n", .{duration_sec});
    std.debug.print("================================================================================\n", .{});
}
