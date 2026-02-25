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
    const warmup: u64 = 100;
    const iterations: u64 = 10000;

    // Warmup
    var w: u64 = 0;
    while (w < warmup) : (w += 1) {
        const result = @call(.auto, op, args);
        std.mem.doNotOptimizeAway(&result);
    }

    // Collect per-iteration samples for real percentile
    const samples = try allocator.alloc(u64, iterations);
    defer allocator.free(samples);

    var total_ns: u64 = 0;
    var i: u64 = 0;
    while (i < iterations) : (i += 1) {
        const iter_timer = abi.shared.time.Timer.start() catch return BenchmarkResult{
            .ops_per_sec = 0,
            .mean_ns = 0,
            .p99_ns = 0,
            .iterations = 0,
        };
        const result = @call(.auto, op, args);
        std.mem.doNotOptimizeAway(&result);
        var it = iter_timer;
        const elapsed = it.read();
        samples[i] = elapsed;
        total_ns += elapsed;
    }

    const mean_ns = @as(f64, @floatFromInt(total_ns)) / @as(f64, @floatFromInt(iterations));
    const ops_per_sec = if (mean_ns > 0) 1_000_000_000.0 / mean_ns else 0;

    // Real p99: sort samples and pick 99th percentile
    std.mem.sort(u64, samples, {}, std.sort.asc(u64));
    const p99_idx = @min(iterations - 1, (iterations * 99) / 100);

    return .{
        .ops_per_sec = ops_per_sec,
        .mean_ns = mean_ns,
        .p99_ns = samples[p99_idx],
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

    // Collect per-iteration samples for real percentile
    const samples = try allocator.alloc(u64, iterations);
    defer allocator.free(samples);

    var total_ns: u64 = 0;
    var i: u64 = 0;
    while (i < iterations) : (i += 1) {
        const iter_timer = abi.shared.time.Timer.start() catch return BenchmarkResult{
            .ops_per_sec = 0,
            .mean_ns = 0,
            .p99_ns = 0,
            .iterations = 0,
        };
        const buf = try allocator.alloc(u8, size);
        allocator.free(buf);
        var it = iter_timer;
        const elapsed = it.read();
        samples[i] = elapsed;
        total_ns += elapsed;
    }

    const mean_ns = @as(f64, @floatFromInt(total_ns)) / @as(f64, @floatFromInt(iterations));
    const ops_per_sec = if (mean_ns > 0) 1_000_000_000.0 / mean_ns else 0;

    // Real p99: sort samples and pick 99th percentile
    std.mem.sort(u64, samples, {}, std.sort.asc(u64));
    const p99_idx = @min(iterations - 1, (iterations * 99) / 100);

    return .{
        .ops_per_sec = ops_per_sec,
        .mean_ns = mean_ns,
        .p99_ns = samples[p99_idx],
        .iterations = iterations,
    };
}

// =============================================================================
// Output Formatting
// =============================================================================

pub fn outputJson(allocator: std.mem.Allocator, results: []const mod.BenchResult, duration_sec: f64, output_file: ?[]const u8) !void {
    const JsonBenchmark = struct {
        name: []const u8,
        category: []const u8,
        ops_per_sec: f64,
        mean_ns: f64,
        iterations: u64,
    };

    const json_results = try allocator.alloc(JsonBenchmark, results.len);
    defer allocator.free(json_results);

    for (results, 0..) |result, idx| {
        json_results[idx] = .{
            .name = result.name,
            .category = result.category,
            .ops_per_sec = result.ops_per_sec,
            .mean_ns = result.mean_ns,
            .iterations = result.iterations,
        };
    }

    const JsonOutput = struct {
        duration_sec: f64,
        benchmarks: []const JsonBenchmark,
    };

    var json_writer: std.Io.Writer.Allocating = .init(allocator);
    defer json_writer.deinit();
    try std.json.Stringify.value(JsonOutput{
        .duration_sec = duration_sec,
        .benchmarks = json_results,
    }, .{ .whitespace = .indent_2 }, &json_writer.writer);
    try json_writer.writer.writeByte('\n');
    const json_text = try json_writer.toOwnedSlice();
    defer allocator.free(json_text);

    if (output_file) |path| {
        var io_backend = cli_io.initIoBackend(allocator);
        defer io_backend.deinit();
        const io = io_backend.io();

        const file = std.Io.Dir.cwd().createFile(io, path, .{ .truncate = true }) catch |err| {
            utils.output.printError("creating output file: {t}", .{err});
            return;
        };
        defer file.close(io);

        file.writeStreamingAll(io, json_text) catch |err| {
            utils.output.printError("writing to file: {t}", .{err});
            return;
        };
        utils.output.printSuccess("Results written to: {s}", .{path});
    } else {
        utils.output.print("{s}", .{json_text});
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
    utils.output.print("{s}", .{header});
}

pub fn printFooter(duration_sec: f64) void {
    utils.output.println("================================================================================", .{});
    utils.output.println(" BENCHMARK COMPLETE - Total time: {d:.2}s", .{duration_sec});
    utils.output.println("================================================================================", .{});
}
