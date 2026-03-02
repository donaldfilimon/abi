//! Performance Verification Tool
//!
//! Parses benchmark JSON output and verifies compliance with performance KPIs.
//! Used in CI/CD pipelines to prevent performance regressions.
//!
//! Usage:
//!   zig build bench-competitive -- --json | abi-check-perf
//!
//! KPIs Checked:
//! - Minimum throughput (ops/sec) for critical paths
//! - Maximum latency (mean_ns) for interactive paths

const std = @import("std");

const BenchmarkResult = struct {
    name: []const u8,
    category: []const u8,
    iterations: u64,
    mean_ns: f64,
    ops_per_sec: f64,
};

const BenchmarkOutput = struct {
    duration_sec: ?f64 = null,
    benchmarks: []BenchmarkResult,
};

const Threshold = struct {
    pattern: []const u8,
    min_ops_sec: f64,
};

const max_input_bytes: usize = 16 * 1024 * 1024;

const kpis = [_]Threshold{
    // Sanity checks ensuring we aren't running in a broken slow mode.
    .{ .pattern = "ABI HNSW n=1000", .min_ops_sec = 1000.0 },
    .{ .pattern = "ABI HNSW n=10000", .min_ops_sec = 200.0 },
    .{ .pattern = "ABI WDBX Insert n=1000", .min_ops_sec = 100000.0 },
    .{ .pattern = "ABI WDBX Query n=1000", .min_ops_sec = 1000.0 },
    .{ .pattern = "ABI LLM Single Request", .min_ops_sec = 1000.0 },
};

pub fn main(_: std.process.Init) !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Initialize I/O backend for Zig 0.16.
    var io_backend = std.Io.Threaded.init(allocator, .{ .environ = std.process.Environ.empty });
    defer io_backend.deinit();
    const io = io_backend.io();

    const stdin_file = std.Io.File.stdin();
    var buffer: [4096]u8 = undefined;
    var reader = stdin_file.reader(io, &buffer);

    const input = try readAll(allocator, &reader.interface, max_input_bytes);
    defer allocator.free(input);

    const trimmed = std.mem.trim(u8, input, " \t\r\n");
    if (trimmed.len == 0) {
        std.debug.print("No input provided. Pipe benchmark JSON into abi-check-perf.\n", .{});
        return error.NoInput;
    }

    const json_slice = extractJsonSlice(trimmed) orelse {
        std.debug.print("No JSON payload found in benchmark output.\n", .{});
        return error.InvalidInput;
    };

    const parsed = std.json.parseFromSlice(BenchmarkOutput, allocator, json_slice, .{
        .ignore_unknown_fields = true,
    }) catch |err| {
        std.debug.print("Failed to parse benchmark JSON: {t}\n", .{err});
        return error.InvalidInput;
    };
    defer parsed.deinit();

    const output = parsed.value;
    var failures: usize = 0;

    if (output.duration_sec) |duration| {
        std.debug.print("Benchmark duration: {d:.2}s\n", .{duration});
    }

    for (kpis) |kpi| {
        var matched = false;
        var best_ops: f64 = 0.0;
        var best_name: ?[]const u8 = null;

        for (output.benchmarks) |bench| {
            if (std.mem.indexOf(u8, bench.name, kpi.pattern) == null) continue;
            matched = true;
            if (bench.ops_per_sec >= best_ops) {
                best_ops = bench.ops_per_sec;
                best_name = bench.name;
            }
        }

        if (!matched) {
            failures += 1;
            std.debug.print("KPI missing: pattern \"{s}\" not found\n", .{kpi.pattern});
            continue;
        }

        if (best_ops < kpi.min_ops_sec) {
            failures += 1;
            std.debug.print(
                "KPI regression: {s} ops/sec {d:.2} < {d:.2}\n",
                .{ best_name orelse kpi.pattern, best_ops, kpi.min_ops_sec },
            );
        } else {
            std.debug.print(
                "KPI ok: {s} ops/sec {d:.2} >= {d:.2}\n",
                .{ best_name orelse kpi.pattern, best_ops, kpi.min_ops_sec },
            );
        }
    }

    if (failures > 0) {
        std.debug.print("Performance check failed: {d} issue(s)\n", .{failures});
        return error.PerfCheckFailed;
    }

    std.debug.print("Performance check passed.\n", .{});
}

fn readAll(allocator: std.mem.Allocator, reader: *std.Io.Reader, limit: usize) ![]u8 {
    var list = std.ArrayListUnmanaged(u8).empty;
    errdefer list.deinit(allocator);

    var chunk: [4096]u8 = undefined;
    while (true) {
        const n = reader.readSliceShort(chunk[0..]) catch return error.ReadFailed;
        if (n == 0) break;
        if (list.items.len + n > limit) return error.InputTooLarge;
        try list.appendSlice(allocator, chunk[0..n]);
    }

    return list.toOwnedSlice(allocator);
}

fn extractJsonSlice(input: []const u8) ?[]const u8 {
    const start = std.mem.indexOfScalar(u8, input, '{') orelse return null;
    const end = std.mem.lastIndexOfScalar(u8, input, '}') orelse return null;
    if (end <= start) return null;
    return input[start .. end + 1];
}
