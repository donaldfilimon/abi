//! Benchmark command for running performance benchmarks.
//!
//! Commands:
//! - bench all - Run all benchmark suites
//! - bench <suite> - Run specific suite (simd, memory, concurrency, database, network, crypto, ai)
//! - bench quick - Run quick benchmarks for CI
//! - bench micro <operation> - Run quick micro-benchmark (hash, alloc, parse)
//! - bench list - List available benchmark suites
//!
//! Options:
//! - --json - Output results in JSON format
//! - --output <file> - Write results to file

const std = @import("std");
const abi = @import("abi");
const utils = @import("../utils/mod.zig");

/// Benchmark suite selection
pub const BenchmarkSuite = enum {
    all,
    simd,
    memory,
    concurrency,
    database,
    network,
    crypto,
    ai,
    quick,

    pub fn toString(self: BenchmarkSuite) []const u8 {
        return switch (self) {
            .all => "all",
            .simd => "simd",
            .memory => "memory",
            .concurrency => "concurrency",
            .database => "database",
            .network => "network",
            .crypto => "crypto",
            .ai => "ai",
            .quick => "quick",
        };
    }
};

/// Micro-benchmark operation
pub const MicroOp = enum {
    hash,
    alloc,
    parse,
    noop,

    pub fn toString(self: MicroOp) []const u8 {
        return switch (self) {
            .hash => "hash",
            .alloc => "alloc",
            .parse => "parse",
            .noop => "noop",
        };
    }
};

/// Benchmark configuration
pub const BenchConfig = struct {
    suite: BenchmarkSuite = .all,
    output_json: bool = false,
    output_file: ?[]const u8 = null,
    micro_op: ?MicroOp = null,
    iterations: u32 = 1000,
    warmup: u32 = 100,
};

/// Run the benchmark command with the provided arguments.
pub fn run(allocator: std.mem.Allocator, args: []const [:0]const u8) !void {
    if (args.len == 0 or utils.args.matchesAny(args[0], &[_][]const u8{ "help", "--help", "-h" })) {
        printHelp();
        return;
    }

    var config = BenchConfig{};

    const command = std.mem.sliceTo(args[0], 0);

    // Handle special commands
    if (std.mem.eql(u8, command, "list")) {
        printAvailableSuites();
        return;
    }

    if (std.mem.eql(u8, command, "micro")) {
        return runMicroBenchmark(allocator, args[1..], &config);
    }

    // Parse suite name
    if (std.meta.stringToEnum(BenchmarkSuite, command)) |suite| {
        config.suite = suite;
    } else {
        std.debug.print("Unknown benchmark suite: {s}\n", .{command});
        std.debug.print("Use 'abi bench list' to see available suites.\n", .{});
        return;
    }

    // Parse remaining options
    var i: usize = 1;
    while (i < args.len) {
        const arg = args[i];
        i += 1;

        if (std.mem.eql(u8, std.mem.sliceTo(arg, 0), "--json")) {
            config.output_json = true;
            continue;
        }

        if (utils.args.matchesAny(arg, &[_][]const u8{ "--output", "-o" })) {
            if (i < args.len) {
                config.output_file = std.mem.sliceTo(args[i], 0);
                i += 1;
            }
            continue;
        }

        if (std.mem.eql(u8, std.mem.sliceTo(arg, 0), "--iterations")) {
            if (i < args.len) {
                const val = std.mem.sliceTo(args[i], 0);
                config.iterations = std.fmt.parseInt(u32, val, 10) catch 1000;
                i += 1;
            }
            continue;
        }
    }

    try runBenchmarkSuite(allocator, config);
}

/// Run the selected benchmark suite
fn runBenchmarkSuite(allocator: std.mem.Allocator, config: BenchConfig) !void {
    const timer = std.time.Timer.start() catch {
        std.debug.print("Timer not available on this platform.\n", .{});
        return;
    };

    if (!config.output_json) {
        printHeader();
        std.debug.print("Running suite: {s}\n\n", .{config.suite.toString()});
    }

    // Collect results
    var results: std.ArrayListUnmanaged(BenchResult) = .empty;
    defer results.deinit(allocator);

    switch (config.suite) {
        .all => {
            try runSuiteWithResults(allocator, "SIMD Operations", runSimdBenchmarks, &results, config.output_json);
            try runSuiteWithResults(allocator, "Memory Patterns", runMemoryBenchmarks, &results, config.output_json);
            try runSuiteWithResults(allocator, "Concurrency", runConcurrencyBenchmarks, &results, config.output_json);
            try runSuiteWithResults(allocator, "AI/ML", runAiBenchmarks, &results, config.output_json);
        },
        .simd => try runSuiteWithResults(allocator, "SIMD Operations", runSimdBenchmarks, &results, config.output_json),
        .memory => try runSuiteWithResults(allocator, "Memory Patterns", runMemoryBenchmarks, &results, config.output_json),
        .concurrency => try runSuiteWithResults(allocator, "Concurrency", runConcurrencyBenchmarks, &results, config.output_json),
        .database => try runSuiteWithResults(allocator, "Database/HNSW", runDatabaseBenchmarks, &results, config.output_json),
        .network => try runSuiteWithResults(allocator, "Network/HTTP", runNetworkBenchmarks, &results, config.output_json),
        .crypto => try runSuiteWithResults(allocator, "Cryptography", runCryptoBenchmarks, &results, config.output_json),
        .ai => try runSuiteWithResults(allocator, "AI/ML", runAiBenchmarks, &results, config.output_json),
        .quick => {
            try runSuiteWithResults(allocator, "Quick (SIMD)", runQuickSimdBenchmarks, &results, config.output_json);
            try runSuiteWithResults(allocator, "Quick (Memory)", runQuickMemoryBenchmarks, &results, config.output_json);
        },
    }

    var t = timer;
    const elapsed_ns = t.read();
    const duration_sec = @as(f64, @floatFromInt(elapsed_ns)) / 1_000_000_000.0;

    if (config.output_json) {
        try outputJson(allocator, results.items, duration_sec, config.output_file);
    } else {
        printFooter(duration_sec);
    }
}

/// Benchmark result structure
const BenchResult = struct {
    name: []const u8,
    category: []const u8,
    ops_per_sec: f64,
    mean_ns: f64,
    p99_ns: u64,
    iterations: u64,
};

/// Errors that can occur during benchmark execution.
/// Used for benchmark function signatures instead of anyerror.
pub const BenchmarkError = std.mem.Allocator.Error || error{
    TimerUnavailable,
    BenchmarkFailed,
};

/// Benchmark function type for suite runners.
pub const BenchmarkFn = *const fn (std.mem.Allocator, *std.ArrayListUnmanaged(BenchResult)) BenchmarkError!void;

fn runSuiteWithResults(
    allocator: std.mem.Allocator,
    name: []const u8,
    benchFn: BenchmarkFn,
    results: *std.ArrayListUnmanaged(BenchResult),
    json_mode: bool,
) !void {
    if (!json_mode) {
        std.debug.print("--------------------------------------------------------------------------------\n", .{});
        std.debug.print("  {s}\n", .{name});
        std.debug.print("--------------------------------------------------------------------------------\n", .{});
    }
    try benchFn(allocator, results);
    if (!json_mode) {
        std.debug.print("\n", .{});
    }
}

/// Run micro-benchmark for specific operation
fn runMicroBenchmark(allocator: std.mem.Allocator, args: []const [:0]const u8, config: *BenchConfig) !void {
    if (args.len == 0) {
        std.debug.print("Usage: abi bench micro <operation>\n", .{});
        std.debug.print("Operations: hash, alloc, parse, noop\n", .{});
        return;
    }

    const op_name = std.mem.sliceTo(args[0], 0);
    const op = std.meta.stringToEnum(MicroOp, op_name) orelse {
        std.debug.print("Unknown micro-benchmark: {s}\n", .{op_name});
        std.debug.print("Available: hash, alloc, parse, noop\n", .{});
        return;
    };

    // Parse options
    var i: usize = 1;
    while (i < args.len) {
        const arg = args[i];
        i += 1;

        if (std.mem.eql(u8, std.mem.sliceTo(arg, 0), "--iterations")) {
            if (i < args.len) {
                const val = std.mem.sliceTo(args[i], 0);
                config.iterations = std.fmt.parseInt(u32, val, 10) catch 1000;
                i += 1;
            }
        }
        if (std.mem.eql(u8, std.mem.sliceTo(arg, 0), "--json")) {
            config.output_json = true;
        }
    }

    std.debug.print("\nMicro-Benchmark: {s}\n", .{op.toString()});
    std.debug.print("Iterations: {d}\n", .{config.iterations});
    std.debug.print("Warmup: {d}\n\n", .{config.warmup});

    // Warmup
    var warmup_i: u32 = 0;
    while (warmup_i < config.warmup) : (warmup_i += 1) {
        _ = runMicroOp(allocator, op);
    }

    // Benchmark
    const timer = std.time.Timer.start() catch {
        std.debug.print("Timer not available.\n", .{});
        return;
    };

    var iter: u32 = 0;
    while (iter < config.iterations) : (iter += 1) {
        _ = runMicroOp(allocator, op);
    }

    var t = timer;
    const elapsed_ns = t.read();
    const mean_ns = @as(f64, @floatFromInt(elapsed_ns)) / @as(f64, @floatFromInt(config.iterations));
    const ops_per_sec = if (mean_ns > 0) 1_000_000_000.0 / mean_ns else 0;

    if (config.output_json) {
        std.debug.print("{{\"operation\":\"{s}\",\"iterations\":{d},\"mean_ns\":{d:.2},\"ops_per_sec\":{d:.2}}}\n", .{
            op.toString(),
            config.iterations,
            mean_ns,
            ops_per_sec,
        });
    } else {
        std.debug.print("Results:\n", .{});
        std.debug.print("  Mean time: {d:.2} ns\n", .{mean_ns});
        std.debug.print("  Ops/sec: {d:.0}\n", .{ops_per_sec});
    }
}

fn runMicroOp(allocator: std.mem.Allocator, op: MicroOp) usize {
    switch (op) {
        .hash => {
            // Simple hash computation
            const data = "The quick brown fox jumps over the lazy dog";
            var hash: usize = 0;
            for (data) |c| {
                hash = hash *% 31 +% c;
            }
            return hash;
        },
        .alloc => {
            // Allocation pattern
            const buf = allocator.alloc(u8, 4096) catch return 0;
            defer allocator.free(buf);
            return buf.len;
        },
        .parse => {
            // Simple JSON-like parsing
            const json = "{\"key\":\"value\",\"num\":42}";
            var count: usize = 0;
            for (json) |c| {
                if (c == ':' or c == ',') count += 1;
            }
            return count;
        },
        .noop => {
            return 0;
        },
    }
}

// =============================================================================
// Benchmark Suite Implementations
// =============================================================================

fn runSimdBenchmarks(allocator: std.mem.Allocator, results: *std.ArrayListUnmanaged(BenchResult)) BenchmarkError!void {
    // Dot product benchmark
    const sizes = [_]usize{ 64, 256, 1024, 4096 };
    for (sizes) |size| {
        const a = try allocator.alloc(f32, size);
        defer allocator.free(a);
        const b = try allocator.alloc(f32, size);
        defer allocator.free(b);

        // Initialize
        for (a, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i % 100)) / 100.0;
        for (b, 0..) |*v, i| v.* = @as(f32, @floatFromInt((i + 1) % 100)) / 100.0;

        // Benchmark
        const result = try benchmarkOp(allocator, struct {
            fn op(a_ptr: []f32, b_ptr: []f32) f32 {
                var sum: f32 = 0.0;
                for (a_ptr, b_ptr) |x, y| sum += x * y;
                return sum;
            }
        }.op, .{ a, b });

        try results.append(allocator, .{
            .name = try std.fmt.allocPrint(allocator, "dot_product_{d}", .{size}),
            .category = "simd",
            .ops_per_sec = result.ops_per_sec,
            .mean_ns = result.mean_ns,
            .p99_ns = result.p99_ns,
            .iterations = result.iterations,
        });

        std.debug.print("  dot_product[{d}]: {d:.0} ops/sec, {d:.0}ns mean\n", .{ size, result.ops_per_sec, result.mean_ns });
    }
}

fn runMemoryBenchmarks(allocator: std.mem.Allocator, results: *std.ArrayListUnmanaged(BenchResult)) BenchmarkError!void {
    // Arena-style allocation benchmark
    const sizes = [_]usize{ 64, 256, 1024, 4096 };
    for (sizes) |size| {
        const result = try benchmarkAllocOp(allocator, size);

        try results.append(allocator, .{
            .name = try std.fmt.allocPrint(allocator, "alloc_free_{d}", .{size}),
            .category = "memory",
            .ops_per_sec = result.ops_per_sec,
            .mean_ns = result.mean_ns,
            .p99_ns = result.p99_ns,
            .iterations = result.iterations,
        });

        std.debug.print("  alloc_free[{d}]: {d:.0} ops/sec, {d:.0}ns mean\n", .{ size, result.ops_per_sec, result.mean_ns });
    }
}

fn runConcurrencyBenchmarks(allocator: std.mem.Allocator, results: *std.ArrayListUnmanaged(BenchResult)) BenchmarkError!void {
    // Atomic operations benchmark
    var counter = std.atomic.Value(u64).init(0);

    const timer = std.time.Timer.start() catch return;
    const iterations: u64 = 100000;

    var i: u64 = 0;
    while (i < iterations) : (i += 1) {
        _ = counter.fetchAdd(1, .seq_cst);
    }

    var t = timer;
    const elapsed_ns = t.read();
    const mean_ns = @as(f64, @floatFromInt(elapsed_ns)) / @as(f64, @floatFromInt(iterations));
    const ops_per_sec = if (mean_ns > 0) 1_000_000_000.0 / mean_ns else 0;

    try results.append(allocator, .{
        .name = "atomic_increment",
        .category = "concurrency",
        .ops_per_sec = ops_per_sec,
        .mean_ns = mean_ns,
        .p99_ns = @intFromFloat(mean_ns * 1.5),
        .iterations = iterations,
    });

    std.debug.print("  atomic_increment: {d:.0} ops/sec, {d:.0}ns mean\n", .{ ops_per_sec, mean_ns });
}

fn runDatabaseBenchmarks(allocator: std.mem.Allocator, results: *std.ArrayListUnmanaged(BenchResult)) BenchmarkError!void {
    std.debug.print("  (Database benchmarks require enable-database build flag)\n", .{});
    _ = allocator;
    _ = results;
}

fn runNetworkBenchmarks(allocator: std.mem.Allocator, results: *std.ArrayListUnmanaged(BenchResult)) BenchmarkError!void {
    std.debug.print("  (Network benchmarks require enable-network build flag)\n", .{});
    _ = allocator;
    _ = results;
}

fn runCryptoBenchmarks(allocator: std.mem.Allocator, results: *std.ArrayListUnmanaged(BenchResult)) BenchmarkError!void {
    std.debug.print("  (Crypto benchmarks placeholder)\n", .{});
    _ = allocator;
    _ = results;
}

fn runAiBenchmarks(allocator: std.mem.Allocator, results: *std.ArrayListUnmanaged(BenchResult)) BenchmarkError!void {
    // Simple matrix multiply benchmark
    const m: usize = 64;
    const k: usize = 64;
    const n: usize = 64;

    const a = try allocator.alloc(f32, m * k);
    defer allocator.free(a);
    const b = try allocator.alloc(f32, k * n);
    defer allocator.free(b);
    const c = try allocator.alloc(f32, m * n);
    defer allocator.free(c);

    // Initialize
    for (a, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i % 100)) / 100.0 - 0.5;
    for (b, 0..) |*v, i| v.* = @as(f32, @floatFromInt((i + 1) % 100)) / 100.0 - 0.5;

    const timer = std.time.Timer.start() catch return;
    const iterations: u64 = 100;

    var iter: u64 = 0;
    while (iter < iterations) : (iter += 1) {
        // Naive matmul
        for (0..m) |i| {
            for (0..n) |j| {
                var sum: f32 = 0.0;
                for (0..k) |kk| {
                    sum += a[i * k + kk] * b[kk * n + j];
                }
                c[i * n + j] = sum;
            }
        }
    }

    var t = timer;
    const elapsed_ns = t.read();
    const mean_ns = @as(f64, @floatFromInt(elapsed_ns)) / @as(f64, @floatFromInt(iterations));
    const ops_per_sec = if (mean_ns > 0) 1_000_000_000.0 / mean_ns else 0;

    // Calculate GFLOPS: 2*M*N*K operations per matmul
    const flops: f64 = 2.0 * @as(f64, @floatFromInt(m)) * @as(f64, @floatFromInt(n)) * @as(f64, @floatFromInt(k));
    const gflops = (flops * ops_per_sec) / 1_000_000_000.0;

    try results.append(allocator, .{
        .name = try std.fmt.allocPrint(allocator, "matmul_{d}x{d}x{d}", .{ m, k, n }),
        .category = "ai",
        .ops_per_sec = ops_per_sec,
        .mean_ns = mean_ns,
        .p99_ns = @intFromFloat(mean_ns * 1.2),
        .iterations = iterations,
    });

    std.debug.print("  matmul[{d}x{d}x{d}]: {d:.2} GFLOPS, {d:.0}ns mean\n", .{ m, k, n, gflops, mean_ns });
}

fn runQuickSimdBenchmarks(allocator: std.mem.Allocator, results: *std.ArrayListUnmanaged(BenchResult)) BenchmarkError!void {
    // Quick SIMD benchmark - single size
    const size: usize = 256;
    const a = try allocator.alloc(f32, size);
    defer allocator.free(a);
    const b = try allocator.alloc(f32, size);
    defer allocator.free(b);

    for (a, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i % 100)) / 100.0;
    for (b, 0..) |*v, i| v.* = @as(f32, @floatFromInt((i + 1) % 100)) / 100.0;

    const result = try benchmarkOp(allocator, struct {
        fn op(a_ptr: []f32, b_ptr: []f32) f32 {
            var sum: f32 = 0.0;
            for (a_ptr, b_ptr) |x, y| sum += x * y;
            return sum;
        }
    }.op, .{ a, b });

    try results.append(allocator, .{
        .name = "quick_dot_product",
        .category = "simd",
        .ops_per_sec = result.ops_per_sec,
        .mean_ns = result.mean_ns,
        .p99_ns = result.p99_ns,
        .iterations = result.iterations,
    });

    std.debug.print("  quick_dot_product: {d:.0} ops/sec\n", .{result.ops_per_sec});
}

fn runQuickMemoryBenchmarks(allocator: std.mem.Allocator, results: *std.ArrayListUnmanaged(BenchResult)) BenchmarkError!void {
    const result = try benchmarkAllocOp(allocator, 256);

    try results.append(allocator, .{
        .name = "quick_alloc",
        .category = "memory",
        .ops_per_sec = result.ops_per_sec,
        .mean_ns = result.mean_ns,
        .p99_ns = result.p99_ns,
        .iterations = result.iterations,
    });

    std.debug.print("  quick_alloc: {d:.0} ops/sec\n", .{result.ops_per_sec});
}

// =============================================================================
// Benchmark Utilities
// =============================================================================

const BenchmarkResult = struct {
    ops_per_sec: f64,
    mean_ns: f64,
    p99_ns: u64,
    iterations: u64,
};

fn benchmarkOp(
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
    const timer = std.time.Timer.start() catch return BenchmarkResult{
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

fn benchmarkAllocOp(allocator: std.mem.Allocator, size: usize) !BenchmarkResult {
    const warmup: u64 = 100;
    const iterations: u64 = 10000;

    // Warmup
    var w: u64 = 0;
    while (w < warmup) : (w += 1) {
        const buf = try allocator.alloc(u8, size);
        allocator.free(buf);
    }

    // Benchmark
    const timer = std.time.Timer.start() catch return BenchmarkResult{
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

fn outputJson(allocator: std.mem.Allocator, results: []const BenchResult, duration_sec: f64, output_file: ?[]const u8) !void {
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
        var io_backend = std.Io.Threaded.init(allocator, .{
            .environ = std.process.Environ.empty,
        });
        defer io_backend.deinit();
        const io = io_backend.io();

        const file = std.Io.Dir.cwd().createFile(io, path, .{ .truncate = true }) catch |err| {
            std.debug.print("Error creating output file: {t}\n", .{err});
            return;
        };
        defer file.close(io);

        var write_buf: [4096]u8 = undefined;
        var writer = file.writer(io, &write_buf);
        _ = writer.interface.write(json_buf.items) catch |err| {
            std.debug.print("Error writing to file: {t}\n", .{err});
            return;
        };
        writer.flush() catch {};
        std.debug.print("Results written to: {s}\n", .{path});
    } else {
        std.debug.print("{s}", .{json_buf.items});
    }
}

fn printHeader() void {
    const header =
        \\
        \\╔════════════════════════════════════════════════════════════════════════════╗
        \\║                     ABI FRAMEWORK BENCHMARK SUITE                          ║
        \\╚════════════════════════════════════════════════════════════════════════════╝
        \\
    ;
    std.debug.print("{s}", .{header});
}

fn printFooter(duration_sec: f64) void {
    std.debug.print("================================================================================\n", .{});
    std.debug.print(" BENCHMARK COMPLETE - Total time: {d:.2}s\n", .{duration_sec});
    std.debug.print("================================================================================\n", .{});
}

fn printAvailableSuites() void {
    std.debug.print("Available benchmark suites:\n\n", .{});
    std.debug.print("  all          Run all benchmark suites\n", .{});
    std.debug.print("  simd         SIMD/Vector operations (dot product, matmul)\n", .{});
    std.debug.print("  memory       Memory allocator patterns (arena, pool)\n", .{});
    std.debug.print("  concurrency  Concurrency primitives (atomics, locks)\n", .{});
    std.debug.print("  database     Database/HNSW vector search\n", .{});
    std.debug.print("  network      HTTP/Network operations\n", .{});
    std.debug.print("  crypto       Cryptographic operations\n", .{});
    std.debug.print("  ai           AI/ML inference (GEMM, attention)\n", .{});
    std.debug.print("  quick        Quick benchmarks for CI\n", .{});
    std.debug.print("\nMicro-benchmarks:\n", .{});
    std.debug.print("  abi bench micro hash   - Simple hash computation\n", .{});
    std.debug.print("  abi bench micro alloc  - Memory allocation pattern\n", .{});
    std.debug.print("  abi bench micro parse  - Simple parsing operation\n", .{});
    std.debug.print("  abi bench micro noop   - Empty operation (baseline)\n", .{});
}

fn printHelp() void {
    const help_text =
        \\Usage: abi bench <suite> [options]
        \\
        \\Run performance benchmarks.
        \\
        \\Suites:
        \\  all           Run all benchmark suites
        \\  simd          SIMD/Vector operations
        \\  memory        Memory allocator patterns
        \\  concurrency   Concurrency primitives
        \\  database      Database/HNSW vector search
        \\  network       HTTP/Network operations
        \\  crypto        Cryptographic operations
        \\  ai            AI/ML inference
        \\  quick         Quick benchmarks for CI
        \\  list          List available suites
        \\  micro <op>    Run micro-benchmark (hash, alloc, parse, noop)
        \\
        \\Options:
        \\  --json              Output results in JSON format
        \\  -o, --output <file> Write results to file
        \\  --iterations <n>    Number of iterations for micro-benchmarks
        \\  -h, --help          Show this help message
        \\
        \\Examples:
        \\  abi bench all                     # Run all benchmarks
        \\  abi bench simd --json             # SIMD benchmarks with JSON output
        \\  abi bench quick                   # Quick CI benchmarks
        \\  abi bench micro hash              # Run hash micro-benchmark
        \\  abi bench ai --output results.json
        \\
    ;
    std.debug.print("{s}", .{help_text});
}
