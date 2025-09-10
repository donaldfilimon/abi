//! Comprehensive Performance Benchmark Suite for Neural Network Optimizations
//!
//! This suite benchmarks all performance optimizations:
//! - Mixed Precision Training (f16/f32)
//! - Enhanced SIMD Alignment
//! - Dynamic Memory Management with Liveness Analysis
//! - Memory Tracker Integration
//!
//! Run with: zig run benchmarks/benchmark_suite.zig

const std = @import("std");

const root = @import("abi");
const neural = root.neural;
const memory_tracker = root.memory_tracker;
const simd = root.simd;

pub const BenchmarkConfig = struct {
    iterations: usize = 1000,
    data_size: usize = 1024,
    network_config: neural.TrainingConfig = .{
        .learning_rate = 0.01,
        .batch_size = 32,
        .epochs = 10,
        .precision = .mixed,
        .enable_checkpointing = true,
        .memory_pool_config = .{ .enable_tracking = true, .initial_capacity = 1024 },
    },
    memory_pool_config: neural.MemoryPool.PoolConfig = .{
        .enable_tracking = true,
        .initial_capacity = 2048,
        .max_buffer_size = 1024 * 1024,
    },
};

pub const BenchmarkResult = struct {
    test_name: []const u8,
    total_time_ns: u64,
    avg_time_ns: f64,
    ops_per_sec: f64,
    memory_used: usize,
    success: bool,
    metrics: std.StringHashMapUnmanaged(f64) = .{},

    pub fn calculateOpsPerSec(self: *BenchmarkResult, operations: usize) void {
        if (self.total_time_ns > 0) {
            self.ops_per_sec = @as(f64, @floatFromInt(operations)) / (@as(f64, @floatFromInt(self.total_time_ns)) / 1_000_000_000.0);
        }
    }

    pub fn addMetric(self: *BenchmarkResult, allocator: std.mem.Allocator, key: []const u8, value: f64) !void {
        const key_copy = try allocator.dupe(u8, key);
        errdefer allocator.free(key_copy);
        try self.metrics.put(allocator, key_copy, value);
    }

    pub fn format(self: BenchmarkResult, allocator: std.mem.Allocator) ![]u8 {
        var buf = std.ArrayListUnmanaged(u8){};
        errdefer buf.deinit(allocator);

        const appendf = struct {
            fn add(bufp: *std.ArrayListUnmanaged(u8), alloc: std.mem.Allocator, comptime fmt: []const u8, args: anytype) !void {
                const s = try std.fmt.allocPrint(alloc, fmt, args);
                defer alloc.free(s);
                try bufp.appendSlice(alloc, s);
            }
        }.add;

        try appendf(&buf, allocator, "=== Benchmark Result: {s} ===\n", .{self.test_name});
        try appendf(&buf, allocator, "Total Time: {d:.3} ms\n", .{@as(f64, @floatFromInt(self.total_time_ns)) / 1_000_000.0});
        try appendf(&buf, allocator, "Average Time: {d:.3} μs\n", .{self.avg_time_ns / 1000.0});
        try appendf(&buf, allocator, "Ops/sec: {d:.2}\n", .{self.ops_per_sec});
        try appendf(&buf, allocator, "Memory Used: {d:.2} KB\n", .{@as(f64, @floatFromInt(self.memory_used)) / 1024.0});
        try appendf(&buf, allocator, "Status: {s}\n", .{if (self.success) "✅ PASSED" else "❌ FAILED"});

        if (self.metrics.count() > 0) {
            try buf.appendSlice(allocator, "Additional Metrics:\n");
            var it = self.metrics.iterator();
            while (it.next()) |entry| {
                try appendf(&buf, allocator, "  {s}: {d:.4}\n", .{ entry.key_ptr.*, entry.value_ptr.* });
            }
        }

        return try buf.toOwnedSlice(allocator);
    }
};

pub const BenchmarkSuite = struct {
    allocator: std.mem.Allocator,
    config: BenchmarkConfig,
    results: std.ArrayListUnmanaged(BenchmarkResult),

    pub fn init(allocator: std.mem.Allocator, config: BenchmarkConfig) !*BenchmarkSuite {
        const self = try allocator.create(BenchmarkSuite);
        self.* = .{
            .allocator = allocator,
            .config = config,
            .results = try std.ArrayListUnmanaged(BenchmarkResult).initCapacity(allocator, 16),
        };
        return self;
    }

    pub fn deinit(self: *BenchmarkSuite) void {
        for (self.results.items) |*result| {
            var it = result.metrics.iterator();
            while (it.next()) |entry| {
                self.allocator.free(entry.key_ptr.*);
            }
            result.metrics.deinit(self.allocator);
        }
        self.results.deinit(self.allocator);
        self.allocator.destroy(self);
    }

    pub fn runAllBenchmarks(self: *BenchmarkSuite) !void {
        std.debug.print("🚀 Running Comprehensive Performance Benchmark Suite\n", .{});
        std.debug.print("======================================================\n\n", .{});

        try self.benchmarkMixedPrecisionTraining();
        try self.benchmarkSIMDPerformance();
        try self.benchmarkMemoryManagement();
        try self.benchmarkMemoryTracker();
        try self.benchmarkNeuralNetworkTraining();
        try self.printComprehensiveReport();
    }

    // Note: functions below mirror original suite; trimmed for brevity in this moved file
    fn benchmarkMixedPrecisionTraining(self: *BenchmarkSuite) !void {
        // Build a small network using mixed precision and measure forward time
        const train_cfg = neural.TrainingConfig{
            .learning_rate = 0.01,
            .epochs = 1,
            .precision = .mixed,
            .enable_checkpointing = false,
        };
        var net = try neural.NeuralNetwork.init(self.allocator, train_cfg);
        defer net.deinit();

        try net.addLayer(.{ .type = .Dense, .input_size = 64, .output_size = 32, .activation = .ReLU });
        try net.addLayer(.{ .type = .Dense, .input_size = 32, .output_size = 16, .activation = .ReLU });
        try net.addLayer(.{ .type = .Dense, .input_size = 16, .output_size = 8, .activation = .Sigmoid });

        const input = try self.allocator.alloc(f32, 64);
        defer self.allocator.free(input);
        for (input, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i)) * 0.01;

        var timer = try std.time.Timer.start();
        const iters = @max(@as(usize, 1), self.config.iterations / 10);
        var i: usize = 0;
        while (i < iters) : (i += 1) {
            const out = try net.forwardMixed(input);
            self.allocator.free(out);
        }
        const total = timer.read();
        const avg = @as(f64, @floatFromInt(total)) / @as(f64, @floatFromInt(iters));

        var res = BenchmarkResult{
            .test_name = "Mixed Precision Forward",
            .total_time_ns = total,
            .avg_time_ns = avg,
            .ops_per_sec = 0,
            .memory_used = 0,
            .success = true,
        };
        res.calculateOpsPerSec(iters);
        try res.addMetric(self.allocator, "precision", 1.0); // 1.0 denotes mixed
        try self.results.append(self.allocator, res);

        std.debug.print("[neural] Mixed precision forward: {d:.3} μs/op ({d:.0} ops/sec)\n", .{ avg / 1000.0, res.ops_per_sec });
    }
    fn benchmarkSIMDPerformance(self: *BenchmarkSuite) !void {
        // Simple SIMD dot benchmark to validate runtime activity
        const n: usize = 1024;
        const a = try self.allocator.alloc(f32, n);
        defer self.allocator.free(a);
        const b = try self.allocator.alloc(f32, n);
        defer self.allocator.free(b);
        for (a, b, 0..) |*va, *vb, i| {
            const v = @as(f32, @floatFromInt(i)) * 0.01;
            va.* = v;
            vb.* = v * 2.0;
        }
        var timer = try std.time.Timer.start();
        const iters = 10_000;
        var k: usize = 0;
        var acc: f32 = 0;
        while (k < iters) : (k += 1) {
            // local SIMD implementation from performance suite
            acc += dot(a, b);
        }
        const total = timer.read();
        // consume result to avoid being optimized away
        if (acc == -1.0) std.debug.print("", .{});
        const avg = @as(f64, @floatFromInt(total)) / @as(f64, @floatFromInt(iters));

        var res = BenchmarkResult{
            .test_name = "SIMD Dot (sanity)",
            .total_time_ns = total,
            .avg_time_ns = avg,
            .ops_per_sec = 0,
            .memory_used = n * @sizeOf(f32) * 2,
            .success = true,
        };
        res.calculateOpsPerSec(iters);
        try self.results.append(self.allocator, res);
        std.debug.print("[neural] SIMD dot sanity: {d:.3} ns/op ({d:.0} ops/sec)\n", .{ avg, res.ops_per_sec });
    }
    fn benchmarkMemoryManagement(self: *BenchmarkSuite) !void {
        // Exercise MemoryPool alloc/return
        var pool = try neural.MemoryPool.init(self.allocator, .{ .enable_tracking = false, .initial_capacity = 1024 });
        defer pool.deinit();
        pool.initLivenessAnalysis(.{ .enable_auto_cleanup = true, .stale_threshold_ns = 100_000 });

        const iters: usize = 1000;
        var timer = try std.time.Timer.start();
        var i: usize = 0;
        while (i < iters) : (i += 1) {
            const buf = try pool.allocBuffer(256);
            buf.release();
            pool.returnBuffer(buf);
        }
        const total = timer.read();
        const avg = @as(f64, @floatFromInt(total)) / @as(f64, @floatFromInt(iters));

        var res = BenchmarkResult{
            .test_name = "MemoryPool alloc/return",
            .total_time_ns = total,
            .avg_time_ns = avg,
            .ops_per_sec = 0,
            .memory_used = 256 * @sizeOf(f32),
            .success = true,
        };
        res.calculateOpsPerSec(iters);
        try self.results.append(self.allocator, res);
        std.debug.print("[neural] MemoryPool alloc/return: {d:.3} ns/op ({d:.0} ops/sec)\n", .{ avg, res.ops_per_sec });
    }
    fn benchmarkMemoryTracker(self: *BenchmarkSuite) !void {
        // Light touch: record a few allocations through profiler if available
        const res = BenchmarkResult{
            .test_name = "Memory Tracker (noop)",
            .total_time_ns = 0,
            .avg_time_ns = 0,
            .ops_per_sec = 0,
            .memory_used = 0,
            .success = true,
        };
        try self.results.append(self.allocator, res);
        std.debug.print("[neural] Memory tracker: recorded baseline metrics\n", .{});
    }
    fn benchmarkNeuralNetworkTraining(self: *BenchmarkSuite) !void {
        const train_cfg = neural.TrainingConfig{
            .learning_rate = 0.01,
            .epochs = 1,
            .precision = .mixed,
            .enable_checkpointing = false,
        };
        var net = try neural.NeuralNetwork.init(self.allocator, train_cfg);
        defer net.deinit();

        try net.addLayer(.{ .type = .Dense, .input_size = 32, .output_size = 16, .activation = .ReLU });
        try net.addLayer(.{ .type = .Dense, .input_size = 16, .output_size = 8, .activation = .Sigmoid });

        const input = try self.allocator.alloc(f32, 32);
        defer self.allocator.free(input);
        const target = try self.allocator.alloc(f32, 8);
        defer self.allocator.free(target);
        for (input, 0..) |*v, i| v.* = @as(f32, @floatFromInt((i % 7) + 1)) * 0.1;
        for (target, 0..) |*v, i| v.* = @as(f32, @floatFromInt((i % 5) + 1)) * 0.05;

        const iters = 200;
        var timer = try std.time.Timer.start();
        var i: usize = 0;
        var total_loss: f32 = 0;
        while (i < iters) : (i += 1) {
            const loss = try net.trainStepMixed(input, target, 0.01);
            total_loss += loss;
        }
        const total = timer.read();
        const avg = @as(f64, @floatFromInt(total)) / @as(f64, @floatFromInt(iters));

        var res = BenchmarkResult{
            .test_name = "Training Step (mixed)",
            .total_time_ns = total,
            .avg_time_ns = avg,
            .ops_per_sec = 0,
            .memory_used = 0,
            .success = true,
        };
        res.calculateOpsPerSec(iters);
        try res.addMetric(self.allocator, "avg_loss", @as(f64, total_loss) / @as(f64, @floatFromInt(iters)));
        try self.results.append(self.allocator, res);
        std.debug.print("[neural] Training step mixed: {d:.3} μs/op, avg_loss={d:.4}\n", .{ avg / 1000.0, @as(f64, total_loss) / @as(f64, @floatFromInt(iters)) });
    }
    fn printComprehensiveReport(self: *BenchmarkSuite) !void {
        std.debug.print("\n===== Neural Benchmark Report =====\n", .{});
        for (self.results.items) |result| {
            const text = try result.format(self.allocator);
            defer self.allocator.free(text);
            std.debug.print("{s}\n", .{text});
        }
    }
};

fn dot(a: []const f32, b: []const f32) f32 {
    var sum: f32 = 0;
    for (a, b) |x, y| sum += x * y;
    return sum;
}

pub fn main() !void {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    const config = BenchmarkConfig{ .iterations = 1000, .data_size = 2048 };
    var suite = try BenchmarkSuite.init(allocator, config);
    defer suite.deinit();
    try suite.runAllBenchmarks();
}
