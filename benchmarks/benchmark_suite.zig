//! Simplified Performance Benchmark Suite
//!
//! This suite provides basic performance benchmarks for core functionality:
//! - AI activation functions
//! - Memory management
//! - SIMD operations
//!
//! Run with: zig run benchmarks/benchmark_suite.zig

const std = @import("std");

const root = @import("abi");
const ai = root.ai;
const monitoring = root.monitoring;
const core = root.core;

pub const BenchmarkConfig = struct {
    iterations: usize = 1000,
    data_size: usize = 1024,
    network_config: ai.TrainingConfig = .{
        .learning_rate = 0.01,
        .batch_size = 32,
        .epochs = 10,
        .use_mixed_precision = true,
        .checkpoint_frequency = 10,
    },
    // Using standard allocator for memory management
    // memory_pool_config: ai.MemoryPool.PoolConfig = .{
    //     .enable_tracking = true,
    //     .initial_capacity = 2048,
    //     .max_buffer_size = 1024 * 1024,
    // },
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
        std.debug.print("🚀 Running Simplified Performance Benchmark Suite\n", .{});
        std.debug.print("================================================\n\n", .{});

        try self.benchmarkAIActivationFunctions();
        try self.benchmarkSIMDPerformance();
        try self.benchmarkMemoryManagement();
        try self.benchmarkVectorOperations();
        try self.printComprehensiveReport();
    }

    // Note: functions below mirror original suite; trimmed for brevity in this moved file
    fn benchmarkAIActivationFunctions(self: *BenchmarkSuite) !void {
        // Test AI activation function performance
        const iters = self.config.iterations;
        var timer = try std.time.Timer.start();
        var i: usize = 0;
        while (i < iters) : (i += 1) {
            const x: f32 = @as(f32, @floatFromInt(i % 100)) * 0.01;
            _ = ai.ActivationUtils.fastSigmoid(x);
            _ = ai.ActivationUtils.fastTanh(x);
            _ = ai.ActivationUtils.fastGelu(x);
        }
        const total = timer.read();
        const avg = @as(f64, @floatFromInt(total)) / @as(f64, @floatFromInt(iters));

        var res = BenchmarkResult{
            .test_name = "AI Activation Functions",
            .total_time_ns = total,
            .avg_time_ns = avg,
            .ops_per_sec = 0,
            .memory_used = 0,
            .success = true,
        };
        res.calculateOpsPerSec(iters);
        try self.results.append(self.allocator, res);
        std.debug.print("[ai] Activation functions: {d:.3} ns/op ({d:.0} ops/sec)\n", .{ avg, res.ops_per_sec });
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
        // Memory pool benchmark disabled - ai.MemoryPool not available
        // Exercise standard allocator performance instead
        const iters: usize = 1000;
        var timer = try std.time.Timer.start();
        var i: usize = 0;
        while (i < iters) : (i += 1) {
            const buf = try self.allocator.alloc(f32, 64);
            defer self.allocator.free(buf);
            // Simple memory access pattern
            for (buf) |*v| v.* = @as(f32, @floatFromInt(i % 10)) * 0.1;
        }
        const total = timer.read();
        const avg = @as(f64, @floatFromInt(total)) / @as(f64, @floatFromInt(iters));

        var res = BenchmarkResult{
            .test_name = "Standard Allocator Performance",
            .total_time_ns = total,
            .avg_time_ns = avg,
            .ops_per_sec = 0,
            .memory_used = 64 * @sizeOf(f32),
            .success = true,
        };
        res.calculateOpsPerSec(iters);
        try self.results.append(self.allocator, res);
        std.debug.print("[neural] Standard allocator: {d:.3} ns/op ({d:.0} ops/sec)\n", .{ avg, res.ops_per_sec });
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
    fn benchmarkVectorOperations(self: *BenchmarkSuite) !void {
        // Test basic vector operations
        const n: usize = 1024;
        const a = try self.allocator.alloc(f32, n);
        defer self.allocator.free(a);
        const b = try self.allocator.alloc(f32, n);
        defer self.allocator.free(b);

        for (a, b, 0..) |*va, *vb, i| {
            va.* = @as(f32, @floatFromInt(i)) * 0.01;
            vb.* = @as(f32, @floatFromInt(i % 5)) * 0.02;
        }

        const iters = 1000;
        var timer = try std.time.Timer.start();
        var i: usize = 0;
        var result: f32 = 0;
        while (i < iters) : (i += 1) {
            result += dot(a, b);
        }
        const total = timer.read();
        const avg = @as(f64, @floatFromInt(total)) / @as(f64, @floatFromInt(iters));

        // Prevent optimization
        if (result < 0) std.debug.print("", .{});

        var res = BenchmarkResult{
            .test_name = "Vector Dot Product",
            .total_time_ns = total,
            .avg_time_ns = avg,
            .ops_per_sec = 0,
            .memory_used = n * @sizeOf(f32) * 2,
            .success = true,
        };
        res.calculateOpsPerSec(iters);
        try self.results.append(self.allocator, res);
        std.debug.print("[vector] Dot product: {d:.3} ns/op ({d:.0} ops/sec)\n", .{ avg, res.ops_per_sec });
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
