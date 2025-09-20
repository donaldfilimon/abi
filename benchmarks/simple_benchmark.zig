//! Enhanced Simple Benchmark Suite
//!
//! This benchmark provides simplified performance testing:
//! - Basic vector operations
//! - Memory allocation patterns
//! - Simple mathematical functions
//! - Quick performance validation

const std = @import("std");
const framework = @import("benchmark_framework.zig");

/// Simple benchmark configuration
pub const SimpleBenchmarkConfig = struct {
    framework_config: framework.BenchmarkConfig = .{
        .warmup_iterations = 10,
        .measurement_iterations = 100,
        .samples = 5,
        .enable_memory_tracking = false,
        .enable_detailed_stats = false,
        .output_format = .console,
    },
    test_sizes: []const usize = &[_]usize{ 100, 1000, 10000 },
};

/// Simple benchmark suite
pub const SimpleBenchmarkSuite = struct {
    framework_suite: *framework.BenchmarkSuite,
    config: SimpleBenchmarkConfig,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, config: SimpleBenchmarkConfig) !*SimpleBenchmarkSuite {
        const framework_suite = try framework.BenchmarkSuite.init(allocator, config.framework_config);
        const self = try allocator.create(SimpleBenchmarkSuite);
        self.* = .{
            .framework_suite = framework_suite,
            .config = config,
            .allocator = allocator,
        };
        return self;
    }

    pub fn deinit(self: *SimpleBenchmarkSuite) void {
        self.framework_suite.deinit();
        self.allocator.destroy(self);
    }

    pub fn runAllBenchmarks(self: *SimpleBenchmarkSuite) !void {
        std.log.info("⚡ Running Simple Benchmarks", .{});
        std.log.info("============================", .{});

        // Basic operations
        try self.benchmarkBasicOperations();

        // Memory operations
        try self.benchmarkMemoryOperations();

        // Mathematical operations
        try self.benchmarkMathOperations();

        // Print results
        try self.framework_suite.printReport();
    }

    fn benchmarkBasicOperations(self: *SimpleBenchmarkSuite) !void {
        std.log.info("🔧 Benchmarking Basic Operations", .{});

        for (self.config.test_sizes) |size| {
            // Array allocation and initialization
            const alloc_context = struct {
                fn allocateArray(context: @This()) !void {
                    const arr = try self.allocator.alloc(i32, context.size);
                    defer self.allocator.free(arr);

                    for (arr, 0..) |*val, i| {
                        val.* = @as(i32, @intCast(i));
                    }
                }
                size: usize,
                self: *SimpleBenchmarkSuite,
            }{
                .size = size,
                .self = self,
            };

            try self.framework_suite.runBenchmarkFmt("Array Allocation ({} elements)", .{size}, "Basic", alloc_context.allocateArray, alloc_context);

            // Array sum
            const sum_context = struct {
                fn arraySum(context: @This()) !i64 {
                    var sum: i64 = 0;
                    for (0..context.size) |i| {
                        sum += @as(i64, @intCast(i));
                    }
                    return sum;
                }
                size: usize,
            }{
                .size = size,
            };

            try self.framework_suite.runBenchmarkFmt("Array Sum ({} elements)", .{size}, "Basic", sum_context.arraySum, sum_context);
        }
    }

    fn benchmarkMemoryOperations(self: *SimpleBenchmarkSuite) !void {
        std.log.info("💾 Benchmarking Memory Operations", .{});

        const memory_context = struct {
            fn memoryAllocation(context: @This()) !void {
                const buffer = try self.allocator.alloc(u8, context.size);
                defer self.allocator.free(buffer);

                // Touch memory
                @memset(buffer, 0x42);
            }
            size: usize,
            self: *SimpleBenchmarkSuite,
        }{
            .size = 1024,
            .self = self,
        };

        try self.framework_suite.runBenchmark("Memory Allocation (1KB)", "Memory", memory_context.memoryAllocation, memory_context);
    }

    fn benchmarkMathOperations(self: *SimpleBenchmarkSuite) !void {
        std.log.info("🧮 Benchmarking Math Operations", .{});

        const math_context = struct {
            fn mathOperations(_: @This()) !f64 {
                var result: f64 = 0.0;
                for (0..1000) |i| {
                    const val = @as(f64, @floatFromInt(i)) * 0.001;
                    result += @sin(val) + @cos(val);
                }
                return result;
            }
        }{};

        try self.framework_suite.runBenchmark("Math Operations (sin, cos)", "Math", math_context.mathOperations, math_context);
    }
};

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const config = SimpleBenchmarkConfig{};
    var suite = try SimpleBenchmarkSuite.init(allocator, config);
    defer suite.deinit();

    try suite.runAllBenchmarks();
}
