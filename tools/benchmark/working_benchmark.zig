const std = @import("std");
const testing = std.testing;

pub const BenchmarkResult = struct {
    name: []const u8,
    duration_ns: u64,
    operations: u64,
    throughput_ops_per_second: f64,
    memory_used: usize,
};

pub const BenchmarkSuite = struct {
    allocator: std.mem.Allocator,
    results: std.ArrayList(BenchmarkResult),

    pub fn init(allocator: std.mem.Allocator) BenchmarkSuite {
        return BenchmarkSuite{
            .allocator = allocator,
            .results = std.ArrayList(BenchmarkResult){},
        };
    }

    pub fn deinit(self: *BenchmarkSuite) void {
        self.results.deinit(self.allocator);
    }

    pub fn benchmark(self: *BenchmarkSuite, comptime name: []const u8, comptime func: anytype, args: anytype) !void {
        const start_time = std.time.nanoTimestamp;
        const start_memory = self.getCurrentMemoryUsage();

        const operations = @call(.auto, func, args);

        const end_time = std.time.nanoTimestamp;
        const end_memory = self.getCurrentMemoryUsage();

        const duration = @as(u64, @intCast(end_time - start_time));
        const throughput = @as(f64, @floatFromInt(operations)) / (@as(f64, @floatFromInt(duration)) / 1e9);

        const result = BenchmarkResult{
            .name = name,
            .duration_ns = duration,
            .operations = operations,
            .throughput_ops_per_second = throughput,
            .memory_used = end_memory -| start_memory,
        };

        try self.results.append(self.allocator, result);
    }

    pub fn benchmarkFallible(self: *BenchmarkSuite, comptime name: []const u8, comptime func: anytype, args: anytype) !void {
        const start_time = std.time.nanoTimestamp;
        const start_memory = self.getCurrentMemoryUsage();

        const operations = try @call(.auto, func, args);

        const end_time = std.time.nanoTimestamp;
        const end_memory = self.getCurrentMemoryUsage();

        const duration = @as(u64, @intCast(end_time - start_time));
        const throughput = @as(f64, @floatFromInt(operations)) / (@as(f64, @floatFromInt(duration)) / 1e9);

        const result = BenchmarkResult{
            .name = name,
            .duration_ns = duration,
            .operations = operations,
            .throughput_ops_per_second = throughput,
            .memory_used = end_memory -| start_memory,
        };

        try self.results.append(self.allocator, result);
    }

    pub fn printResults(self: *const BenchmarkSuite) void {
        std.debug.print("\n=== Benchmark Results ===\n", .{});
        for (self.results.items) |result| {
            std.debug.print("{s}: {d:.2}ms ({d} ops, {d:.0} ops/sec, {d} bytes)\n", .{
                result.name,
                @as(f64, @floatFromInt(result.duration_ns)) / 1e6,
                result.operations,
                result.throughput_ops_per_second,
                result.memory_used,
            });
        }
        std.debug.print("=========================\n", .{});
    }

    fn getCurrentMemoryUsage(self: *const BenchmarkSuite) usize {
        _ = self;
        // Simple memory usage estimation
        // In a real implementation, this would use the tracking allocator
        return 0;
    }
};

// Benchmark functions
fn vectorAddBenchmark(comptime size: usize) u64 {
    var sum: f32 = 0;
    var i: usize = 0;
    while (i < size) : (i += 1) {
        sum += @as(f32, @floatFromInt(i));
    }
    return size;
}

fn arrayListBenchmark(allocator: std.mem.Allocator, comptime size: usize) !u64 {
    var list = std.ArrayList(u32){};
    defer list.deinit(allocator);

    var i: u32 = 0;
    while (i < size) : (i += 1) {
        try list.append(allocator, i);
    }

    return size;
}

fn hashMapBenchmark(allocator: std.mem.Allocator, comptime size: usize) !u64 {
    var map = std.HashMap(u32, u32, std.hash_map.AutoContext(u32), std.hash_map.default_max_load_percentage).init(allocator);
    defer map.deinit();

    var i: u32 = 0;
    while (i < size) : (i += 1) {
        try map.put(i, i * 2);
    }

    return size;
}

test "benchmark suite" {
    var suite = BenchmarkSuite.init(testing.allocator);
    defer suite.deinit();

    // Simple vector operations
    try suite.benchmark("Vector Add 1K", vectorAddBenchmark, .{1000});
    try suite.benchmark("Vector Add 10K", vectorAddBenchmark, .{10000});

    // ArrayList operations (fallible)
    try suite.benchmarkFallible("ArrayList 1K", arrayListBenchmark, .{ testing.allocator, 1000 });
    try suite.benchmarkFallible("ArrayList 10K", arrayListBenchmark, .{ testing.allocator, 10000 });

    // HashMap operations (fallible)
    try suite.benchmarkFallible("HashMap 1K", hashMapBenchmark, .{ testing.allocator, 1000 });

    suite.printResults();
}
