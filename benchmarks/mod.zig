const std = @import("std");

pub const BenchmarkResult = struct {
    name: []const u8,
    iterations: u64,
    duration_ns: u64,
    ops_per_sec: f64,
    error_count: u64,
};

pub const BenchmarkSuite = struct {
    allocator: std.mem.Allocator,
    results: std.ArrayListUnmanaged(BenchmarkResult),

    pub fn init(allocator: std.mem.Allocator) BenchmarkSuite {
        return .{
            .allocator = allocator,
            .results = .empty,
        };
    }

    pub fn deinit(self: *BenchmarkSuite) void {
        for (self.results.items) |result| {
            self.allocator.free(result.name);
        }
        self.results.deinit(self.allocator);
    }

    pub fn runBenchmark(self: *BenchmarkSuite, comptime name: []const u8, benchmark_fn: anytype, args: anytype) !void {
        std.debug.print("Running: {s}\n", .{name});

        var timer = try std.time.Timer.start();

        var iterations: u64 = 0;
        var errors: u64 = 0;
        const start = timer.read();

        // Run benchmark for at least 1 second or 1000 iterations
        const min_duration_ns = 1_000_000_000; // 1 second
        const max_iterations = 100_000;

        while (iterations < max_iterations) {
            if (@call(.auto, benchmark_fn, args)) |_| {
                iterations += 1;
            } else |_| {
                errors += 1;
                iterations += 1;
            }

            if (timer.read() - start >= min_duration_ns) {
                break;
            }
        }

        const duration_ns = timer.read() - start;

        const seconds = @as(f64, @floatFromInt(duration_ns)) / 1_000_000_000.0;
        const ops_per_sec = if (seconds == 0) 0 else @as(f64, @floatFromInt(iterations)) / seconds;

        std.debug.print("  iterations: {d}\n", .{iterations});
        std.debug.print("  duration: {D} ns ({d:.3}s)\n", .{ duration_ns, seconds });
        std.debug.print("  avg: {D} ns/op\n", .{duration_ns / iterations});
        std.debug.print("  ops/sec: {d:.0}\n", .{ops_per_sec});
        if (errors > 0) {
            std.debug.print("  errors: {d}\n", .{errors});
        }
        std.debug.print("\n", .{});

        try self.results.append(self.allocator, BenchmarkResult{
            .name = try self.allocator.dupe(u8, name),
            .iterations = iterations,
            .duration_ns = duration_ns,
            .ops_per_sec = ops_per_sec,
            .error_count = errors,
        });
    }

    pub fn printSummary(self: *BenchmarkSuite) void {
        std.debug.print("=== ABI Framework Benchmark Results ===\n\n", .{});

        var total_ops: f64 = 0;
        var total_errors: u64 = 0;

        for (self.results.items) |result| {
            std.debug.print("{s}:\n", .{result.name});
            std.debug.print("  {d:.0} ops/sec, {d} errors\n", .{
                result.ops_per_sec,
                result.error_count,
            });
            total_ops += result.ops_per_sec;
            total_errors += result.error_count;
        }

        std.debug.print("\nSummary:\n", .{});
        std.debug.print("  Total benchmarks: {d}\n", .{self.results.items.len});
        std.debug.print("  Average ops/sec: {d:.0}\n", .{total_ops / @as(f64, @floatFromInt(self.results.items.len))});
        std.debug.print("  Total errors: {d}\n", .{total_errors});
    }
};
