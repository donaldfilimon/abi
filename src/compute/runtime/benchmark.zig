//! Performance benchmark workloads
//!
//! Provides various benchmark workloads for testing compute performance.

const std = @import("std");
const workload = @import("../runtime/workload.zig");

pub const ComputeBenchmark = struct {
    name: []const u8,
    iterations: u64,
    work_fn: *const fn (iteration: u64) anyerror!void,
};

pub const BenchmarkResult = struct {
    name: []const u8,
    iterations: u64,
    total_ns: u64,
    avg_ns: f64,
    min_ns: u64,
    max_ns: u64,
    throughput_per_second: f64,
};

pub fn runBenchmark(_: std.mem.Allocator, benchmark: ComputeBenchmark) !BenchmarkResult {
    var total_ns: u64 = 0;
    var min_ns: u64 = std.math.maxInt(u64);
    var max_ns: u64 = 0;

    var i: u64 = 0;
    while (i < benchmark.iterations) : (i += 1) {
        var timer = try std.time.Timer.start();
        const start = timer.read();

        try benchmark.work_fn(i);

        const end = timer.read();
        const duration = end - start;

        total_ns += duration;
        min_ns = @min(min_ns, duration);
        max_ns = @max(max_ns, duration);
    }

    const avg_ns = @as(f64, @floatFromInt(total_ns)) / @as(f64, @floatFromInt(benchmark.iterations));
    const total_seconds = @as(f64, @floatFromInt(total_ns)) / 1_000_000_000.0;
    const throughput = if (total_seconds > 0) @as(f64, @floatFromInt(benchmark.iterations)) / total_seconds else 0;

    return BenchmarkResult{
        .name = benchmark.name,
        .iterations = benchmark.iterations,
        .total_ns = total_ns,
        .avg_ns = avg_ns,
        .min_ns = if (benchmark.iterations > 0) min_ns else 0,
        .max_ns = max_ns,
        .throughput_per_second = throughput,
    };
}

pub const MatrixMultBenchmark = struct {
    matrix_size: usize,
    iterations: u64 = 10,

    pub fn create(self: MatrixMultBenchmark, _: std.mem.Allocator) ComputeBenchmark {
        return ComputeBenchmark{
            .name = "matrix_multiplication",
            .iterations = self.iterations,
            .work_fn = struct {
                fn run(iteration: u64) anyerror!void {
                    _ = iteration;

                    const n: usize = 256;
                    var a: [256][256]f32 = undefined;
                    var b: [256][256]f32 = undefined;
                    var c: [256][256]f32 = undefined;

                    var i: usize = 0;
                    while (i < n) : (i += 1) {
                        var j: usize = 0;
                        while (j < n) : (j += 1) {
                            a[i][j] = @floatFromInt(i + j);
                            b[i][j] = @floatFromInt(i * j);
                            c[i][j] = 0;
                        }
                    }

                    i = 0;
                    while (i < n) : (i += 1) {
                        var j: usize = 0;
                        while (j < n) : (j += 1) {
                            var k: usize = 0;
                            while (k < n) : (k += 1) {
                                c[i][j] += a[i][k] * b[k][j];
                            }
                        }
                    }
                }
            }.run,
        };
    }
};

pub const MemoryAllocationBenchmark = struct {
    allocation_size: usize,
    iterations: u64 = 1000,

    pub fn create(self: MemoryAllocationBenchmark, allocator: std.mem.Allocator) ComputeBenchmark {
        _ = allocator;

        return ComputeBenchmark{
            .name = "memory_allocation",
            .iterations = self.iterations,
            .work_fn = struct {
                fn run(iteration: u64) anyerror!void {
                    _ = iteration;

                    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
                    defer arena.deinit();

                    const buffer = try arena.allocator().alloc(u8, 4096);
                    @memset(buffer, 0);
                }
            }.run,
        };
    }
};

pub const FibonacciBenchmark = struct {
    n: usize,
    iterations: u64 = 1000,

    pub fn create(self: FibonacciBenchmark, allocator: std.mem.Allocator) ComputeBenchmark {
        _ = allocator;

        return ComputeBenchmark{
            .name = "fibonacci",
            .iterations = self.iterations,
            .work_fn = struct {
                fn run(iteration: u64) anyerror!void {
                    _ = iteration;
                    _ = fibonacci(30);
                }

                fn fibonacci(n: usize) usize {
                    if (n <= 1) return n;
                    return fibonacci(n - 1) + fibonacci(n - 2);
                }
            }.run,
        };
    }
};

pub const HashingBenchmark = struct {
    data_size: usize,
    iterations: u64 = 10000,

    pub fn create(self: HashingBenchmark, allocator: std.mem.Allocator) ComputeBenchmark {
        _ = allocator;

        return ComputeBenchmark{
            .name = "hashing",
            .iterations = self.iterations,
            .work_fn = struct {
                fn run(iteration: u64) anyerror!void {
                    _ = iteration;

                    var data: [1024]u8 = undefined;
                    var out: [32]u8 = undefined;
                    std.crypto.hash.sha2.Sha256.hash(&data, &out, .{});
                }
            }.run,
        };
    }
};

pub fn printBenchmarkResults(result: BenchmarkResult) void {
    std.debug.print("\n=== Benchmark: {s} ===\n", .{result.name});
    std.debug.print("Iterations: {}\n", .{result.iterations});
    std.debug.print("Total time: {:.2} ms\n", .{@as(f64, @floatFromInt(result.total_ns)) / 1_000_000.0});
    std.debug.print("Avg time: {:.2} μs\n", .{result.avg_ns / 1_000.0});
    std.debug.print("Min time: {:.2} μs\n", .{@as(f64, @floatFromInt(result.min_ns)) / 1_000.0});
    std.debug.print("Max time: {:.2} μs\n", .{@as(f64, @floatFromInt(result.max_ns)) / 1_000.0});
    std.debug.print("Throughput: {:.2} ops/sec\n", .{result.throughput_per_second});
}
