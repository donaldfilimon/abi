//! Benchmark demo executable
//!
//! Runs various performance benchmarks and prints results.

const std = @import("std");
const abi = @import("abi");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    std.debug.print("ABI Performance Benchmarks\n", .{});
    std.debug.print("===========================\n\n", .{});

    // Matrix multiplication benchmark
    const matrix_bench = abi.compute.MatrixMultBenchmark{
        .matrix_size = 256,
        .iterations = 100,
    };
    const matrix_benchmark = matrix_bench.create(allocator);
    const matrix_result = try abi.compute.runBenchmark(allocator, matrix_benchmark);
    abi.compute.printBenchmarkResults(matrix_result);

    // Memory allocation benchmark
    const mem_bench = abi.compute.MemoryAllocationBenchmark{
        .allocation_size = 4096,
        .iterations = 10000,
    };
    const mem_benchmark = mem_bench.create(allocator);
    const mem_result = try abi.compute.runBenchmark(allocator, mem_benchmark);
    abi.compute.printBenchmarkResults(mem_result);

    // Fibonacci benchmark
    const fib_bench = abi.compute.FibonacciBenchmark{
        .n = 30,
        .iterations = 1000,
    };
    const fib_benchmark = fib_bench.create(allocator);
    const fib_result = try abi.compute.runBenchmark(allocator, fib_benchmark);
    abi.compute.printBenchmarkResults(fib_result);

    // Hashing benchmark
    const hash_bench = abi.compute.HashingBenchmark{
        .data_size = 1024,
        .iterations = 10000,
    };
    const hash_benchmark = hash_bench.create(allocator);
    const hash_result = try abi.compute.runBenchmark(allocator, hash_benchmark);
    abi.compute.printBenchmarkResults(hash_result);

    std.debug.print("\n===========================\n", .{});
    std.debug.print("Benchmarks complete!\n", .{});
}
