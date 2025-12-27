const std = @import("std");
const benchmark = @import("mod.zig");

fn emptyTask(_: std.mem.Allocator) !void {
    @import("std").time.sleep(1 * std.time.ns_per_ms);
}

fn computeTask(_: std.mem.Allocator) !u64 {
    var sum: u64 = 0;
    var i: u64 = 0;
    while (i < 1000) : (i += 1) {
        sum += i;
    }
    return sum;
}

fn memoryTask(allocator: std.mem.Allocator) ![]u8 {
    return try allocator.alloc(u8, 1024);
}

fn vectorTask(_: std.mem.Allocator) !f32 {
    const vec_a = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const vec_b = [_]f32{ 4.0, 3.0, 2.0, 1.0 };
    return vec_a[0] * vec_b[0] + vec_a[1] * vec_b[1] + vec_a[2] * vec_b[2] + vec_a[3] * vec_b[3];
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    std.debug.print("=== ABI Benchmark Suite ===\n", .{});

    _ = try benchmark.executeBenchmark(allocator, "Empty Task (noop)", emptyTask);

    const compute_result = try benchmark.executeBenchmark(allocator, "Simple Compute (sum 1-1000)", computeTask);
    _ = allocator.free(compute_result.name);

    const mem_result = try benchmark.executeBenchmark(allocator, "Memory Allocation (1KB)", memoryTask);
    _ = allocator.free(mem_result.name);

    _ = try benchmark.executeBenchmark(allocator, "Vector Dot Product (4D)", vectorTask);

    std.debug.print("\n=== Benchmarks Complete ===\n", .{});
}
