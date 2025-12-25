const std = @import("std");
const benchmark = @import("benchmark.zig");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const results = try benchmark.runBenchmarks(allocator);
    defer allocator.free(results);

    std.debug.print("ABI Benchmarks\n", .{});
    for (results) |result| {
        std.debug.print(
            "  {s}: {d} iters, {d} ns, {d:.2} ops/sec\n",
            .{ result.name, result.iterations, result.duration_ns, result.ops_per_sec },
        );
    }
}
