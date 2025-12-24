const std = @import("std");
const abbey = @import("main.zig");

fn benchMatmul(allocator: std.mem.Allocator, size: usize) !void {
    var a = try abbey.workload.Matrix.init(allocator, size, size);
    defer a.deinit();
    var b = try abbey.workload.Matrix.init(allocator, size, size);
    defer b.deinit();
    var out = try abbey.workload.Matrix.init(allocator, size, size);
    defer out.deinit();

    for (0..size) |i| {
        for (0..size) |j| {
            a.set(i, j, @as(f32, @floatFromInt((i + j) % 7)));
            b.set(i, j, @as(f32, @floatFromInt((i * j) % 5)));
        }
    }

    var timer = try std.time.Timer.start();
    abbey.workload.matmulBlocked(&out, a, b, 16);
    const elapsed_ns = timer.read();

    const flops = 2.0 * @as(f64, @floatFromInt(size)) * @as(f64, @floatFromInt(size)) * @as(f64, @floatFromInt(size));
    const seconds = @as(f64, @floatFromInt(elapsed_ns)) / 1_000_000_000.0;
    const gflops = (flops / seconds) / 1_000_000_000.0;

    std.debug.print("matmul {d}x{d}: {d:.3} ms, {d:.2} GFLOP/s\n", .{
        size,
        size,
        @as(f64, @floatFromInt(elapsed_ns)) / 1_000_000.0,
        gflops,
    });
}

pub fn main() !void {
    const allocator = std.heap.page_allocator;
    std.debug.print("Abbey Benchmark Suite\n", .{});

    try benchMatmul(allocator, 64);
    try benchMatmul(allocator, 128);
}
