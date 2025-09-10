const std = @import("std");
const abi = @import("abi");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var timer = try std.time.Timer.start();

    const N = 1_000_000; // 1M elements
    var a = try allocator.alloc(f32, N);
    defer allocator.free(a);
    var b = try allocator.alloc(f32, N);
    defer allocator.free(b);
    var r = try allocator.alloc(f32, N);
    defer allocator.free(r);

    for (a, 0..) |*v, i| v.* = @floatFromInt(i % 100);
    for (b, 0..) |*v, i| v.* = @floatFromInt((i * 3) % 97);

    // add
    timer.reset();
    abi.simd.add(r, a, b);
    const add_ns = timer.read();

    // multiply
    timer.reset();
    abi.simd.multiply(r, a, b);
    const mul_ns = timer.read();

    // dot product
    timer.reset();
    const dot = abi.simd.dotProduct(a, b);
    const dot_ns = timer.read();

    // l1 distance
    timer.reset();
    const l1 = abi.simd.l1Distance(a, b);
    const l1_ns = timer.read();

    std.debug.print("SIMD micro (N={d}): add={d}ns mul={d}ns dot={d}ns l1={d}ns dot={d:.3} l1={d:.3}\n", .{ N, add_ns, mul_ns, dot_ns, l1_ns, dot, l1 });
}
