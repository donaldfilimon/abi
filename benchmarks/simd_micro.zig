const std = @import("std");
const abi = @import("abi");

fn fillLinear(buf: []f32, mul: f32) void {
    for (buf, 0..) |*v, i| v.* = mul * @as(f32, @floatFromInt(i % 100));
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var timer = try std.time.Timer.start();

    const N = 1_000_000; // 1M elements
    const a = try allocator.alloc(f32, N);
    defer allocator.free(a);
    const b = try allocator.alloc(f32, N);
    defer allocator.free(b);
    const r = try allocator.alloc(f32, N);
    defer allocator.free(r);

    fillLinear(a, 1.0);
    fillLinear(b, 0.5);

    // add
    timer.reset();
    abi.simd.add(r, a, b);
    const add_ns = timer.read();

    // multiply
    timer.reset();
    abi.simd.multiply(r, a, b);
    const mul_ns = timer.read();

    // scale
    timer.reset();
    abi.simd.scale(r, a, 1.2345);
    const scale_ns = timer.read();

    // normalize
    timer.reset();
    abi.simd.normalize(r, a);
    const norm_ns = timer.read();

    // sum/mean/variance/stddev
    timer.reset();
    var sum_val = abi.simd.sum(a);
    const sum_ns = timer.read();
    const mean_val = abi.simd.mean(a);
    timer.reset();
    const var_val = abi.simd.variance(a);
    const var_ns = timer.read();
    const stddev_val = abi.simd.stddev(a);

    // dot product
    timer.reset();
    var dot = abi.simd.dotProduct(a, b);
    const dot_ns = timer.read();

    // l1 distance
    timer.reset();
    var l1 = abi.simd.l1Distance(a, b);
    const l1_ns = timer.read();

    // small matrix multiply (256x64) * (64x64)
    const M: usize = 256;
    const K: usize = 64;
    const Ncol: usize = 64;
    const mat_a = try allocator.alloc(f32, M * K);
    defer allocator.free(mat_a);
    const mat_b = try allocator.alloc(f32, K * Ncol);
    defer allocator.free(mat_b);
    var mat_r = try allocator.alloc(f32, M * Ncol);
    defer allocator.free(mat_r);
    for (mat_a, 0..) |*v, i| v.* = @as(f32, @floatFromInt((i * 7) % 31)) * 0.03125;
    for (mat_b, 0..) |*v, i| v.* = @as(f32, @floatFromInt((i * 11) % 29)) * 0.03448;
    timer.reset();
    abi.simd.matrixMultiply(mat_r, mat_a, mat_b, M, K, Ncol);
    const mm_ns = timer.read();

    std.debug.print(
        "SIMD micro (N={d})\n  add={d}ns mul={d}ns scale={d}ns norm={d}ns\n  sum={d}ns sum_val={d:.3} mean={d:.3} var={d:.3} (var_ns={d}ns) stddev={d:.3}\n  dot={d}ns dot_val={d:.3} l1={d}ns l1_val={d:.3}\n  mm(256x64 * 64x64)={d}ns\n",
        .{ N, add_ns, mul_ns, scale_ns, norm_ns, sum_ns, mean_val, var_val, var_ns, stddev_val, dot_ns, l1_ns, mm_ns },
    );
}


