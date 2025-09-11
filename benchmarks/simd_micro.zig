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

    // Euclidean distance
    timer.reset();
    const dist_val = abi.core.VectorOps.distance(a, b);
    const dist_ns = timer.read();

    // Cosine similarity
    timer.reset();
    const cos_val = abi.core.VectorOps.cosineSimilarity(a, b);
    const cos_ns = timer.read();

    // Simple scalar operations for comparison
    timer.reset();
    for (r, a, 0..) |*rv, av, i| {
        rv.* = av + b[i % b.len];
    }
    const add_ns = timer.read();

    timer.reset();
    var sum_val: f32 = 0;
    for (a) |v| sum_val += v;
    const sum_ns = timer.read();

    // Unused constants removed to avoid warnings
    const mean_val = if (a.len > 0) sum_val / @as(f32, @floatFromInt(a.len)) else 0;

    // Matrix multiply (simplified scalar version)
    timer.reset();
    const M: usize = 64;
    const K: usize = 32;
    const Ncol: usize = 32;
    const mat_a = try allocator.alloc(f32, M * K);
    defer allocator.free(mat_a);
    const mat_b = try allocator.alloc(f32, K * Ncol);
    defer allocator.free(mat_b);
    const mat_r = try allocator.alloc(f32, M * Ncol);
    defer allocator.free(mat_r);
    for (mat_a, 0..) |*v, i| v.* = @as(f32, @floatFromInt((i * 7) % 31)) * 0.03125;
    for (mat_b, 0..) |*v, i| v.* = @as(f32, @floatFromInt((i * 11) % 29)) * 0.03448;

    // Simple scalar matrix multiply
    for (0..M) |i| {
        for (0..Ncol) |j| {
            var sum: f32 = 0;
            for (0..K) |k| {
                sum += mat_a[i * K + k] * mat_b[k * Ncol + j];
            }
            mat_r[i * Ncol + j] = sum;
        }
    }
    const mm_ns = timer.read();

    std.debug.print("SIMD micro (N={d})\n  distance={d}ns dist_val={d:.3} cosine={d}ns cos_val={d:.3}\n  add={d}ns sum={d}ns sum_val={d:.3} mean={d:.3}\n  mm({d}x{d} * {d}x{d})={d}ns\n", .{ N, dist_ns, dist_val, cos_ns, cos_val, add_ns, sum_ns, sum_val, mean_val, M, K, K, Ncol, mm_ns });
}
