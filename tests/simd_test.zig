//! SIMD Distance Function Tests

const std = @import("std");
const simd = @import("../src/core/database/simd.zig");

test "simd width is reasonable" {
    try std.testing.expect(simd.simd_width >= 4);
}

test "cosine similarity - parallel vectors" {
    const a = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 };
    const b = [_]f32{ 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0 };
    const sim = simd.Distance.cosine(&a, &b);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), sim, 1e-5);
}

test "cosine similarity - anti-parallel" {
    const a = [_]f32{ 1.0, 0.0, 0.0, 0.0 };
    const b = [_]f32{ -1.0, 0.0, 0.0, 0.0 };
    const sim = simd.Distance.cosine(&a, &b);
    try std.testing.expectApproxEqAbs(@as(f32, -1.0), sim, 1e-5);
}

test "l2 squared - zero distance" {
    const a = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const dist = simd.Distance.l2Squared(&a, &a);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), dist, 1e-5);
}

test "inner product commutative" {
    const a = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const b = [_]f32{ 5.0, 6.0, 7.0, 8.0 };
    const ab = simd.Distance.innerProduct(&a, &b);
    const ba = simd.Distance.innerProduct(&b, &a);
    try std.testing.expectApproxEqAbs(ab, ba, 1e-5);
}

test "binary quantization preserves signs" {
    const src = [_]f32{ 1.0, -1.0, 0.5, -0.5, 0.0, 1.0, -1.0, 0.1 };
    var dst: [1]u64 = undefined;
    simd.Quantize.toBinary(&src, &dst);
    // bit 0 should be set (1.0 > 0)
    try std.testing.expect(dst[0] & 1 == 1);
    // bit 1 should not be set (-1.0 <= 0)
    try std.testing.expect(dst[0] & 2 == 0);
}
