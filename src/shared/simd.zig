const std = @import("std");

pub const VectorOps = struct {
    pub fn dot(a: []const f32, b: []const f32) f32 {
        const len = @min(a.len, b.len);
        var sum: f32 = 0;
        var i: usize = 0;
        while (i < len) : (i += 1) {
            sum += a[i] * b[i];
        }
        return sum;
    }

    pub fn l2Norm(a: []const f32) f32 {
        var sum: f32 = 0;
        for (a) |value| {
            sum += value * value;
        }
        return std.math.sqrt(sum);
    }

    pub fn cosineSimilarity(a: []const f32, b: []const f32) f32 {
        const denom = l2Norm(a) * l2Norm(b);
        if (denom == 0) return 0;
        return dot(a, b) / denom;
    }

    pub fn add(a: []const f32, b: []const f32, out: []f32) void {
        const len = @min(@min(a.len, b.len), out.len);
        var i: usize = 0;
        while (i < len) : (i += 1) {
            out[i] = a[i] + b[i];
        }
    }

    pub fn scale(a: []const f32, scalar: f32, out: []f32) void {
        const len = @min(a.len, out.len);
        var i: usize = 0;
        while (i < len) : (i += 1) {
            out[i] = a[i] * scalar;
        }
    }

    pub fn normalizeInPlace(a: []f32) void {
        const norm = l2Norm(a);
        if (norm == 0) return;
        for (a) |*value| {
            value.* /= norm;
        }
    }
};

test "vector ops basics" {
    const a = [_]f32{ 1, 2, 3 };
    const b = [_]f32{ 4, 5, 6 };
    const dot = VectorOps.dot(&a, &b);
    try std.testing.expect(std.math.approxEqAbs(f32, dot, 32.0, 0.0001));

    var out: [3]f32 = undefined;
    VectorOps.add(&a, &b, out[0..]);
    try std.testing.expectEqualSlices(f32, &.{ 5, 7, 9 }, &out);

    VectorOps.scale(&a, 2.0, out[0..]);
    try std.testing.expectEqualSlices(f32, &.{ 2, 4, 6 }, &out);
}

test "vector ops normalize" {
    var values = [_]f32{ 3, 4 };
    VectorOps.normalizeInPlace(values[0..]);
    const norm = VectorOps.l2Norm(values[0..]);
    try std.testing.expect(std.math.approxEqAbs(f32, norm, 1.0, 0.0001));
}
