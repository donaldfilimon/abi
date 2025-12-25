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
};
