const std = @import("std");
const build_options = @import("build_options");
const gpu = if (build_options.feat_gpu) @import("../gpu/mod.zig") else @import("../gpu/stub.zig");

pub const Candidate = struct {
    id: u32,
    distance: f32,
};

pub fn lessDistance(_: void, lhs: Candidate, rhs: Candidate) bool {
    return lhs.distance < rhs.distance;
}

pub fn cosineDistanceSIMD(a: []const f32, b: []const f32) f32 {
    if (a.len != b.len) return 1.0;
    const len = a.len;
    if (len == 0) return 1.0;

    const simd_width = std.simd.suggestVectorLength(f32) orelse 4;

    var dot: f32 = 0;
    var norm_a: f32 = 0;
    var norm_b: f32 = 0;

    const simd_len = (len / simd_width) * simd_width;

    var i: usize = 0;
    while (i < simd_len) : (i += simd_width) {
        const va: @Vector(simd_width, f32) = a[i .. i + simd_width][0..simd_width].*;
        const vb: @Vector(simd_width, f32) = b[i .. i + simd_width][0..simd_width].*;
        dot += @reduce(.Add, va * vb);
        norm_a += @reduce(.Add, va * va);
        norm_b += @reduce(.Add, vb * vb);
    }

    while (i < len) : (i += 1) {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }

    const denom = @sqrt(norm_a) * @sqrt(norm_b);
    if (denom == 0) return 1.0;
    return 1.0 - (dot / denom);
}

pub fn cosineDistanceWithOps(ops: gpu.VectorOps, a: []const f32, b: []const f32) f32 {
    if (a.len != b.len) return 1.0;
    const similarity = ops.cosineSimilarity(a, b) catch |err| {
        std.log.warn("gpu vector cosine distance failed: {s}; using SIMD fallback", .{@errorName(err)});
        return cosineDistanceSIMD(a, b);
    };
    return 1.0 - similarity;
}

pub fn batchCosineDistancesWithOps(
    allocator: std.mem.Allocator,
    ops: gpu.VectorOps,
    query: []const f32,
    candidates: []const []const f32,
    out: []f32,
) !void {
    if (out.len != candidates.len) return error.DimensionMismatch;
    if (candidates.len == 0) return;
    for (candidates) |candidate| {
        if (candidate.len != query.len) return error.DimensionMismatch;
    }

    const similarities = try allocator.alloc(f32, candidates.len);
    defer allocator.free(similarities);

    ops.batchCosineSimilarity(query, candidates, similarities) catch |err| {
        std.log.warn("gpu vector batch cosine distance failed: {s}; using SIMD fallback", .{@errorName(err)});
        for (candidates, out) |candidate, *slot| {
            slot.* = cosineDistanceSIMD(candidate, query);
        }
        return;
    };

    for (similarities, out) |similarity, *slot| {
        slot.* = 1.0 - similarity;
    }
}

test "cosineDistanceSIMD identical vectors" {
    const a = [_]f32{ 1.0, 0.0, 0.0, 0.0 };
    const b = [_]f32{ 1.0, 0.0, 0.0, 0.0 };
    const dist = cosineDistanceSIMD(&a, &b);
    try std.testing.expect(dist < 0.001);
}

test "cosineDistanceSIMD orthogonal vectors" {
    const a = [_]f32{ 1.0, 0.0, 0.0, 0.0 };
    const b = [_]f32{ 0.0, 1.0, 0.0, 0.0 };
    const dist = cosineDistanceSIMD(&a, &b);
    try std.testing.expect(dist > 0.99);
}

test "cosineDistanceWithOps matches SIMD fallback" {
    const ops = gpu.vectorOps();
    const a = [_]f32{ 0.25, 0.5, 0.75, 1.0 };
    const b = [_]f32{ 1.0, 0.75, 0.5, 0.25 };
    const accelerated = cosineDistanceWithOps(ops, &a, &b);
    const simd = cosineDistanceSIMD(&a, &b);
    try std.testing.expect(@abs(accelerated - simd) < 0.0001);
    try std.testing.expectEqual(@as(f32, 1.0), cosineDistanceWithOps(ops, &a, &.{ 1.0, 2.0 }));
}

test "batchCosineDistancesWithOps matches pairwise distances" {
    const ops = gpu.vectorOps();
    const query = [_]f32{ 1.0, 0.0, 0.0, 0.0 };
    const c0 = [_]f32{ 1.0, 0.0, 0.0, 0.0 };
    const c1 = [_]f32{ 0.0, 1.0, 0.0, 0.0 };
    const c2 = [_]f32{ 0.5, 0.5, 0.0, 0.0 };
    const candidates = [_][]const f32{ &c0, &c1, &c2 };

    var out: [3]f32 = undefined;
    try batchCosineDistancesWithOps(std.testing.allocator, ops, &query, &candidates, &out);

    try std.testing.expectApproxEqAbs(cosineDistanceWithOps(ops, &c0, &query), out[0], 1e-6);
    try std.testing.expectApproxEqAbs(cosineDistanceWithOps(ops, &c1, &query), out[1], 1e-6);
    try std.testing.expectApproxEqAbs(cosineDistanceWithOps(ops, &c2, &query), out[2], 1e-6);
}

test "batchCosineDistancesWithOps rejects mismatched result or candidate dimensions" {
    const ops = gpu.vectorOps();
    const query = [_]f32{ 1.0, 0.0, 0.0 };
    const c0 = [_]f32{ 1.0, 0.0, 0.0 };
    const bad = [_]f32{ 1.0, 0.0 };
    const candidates = [_][]const f32{ &c0, &bad };

    var out: [2]f32 = undefined;
    try std.testing.expectError(error.DimensionMismatch, batchCosineDistancesWithOps(std.testing.allocator, ops, &query, &candidates, &out));

    const valid_candidates = [_][]const f32{&c0};
    try std.testing.expectError(error.DimensionMismatch, batchCosineDistancesWithOps(std.testing.allocator, ops, &query, &valid_candidates, &out));
}

test {
    std.testing.refAllDecls(@This());
}
