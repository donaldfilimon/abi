const std = @import("std");
const builtin = @import("builtin");
const backends = @import("backends.zig");
const metal = @import("metal_shared.zig");

/// Host-side vectorized sum. Used as CPU fallback and for tiny buffers.
/// Prefer `reduceSum` after Metal map kernels when Metal is initialized.
fn sumF32(values: []const f32) f32 {
    var sum: f32 = 0;
    var i: usize = 0;
    const VLen = comptime std.simd.suggestVectorLength(f32) orelse 4;
    while (i + VLen <= values.len) : (i += VLen) {
        const v: @Vector(VLen, f32) = values[i..][0..VLen].*;
        sum += @reduce(.Add, v);
    }
    while (i < values.len) : (i += 1) sum += values[i];
    return sum;
}

/// Prefer Metal threadgroup reduce (256-wide partials + host SIMD of partials)
/// when initialized; otherwise host `sumF32`. Multi-pass full-device tree
/// reduce remains Proposed.
fn reduceSum(values: []const f32) f32 {
    if (builtin.target.os.tag == .macos and metal.g_metal_context.initialized) {
        return metal.g_metal_context.runReduceSum(values) catch sumF32(values);
    }
    return sumF32(values);
}

pub const VectorOps = struct {
    backend: backends.BackendStatus,

    pub fn init() VectorOps {
        if (builtin.target.os.tag == .macos) {
            metal.g_metal_context.init(std.heap.page_allocator) catch |err| {
                std.log.warn("Metal initialization failed: {s}; using vectorized CPU fallback", .{@errorName(err)});
            };
        }
        return .{ .backend = backends.detectBackend() };
    }

    pub fn dot(self: VectorOps, a: []const f32, b: []const f32) !f32 {
        if (a.len != b.len) return error.DimensionMismatch;
        if (a.len == 0) return 0;

        if (self.backend.accelerated and builtin.target.os.tag == .macos and metal.g_metal_context.initialized) {
            var stack_buf: [4096]f32 = undefined;
            const res = if (a.len <= 4096)
                stack_buf[0..a.len]
            else
                try std.heap.page_allocator.alloc(f32, a.len);
            defer if (a.len > 4096) std.heap.page_allocator.free(res);

            try metal.g_metal_context.runKernel(metal.g_metal_context.dot_pipeline, a.len, a, b, res);
            return reduceSum(res);
        }

        var sum: f32 = 0;
        var i: usize = 0;
        const VLen = comptime std.simd.suggestVectorLength(f32) orelse 4;
        while (i + VLen <= a.len) : (i += VLen) {
            const av: @Vector(VLen, f32) = a[i..][0..VLen].*;
            const bv: @Vector(VLen, f32) = b[i..][0..VLen].*;
            sum += @reduce(.Add, av * bv);
        }
        while (i < a.len) : (i += 1) sum += a[i] * b[i];
        return sum;
    }

    pub fn squaredL2(self: VectorOps, a: []const f32, b: []const f32) !f32 {
        if (a.len != b.len) return error.DimensionMismatch;
        if (a.len == 0) return 0;

        if (self.backend.accelerated and builtin.target.os.tag == .macos and metal.g_metal_context.initialized) {
            var stack_buf: [4096]f32 = undefined;
            const res = if (a.len <= 4096)
                stack_buf[0..a.len]
            else
                try std.heap.page_allocator.alloc(f32, a.len);
            defer if (a.len > 4096) std.heap.page_allocator.free(res);

            try metal.g_metal_context.runKernel(metal.g_metal_context.l2_pipeline, a.len, a, b, res);
            return reduceSum(res);
        }

        var sum: f32 = 0;
        var i: usize = 0;
        const VLen = comptime std.simd.suggestVectorLength(f32) orelse 4;
        while (i + VLen <= a.len) : (i += VLen) {
            const av: @Vector(VLen, f32) = a[i..][0..VLen].*;
            const bv: @Vector(VLen, f32) = b[i..][0..VLen].*;
            const diff = av - bv;
            sum += @reduce(.Add, diff * diff);
        }
        while (i < a.len) : (i += 1) {
            const diff = a[i] - b[i];
            sum += diff * diff;
        }
        return sum;
    }

    pub fn cosineSimilarity(self: VectorOps, a: []const f32, b: []const f32) !f32 {
        if (a.len != b.len) return error.DimensionMismatch;
        if (a.len == 0) return 0;

        if (self.backend.accelerated and builtin.target.os.tag == .macos and metal.g_metal_context.initialized) {
            var stack_ab: [4096]f32 = undefined;
            var stack_aa: [4096]f32 = undefined;
            var stack_bb: [4096]f32 = undefined;
            const ab = if (a.len <= 4096) stack_ab[0..a.len] else try std.heap.page_allocator.alloc(f32, a.len);
            defer if (a.len > 4096) std.heap.page_allocator.free(ab);
            const aa_buf = if (a.len <= 4096) stack_aa[0..a.len] else try std.heap.page_allocator.alloc(f32, a.len);
            defer if (a.len > 4096) std.heap.page_allocator.free(aa_buf);
            const bb_buf = if (a.len <= 4096) stack_bb[0..a.len] else try std.heap.page_allocator.alloc(f32, a.len);
            defer if (a.len > 4096) std.heap.page_allocator.free(bb_buf);

            try metal.g_metal_context.runCosinePartsKernel(a.len, a, b, ab, aa_buf, bb_buf);
            const sum_ab = reduceSum(ab);
            const sum_aa = reduceSum(aa_buf);
            const sum_bb = reduceSum(bb_buf);
            if (sum_aa == 0 or sum_bb == 0) return 0;
            return sum_ab / @sqrt(sum_aa * sum_bb);
        }

        const ab = try self.dot(a, b);
        const aa = try self.dot(a, a);
        const bb = try self.dot(b, b);
        if (aa == 0 or bb == 0) return 0;
        return ab / @sqrt(aa * bb);
    }

    /// Cosine similarity of `query` against each row of `candidates`, written
    /// into `out` (len must equal candidates.len). The batched form of
    /// `cosineSimilarity`: the query norm is computed once and reused across the
    /// whole batch (instead of re-deriving it per pair), while each dot still
    /// routes through the same accelerated/SIMD path. Useful for scoring a query
    /// against a candidate set in one call.
    pub fn batchCosineSimilarity(self: VectorOps, query: []const f32, candidates: []const []const f32, out: []f32) !void {
        if (out.len != candidates.len) return error.DimensionMismatch;
        const q_norm_sq = try self.dot(query, query);
        const q_norm = @sqrt(q_norm_sq);
        for (candidates, out) |cand, *slot| {
            if (cand.len != query.len) return error.DimensionMismatch;
            const cc = try self.dot(cand, cand);
            if (q_norm == 0 or cc == 0) {
                slot.* = 0;
                continue;
            }
            const qc = try self.dot(query, cand);
            slot.* = qc / (q_norm * @sqrt(cc));
        }
    }
};

pub fn executeKernel(spec: backends.KernelSpec) !backends.KernelResult {
    if (spec.name.len == 0) return error.InvalidKernelName;
    const status = backends.detectBackend();
    return .{
        .backend = status.backend,
        .mode = .cpu_fallback,
        .work_items = spec.work_items,
        .message = "kernel metadata validated; vectorized CPU fallback selected",
    };
}

pub fn vectorOps() VectorOps {
    return VectorOps.init();
}

test "host sumF32 matches scalar reduce across SIMD width and tail" {
    // Length 10 forces both the vectorized loop and the scalar remainder.
    const values = [_]f32{ 0.5, -1.0, 2.25, 3.0, -0.75, 1.5, 0.0, 4.0, -2.0, 0.125 };
    var expected: f32 = 0;
    for (values) |v| expected += v;
    try std.testing.expectApproxEqAbs(expected, sumF32(&values), 1e-6);
    try std.testing.expectEqual(@as(f32, 0), sumF32(&.{}));
}

test "gpu vector ops: reduceSum matches host sum across multiple threadgroups" {
    _ = vectorOps(); // ensure Metal init on macOS when available
    var values: [520]f32 = undefined;
    var expected: f32 = 0;
    for (&values, 0..) |*slot, i| {
        slot.* = @as(f32, @floatFromInt(i % 7)) * 0.125 - 0.5;
        expected += slot.*;
    }
    try std.testing.expectApproxEqAbs(expected, reduceSum(&values), 1e-2);
    try std.testing.expectEqual(@as(f32, 0), reduceSum(&.{}));
}

test "gpu vector ops provide deterministic acceleration" {
    const ops = vectorOps();
    try std.testing.expectEqual(@as(f32, 32), try ops.dot(&.{ 1, 2, 3 }, &.{ 4, 5, 6 }));
    try std.testing.expectEqual(@as(f32, 27), try ops.squaredL2(&.{ 1, 2, 3 }, &.{ 4, 5, 6 }));
    try std.testing.expect((try ops.cosineSimilarity(&.{ 1, 0 }, &.{ 1, 0 })) == 1);
}

test "gpu vector ops: active backend matches scalar reference (CPU/GPU parity)" {
    // Whatever backend init() selected — the Metal kernel when initialized on
    // macOS, otherwise the vectorized CPU fallback — must agree with an
    // independent scalar reference. This is the mandated CPU/GPU parity check:
    // it validates the active path rather than asserting a specific backend.
    const ops = vectorOps();
    // Length 10 exercises both the 4-wide vector loop and the scalar tail.
    const a = [_]f32{ 0.5, -1.0, 2.25, 3.0, -0.75, 1.5, 0.0, 4.0, -2.0, 0.125 };
    const b = [_]f32{ 1.0, 2.0, -0.5, 0.25, 4.0, -1.0, 3.0, 0.5, 1.25, -3.0 };

    var ref_dot: f32 = 0;
    var ref_l2: f32 = 0;
    for (a, b) |x, y| {
        ref_dot += x * y;
        ref_l2 += (x - y) * (x - y);
    }

    try std.testing.expectApproxEqAbs(ref_dot, try ops.dot(&a, &b), 1e-3);
    try std.testing.expectApproxEqAbs(ref_l2, try ops.squaredL2(&a, &b), 1e-3);
}

test "gpu batched cosine similarity matches the pairwise result" {
    const ops = vectorOps();
    const query = [_]f32{ 1, 0, 0 };
    const c0 = [_]f32{ 1, 0, 0 };
    const c1 = [_]f32{ 0, 1, 0 };
    const c2 = [_]f32{ 1, 1, 0 };
    const candidates = [_][]const f32{ &c0, &c1, &c2 };

    var out: [3]f32 = undefined;
    try ops.batchCosineSimilarity(&query, &candidates, &out);

    try std.testing.expectApproxEqAbs(@as(f32, 1.0), out[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), out[1], 1e-6);
    try std.testing.expectApproxEqAbs(try ops.cosineSimilarity(&query, &c2), out[2], 1e-6);

    var bad: [1]f32 = undefined;
    try std.testing.expectError(error.DimensionMismatch, ops.batchCosineSimilarity(&query, &candidates, &bad));
}

test "generic kernel status does not claim native dispatch" {
    _ = vectorOps();
    const result = try executeKernel(.{ .name = "test.metadata_only", .work_items = 4 });
    try std.testing.expectEqual(backends.ExecutionMode.cpu_fallback, result.mode);
    try std.testing.expectEqualStrings("kernel metadata validated; vectorized CPU fallback selected", result.message);
}

test {
    std.testing.refAllDecls(@This());
}
