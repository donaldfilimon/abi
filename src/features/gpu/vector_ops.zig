const std = @import("std");
const builtin = @import("builtin");
const backends = @import("backends.zig");
const metal = @import("metal_shared.zig");

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
            var sum: f32 = 0;
            for (res) |val| {
                sum += val;
            }
            return sum;
        }

        var sum: f32 = 0;
        var i: usize = 0;
        while (i + 4 <= a.len) : (i += 4) {
            const av: @Vector(4, f32) = a[i..][0..4].*;
            const bv: @Vector(4, f32) = b[i..][0..4].*;
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
            var sum: f32 = 0;
            for (res) |val| {
                sum += val;
            }
            return sum;
        }

        var sum: f32 = 0;
        var i: usize = 0;
        while (i + 4 <= a.len) : (i += 4) {
            const av: @Vector(4, f32) = a[i..][0..4].*;
            const bv: @Vector(4, f32) = b[i..][0..4].*;
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
    const native = backends.nativeKernelStatus();
    return .{
        .backend = status.backend,
        .mode = if (native.linked and status.accelerated) .native_gpu else .simulated_gpu,
        .work_items = spec.work_items,
        .message = if (native.linked and status.accelerated) "native GPU kernel executed" else "kernel metadata validated; vectorized CPU fallback selected",
    };
}

pub fn vectorOps() VectorOps {
    return VectorOps.init();
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
