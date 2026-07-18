const std = @import("std");
const builtin = @import("builtin");
const backends = @import("backends.zig");
const metal = @import("metal_shared.zig");

/// Host-side vectorized sum. Used as CPU fallback and for tiny buffers.
/// Prefer Metal map+reduce / fused cosine when Metal is initialized.
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

/// Prefer Metal multi-pass threadgroup reduce (256-wide until one scalar)
/// when initialized; otherwise host `sumF32`. On Metal failure, log and use
/// the explicit CPU fallback — never swallow silently. Broader kernels /
/// CUDA remain Proposed.
fn reduceSum(values: []const f32) f32 {
    if (builtin.target.os.tag == .macos and metal.g_metal_context.initialized) {
        return metal.g_metal_context.runReduceSum(values) catch |err| {
            std.log.warn("Metal reduceSum failed ({s}); using vectorized CPU fallback", .{@errorName(err)});
            return sumF32(values);
        };
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
            return metal.g_metal_context.runMapAndReduce(metal.g_metal_context.dot_pipeline, a.len, a, b) catch |err| {
                std.log.warn("Metal dot map+reduce failed ({s}); using vectorized CPU fallback", .{@errorName(err)});
                return cpuDot(a, b);
            };
        }

        return cpuDot(a, b);
    }

    pub fn squaredL2(self: VectorOps, a: []const f32, b: []const f32) !f32 {
        if (a.len != b.len) return error.DimensionMismatch;
        if (a.len == 0) return 0;

        if (self.backend.accelerated and builtin.target.os.tag == .macos and metal.g_metal_context.initialized) {
            return metal.g_metal_context.runMapAndReduce(metal.g_metal_context.l2_pipeline, a.len, a, b) catch |err| {
                std.log.warn("Metal L2 map+reduce failed ({s}); using vectorized CPU fallback", .{@errorName(err)});
                return cpuSquaredL2(a, b);
            };
        }

        return cpuSquaredL2(a, b);
    }

    pub fn cosineSimilarity(self: VectorOps, a: []const f32, b: []const f32) !f32 {
        if (a.len != b.len) return error.DimensionMismatch;
        if (a.len == 0) return 0;

        if (self.backend.accelerated and builtin.target.os.tag == .macos and metal.g_metal_context.initialized) {
            const sums = metal.g_metal_context.runCosineFused(a.len, a, b) catch |err| {
                std.log.warn("Metal fused cosine failed ({s}); using vectorized CPU fallback", .{@errorName(err)});
                return cpuCosine(self, a, b);
            };
            if (sums.aa == 0 or sums.bb == 0) return 0;
            return sums.ab / @sqrt(sums.aa * sums.bb);
        }

        return cpuCosine(self, a, b);
    }

    /// Cosine similarity of `query` against each row of `candidates`, written
    /// into `out` (len must equal candidates.len). The batched form of
    /// `cosineSimilarity`: the query norm is computed once and reused across the
    /// whole batch (instead of re-deriving it per pair), while each dot still
    /// routes through the same accelerated/SIMD path. Useful for scoring a query
    /// against a candidate set in one call.
    pub fn batchCosineSimilarity(self: VectorOps, query: []const f32, candidates: []const []const f32, out: []f32) !void {
        if (out.len != candidates.len) return error.DimensionMismatch;
        for (candidates) |cand| {
            if (cand.len != query.len) return error.DimensionMismatch;
        }
        if (candidates.len == 0) return;

        if (self.backend.accelerated and builtin.target.os.tag == .macos and metal.g_metal_context.initialized) {
            if (self.runBatchedMetal(query, candidates, out)) |_| {
                return;
            } else |err| {
                std.log.warn("Metal batch cosine failed ({s}); using per-pair fallback", .{@errorName(err)});
            }
        }

        return self.batchCosineSimilarityFallback(query, candidates, out);
    }

    /// Flattens `candidates` (non-contiguous, pointing into HNSW/caller
    /// storage) into one contiguous host buffer of `candidates.len * query.len`
    /// floats and dispatches the fused batched Metal kernel in a single
    /// command buffer. The flatten is a real host-side copy — not free — but
    /// it replaces up to `2*candidates.len + 1` synchronous per-pair GPU
    /// round-trips (see `batchCosineSimilarityFallback`) with exactly one.
    /// Correctness-parity proven (see tests below); no speedup claim.
    fn runBatchedMetal(self: VectorOps, query: []const f32, candidates: []const []const f32, out: []f32) !void {
        _ = self;
        const allocator = std.heap.page_allocator;
        const d = query.len;
        const n = candidates.len;
        const flat = try allocator.alloc(f32, n * d);
        defer allocator.free(flat);
        for (candidates, 0..) |cand, i| {
            @memcpy(flat[i * d ..][0..d], cand);
        }
        try metal.g_metal_context.runBatchCosineFused(query, flat, n, d, out);
    }

    /// Per-pair CPU/Metal-dot fallback: identical math to the batched kernel,
    /// used when Metal batching is unavailable or fails. Never silently wrong
    /// — callers are warned before falling back here.
    fn batchCosineSimilarityFallback(self: VectorOps, query: []const f32, candidates: []const []const f32, out: []f32) !void {
        const q_norm_sq = try self.dot(query, query);
        const q_norm = @sqrt(q_norm_sq);
        for (candidates, out) |cand, *slot| {
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

fn cpuDot(a: []const f32, b: []const f32) f32 {
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

fn cpuSquaredL2(a: []const f32, b: []const f32) f32 {
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

fn cpuCosine(self: VectorOps, a: []const f32, b: []const f32) !f32 {
    _ = self;
    const ab = cpuDot(a, b);
    const aa = cpuDot(a, a);
    const bb = cpuDot(b, b);
    if (aa == 0 or bb == 0) return 0;
    return ab / @sqrt(aa * bb);
}

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

    var ref_ab: f32 = 0;
    var ref_aa: f32 = 0;
    var ref_bb: f32 = 0;
    for (a, b) |x, y| {
        ref_ab += x * y;
        ref_aa += x * x;
        ref_bb += y * y;
    }
    const ref_cos = ref_ab / @sqrt(ref_aa * ref_bb);
    try std.testing.expectApproxEqAbs(ref_cos, try ops.cosineSimilarity(&a, &b), 1e-3);
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

test "gpu batched cosine similarity matches an independent scalar reference" {
    // Independent of both batchCosineSimilarityFallback and cosineSimilarity —
    // a scalar loop written fresh here so a shared bug in either production
    // path can't hide behind a matching test. Exercises N=5 candidates of
    // D=17 (forces the SIMD-width tail in the CPU path; the strided Metal
    // load loop `for (i = lid; i < dim; i += 256)` runs at most once per
    // thread at this width — see the D=1000 test below for the multi-
    // iteration case that matters at realistic HNSW dimensions).
    const ops = vectorOps();

    var query: [17]f32 = undefined;
    for (&query, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i % 5)) * 0.3 - 0.6;

    var storage: [5][17]f32 = undefined;
    var candidates: [5][]const f32 = undefined;
    for (&storage, 0..) |*row, r| {
        for (row, 0..) |*v, i| {
            v.* = @as(f32, @floatFromInt((i + r * 3) % 7)) * 0.2 - 0.7;
        }
        candidates[r] = row;
    }

    var expected: [5]f32 = undefined;
    for (candidates, 0..) |cand, r| {
        var qc: f32 = 0;
        var qq: f32 = 0;
        var cc: f32 = 0;
        for (query, cand) |qv, cv| {
            qc += qv * cv;
            qq += qv * qv;
            cc += cv * cv;
        }
        expected[r] = if (qq == 0 or cc == 0) 0 else qc / (@sqrt(qq) * @sqrt(cc));
    }

    var out: [5]f32 = undefined;
    try ops.batchCosineSimilarity(&query, &candidates, &out);

    for (expected, out) |exp, got| {
        try std.testing.expectApproxEqAbs(exp, got, 1e-4);
    }
}

test "gpu batched cosine similarity matches an independent scalar reference at D>256 (multi-stride load)" {
    // D=1000 (> the fixed 256-thread threadgroup) forces every thread in
    // batch_cosine_kernel's strided load loop
    // (`for (i = lid; i < dim; i += 256)`) to accumulate multiple elements
    // per lane instead of at most one — the realistic-HNSW-dimension path
    // (D=768/1536 per the scoping review) that the D=17 test above cannot
    // exercise. Reference is an independent scalar loop, not the CPU
    // fallback function.
    const allocator = std.testing.allocator;
    const d: usize = 1000;
    const n: usize = 6;

    const query = try allocator.alloc(f32, d);
    defer allocator.free(query);
    for (query, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i % 11)) * 0.11 - 0.55;

    var candidates: [n][]f32 = undefined;
    for (&candidates, 0..) |*row, r| {
        row.* = try allocator.alloc(f32, d);
        for (row.*, 0..) |*v, i| {
            v.* = @as(f32, @floatFromInt((i + r * 37) % 13)) * 0.09 - 0.6;
        }
    }
    defer for (candidates) |row| allocator.free(row);

    var candidate_slices: [n][]const f32 = undefined;
    for (candidates, 0..) |row, i| candidate_slices[i] = row;

    var expected: [n]f32 = undefined;
    for (candidate_slices, 0..) |cand, r| {
        var qc: f32 = 0;
        var qq: f32 = 0;
        var cc: f32 = 0;
        for (query, cand) |qv, cv| {
            qc += qv * cv;
            qq += qv * qv;
            cc += cv * cv;
        }
        expected[r] = if (qq == 0 or cc == 0) 0 else qc / (@sqrt(qq) * @sqrt(cc));
    }

    const ops = vectorOps();
    var out: [n]f32 = undefined;
    try ops.batchCosineSimilarity(query, &candidate_slices, &out);

    for (expected, out) |exp, got| {
        try std.testing.expectApproxEqAbs(exp, got, 1e-3);
    }
}

test "gpu batched cosine similarity rejects a mismatched candidate dimension" {
    const ops = vectorOps();
    const query = [_]f32{ 1, 0, 0 };
    const c0 = [_]f32{ 1, 0, 0 };
    const bad = [_]f32{ 1, 0 }; // wrong dimension
    const candidates = [_][]const f32{ &c0, &bad };

    var out: [2]f32 = undefined;
    try std.testing.expectError(error.DimensionMismatch, ops.batchCosineSimilarity(&query, &candidates, &out));
}

test "cpu dot product with known vectors" {
    const a = [_]f32{ 1, 0, 0 };
    const b = [_]f32{ 0, 1, 0 };
    try std.testing.expectEqual(@as(f32, 0), cpuDot(&a, &b));

    const c = [_]f32{ 2, 3, 4 };
    const d = [_]f32{ 5, 6, 7 };
    // 2*5 + 3*6 + 4*7 = 10 + 18 + 28 = 56
    try std.testing.expectEqual(@as(f32, 56), cpuDot(&c, &d));
}

test "cpu squaredL2 with known vectors" {
    const a = [_]f32{ 1, 2, 3 };
    const b = [_]f32{ 4, 6, 3 };
    // (1-4)^2 + (2-6)^2 + (3-3)^2 = 9 + 16 + 0 = 25
    try std.testing.expectEqual(@as(f32, 25), cpuSquaredL2(&a, &b));

    const c = [_]f32{ 1, 0, 0 };
    const d = [_]f32{ 1, 0, 0 };
    try std.testing.expectEqual(@as(f32, 0), cpuSquaredL2(&c, &d));
}

test "cpu cosine similarity: parallel = 1.0, orthogonal = 0.0" {
    const ops = vectorOps();
    const x = [_]f32{ 1, 0, 0 };
    const y = [_]f32{ 0, 1, 0 };
    const z = [_]f32{ 3, 0, 0 };

    try std.testing.expectApproxEqAbs(@as(f32, 1.0), try ops.cosineSimilarity(&x, &z), 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), try ops.cosineSimilarity(&x, &y), 1e-6);
}

test "cpu cosine similarity: zero vector returns 0" {
    const ops = vectorOps();
    const zero = [_]f32{ 0, 0, 0 };
    const v = [_]f32{ 1, 2, 3 };
    try std.testing.expectEqual(@as(f32, 0), try ops.cosineSimilarity(&zero, &v));
    try std.testing.expectEqual(@as(f32, 0), try ops.cosineSimilarity(&v, &zero));
}

test "cpu dot with negative values" {
    const a = [_]f32{ -1, 2, -3 };
    const b = [_]f32{ 4, -5, 6 };
    // -4 + -10 + -18 = -32
    try std.testing.expectEqual(@as(f32, -32), cpuDot(&a, &b));
}

test "reduceSum CPU fallback matches scalar reference" {
    var values: [256]f32 = undefined;
    var expected: f32 = 0;
    for (&values, 0..) |*slot, i| {
        slot.* = @as(f32, @floatFromInt(i)) * 0.5 - 64.0;
        expected += slot.*;
    }
    try std.testing.expectApproxEqAbs(expected, reduceSum(&values), 1e-2);
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
