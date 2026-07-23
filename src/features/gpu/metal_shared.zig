//! Shared Metal context for vector_ops / backends / compute_api.
//! ObjC FFI helpers: metal_objc.zig. Kernel dispatch + MetalContext: metal_kernels.zig.
const builtin = @import("builtin");
const std = @import("std");
const metal_kernels = @import("metal_kernels.zig");

pub const MetalContext = metal_kernels.MetalContext;

/// Global Metal runtime context. Call sites (vector_ops / backends / compute_api)
/// continue to use `metal.g_metal_context` unchanged.
pub var g_metal_context = MetalContext{};

/// Skip the calling test when Metal is unavailable (off-macOS, or macOS
/// without a usable device). Mirrors the active-backend pattern in
/// vector_ops.zig: the GPU path is exercised only where it can actually run;
/// headless/CI stays green because the test returns without asserting.
fn ensureMetalInitialized() bool {
    if (comptime builtin.target.os.tag != .macos) return false;
    if (g_metal_context.initialized) return true;
    g_metal_context.init(std.heap.page_allocator) catch return false;
    return g_metal_context.initialized;
}

test "metal reduceSum matches scalar reference" {
    if (!ensureMetalInitialized()) return;
    const values = [_]f32{ 0.5, -1.0, 2.25, 3.0, -0.75, 1.5, 0.0, 4.0, -2.0, 0.125, 1.0, -3.0 };
    var expected: f32 = 0;
    for (values) |v| expected += v;
    const got = try g_metal_context.runReduceSum(&values);
    try std.testing.expectApproxEqAbs(expected, got, 1e-2);
}

test "metal reduceMax matches scalar reference" {
    if (!ensureMetalInitialized()) return;
    const values = [_]f32{ 0.5, -1.0, 2.25, 3.0, -0.75, 1.5, 0.0, 4.0, -2.0, 0.125, 1.0, -3.0 };
    var expected: f32 = values[0];
    for (values[1..]) |v| {
        if (v > expected) expected = v;
    }
    const got = try g_metal_context.runReduceMax(&values);
    try std.testing.expectApproxEqAbs(expected, got, 1e-4);
}

test "metal map+reduce dot matches scalar reference" {
    if (!ensureMetalInitialized()) return;
    const a = [_]f32{ 1.0, 2.0, -0.5, 0.25, 4.0, -1.0, 3.0, 0.5, 1.25, -3.0 };
    const b = [_]f32{ 0.5, -1.0, 2.25, 3.0, -0.75, 1.5, 0.0, 4.0, -2.0, 0.125 };
    var expected: f32 = 0;
    for (a, b) |x, y| expected += x * y;
    const got = try g_metal_context.runMapAndReduce(
        g_metal_context.dot_pipeline,
        a.len,
        &a,
        &b,
    );
    try std.testing.expectApproxEqAbs(expected, got, 1e-2);
}

test "metal fused cosine returns correct ab/aa/bb" {
    if (!ensureMetalInitialized()) return;
    const a = [_]f32{ 0.5, -1.0, 2.25, 3.0, -0.75, 1.5, 0.0, 4.0, -2.0, 0.125 };
    const b = [_]f32{ 1.0, 2.0, -0.5, 0.25, 4.0, -1.0, 3.0, 0.5, 1.25, -3.0 };
    var ref_ab: f32 = 0;
    var ref_aa: f32 = 0;
    var ref_bb: f32 = 0;
    for (a, b) |x, y| {
        ref_ab += x * y;
        ref_aa += x * x;
        ref_bb += y * y;
    }
    const sums = try g_metal_context.runCosineFused(a.len, &a, &b);
    try std.testing.expectApproxEqAbs(ref_ab, sums.ab, 1e-2);
    try std.testing.expectApproxEqAbs(ref_aa, sums.aa, 1e-2);
    try std.testing.expectApproxEqAbs(ref_bb, sums.bb, 1e-2);
}

test "metal batched cosine fused matches scalar reference" {
    if (!ensureMetalInitialized()) return;
    const query = [_]f32{ 0.5, -1.0, 2.25, 3.0, -0.75 };
    const c0 = [_]f32{ 1.0, 2.0, -0.5, 0.25, 4.0 };
    const c1 = [_]f32{ 0.0, 1.0, 0.0, -1.0, 2.0 };
    const c2 = [_]f32{ 2.25, 3.0, -0.75, 1.5, 0.0 };
    const n = 3;
    const d = query.len;
    const flat = [_]f32{
        c0[0], c0[1], c0[2], c0[3], c0[4],
        c1[0], c1[1], c1[2], c1[3], c1[4],
        c2[0], c2[1], c2[2], c2[3], c2[4],
    };
    var out: [n]f32 = undefined;
    try g_metal_context.runBatchCosineFused(&query, &flat, n, d, &out);

    var expected: [n]f32 = undefined;
    const candidates = [_][]const f32{ &c0, &c1, &c2 };
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
    for (expected, out) |exp, got| {
        try std.testing.expectApproxEqAbs(exp, got, 1e-2);
    }
}

test {
    std.testing.refAllDecls(@This());
    _ = metal_kernels;
}
