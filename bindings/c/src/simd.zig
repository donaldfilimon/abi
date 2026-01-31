//! C-compatible SIMD operation exports.
//! Wraps the ABI SIMD functions for C FFI.

const std = @import("std");
const builtin = @import("builtin");

// Import SIMD from the library's path relative to bindings
const simd = @import("simd_impl.zig");

/// SIMD capabilities struct matching C header.
pub const SimdCaps = extern struct {
    sse: bool = false,
    sse2: bool = false,
    sse3: bool = false,
    ssse3: bool = false,
    sse4_1: bool = false,
    sse4_2: bool = false,
    avx: bool = false,
    avx2: bool = false,
    avx512f: bool = false,
    neon: bool = false,
};

/// Query SIMD capabilities based on target architecture.
pub export fn abi_simd_get_caps(out_caps: *SimdCaps) void {
    out_caps.* = .{};

    // Detect capabilities at comptime based on target
    const target = builtin.cpu;

    if (builtin.cpu.arch == .x86_64 or builtin.cpu.arch == .x86) {
        // x86/x86_64 SIMD detection
        if (std.Target.x86.featureSetHas(target.features, .sse)) out_caps.sse = true;
        if (std.Target.x86.featureSetHas(target.features, .sse2)) out_caps.sse2 = true;
        if (std.Target.x86.featureSetHas(target.features, .sse3)) out_caps.sse3 = true;
        if (std.Target.x86.featureSetHas(target.features, .ssse3)) out_caps.ssse3 = true;
        if (std.Target.x86.featureSetHas(target.features, .sse4_1)) out_caps.sse4_1 = true;
        if (std.Target.x86.featureSetHas(target.features, .sse4_2)) out_caps.sse4_2 = true;
        if (std.Target.x86.featureSetHas(target.features, .avx)) out_caps.avx = true;
        if (std.Target.x86.featureSetHas(target.features, .avx2)) out_caps.avx2 = true;
        if (std.Target.x86.featureSetHas(target.features, .avx512f)) out_caps.avx512f = true;
    } else if (builtin.cpu.arch == .aarch64 or builtin.cpu.arch == .arm) {
        // ARM NEON is baseline on aarch64
        if (builtin.cpu.arch == .aarch64) {
            out_caps.neon = true;
        } else if (std.Target.arm.featureSetHas(target.features, .neon)) {
            out_caps.neon = true;
        }
    }
}

/// Check if any SIMD is available.
pub export fn abi_simd_available() bool {
    var caps: SimdCaps = .{};
    abi_simd_get_caps(&caps);
    return caps.sse or caps.sse2 or caps.neon;
}

/// Vector addition: result[i] = a[i] + b[i]
pub export fn abi_simd_vector_add(
    a: [*]const f32,
    b: [*]const f32,
    result: [*]f32,
    len: usize,
) void {
    if (len == 0) return;
    simd.vectorAdd(a[0..len], b[0..len], result[0..len]);
}

/// Vector dot product.
pub export fn abi_simd_vector_dot(a: [*]const f32, b: [*]const f32, len: usize) f32 {
    if (len == 0) return 0.0;
    return simd.vectorDot(a[0..len], b[0..len]);
}

/// Vector L2 norm.
pub export fn abi_simd_vector_l2_norm(v: [*]const f32, len: usize) f32 {
    if (len == 0) return 0.0;
    return simd.vectorL2Norm(v[0..len]);
}

/// Cosine similarity.
pub export fn abi_simd_cosine_similarity(a: [*]const f32, b: [*]const f32, len: usize) f32 {
    if (len == 0) return 0.0;
    return simd.cosineSimilarity(a[0..len], b[0..len]);
}

test "simd caps export" {
    var caps: SimdCaps = .{};
    abi_simd_get_caps(&caps);
    // Should detect something on any modern CPU
    _ = abi_simd_available();
}

test "simd vector ops" {
    const a = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const b = [_]f32{ 4.0, 3.0, 2.0, 1.0 };
    var result: [4]f32 = undefined;

    abi_simd_vector_add(&a, &b, &result, 4);
    try std.testing.expectApproxEqAbs(@as(f32, 5.0), result[0], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 5.0), result[1], 0.001);

    const dot = abi_simd_vector_dot(&a, &b, 4);
    try std.testing.expectApproxEqAbs(@as(f32, 20.0), dot, 0.001);

    const norm = abi_simd_vector_l2_norm(&a, 4);
    try std.testing.expectApproxEqAbs(@as(f32, 5.477), norm, 0.01);
}
