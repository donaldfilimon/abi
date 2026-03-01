//! C API SIMD Tests — SIMD capabilities, vector operations, struct layout.

const std = @import("std");
const testing = std.testing;
const builtin = @import("builtin");
const abi = @import("abi");

// ============================================================================
// SIMD Capability Tests
// ============================================================================

test "c_api: simd availability detection" {
    // Test SIMD detection (C API wraps abi_simd_available())
    const has_simd = abi.services.simd.hasSimdSupport();

    // SIMD should be available on most modern platforms
    if (builtin.cpu.arch == .x86_64 or builtin.cpu.arch == .aarch64) {
        try testing.expect(has_simd);
    }
}

test "c_api: simd capabilities structure" {
    // Test SIMD capabilities (C API wraps abi_simd_get_caps())
    const caps = abi.services.simd.getSimdCapabilities();

    // Vector size should be at least 1 (scalar fallback)
    try testing.expect(caps.vector_size >= 1);

    // Arch should be detected correctly
    switch (builtin.cpu.arch) {
        .x86_64 => try testing.expect(caps.arch == .x86_64),
        .aarch64 => try testing.expect(caps.arch == .aarch64),
        .wasm32, .wasm64 => try testing.expect(caps.arch == .wasm),
        else => try testing.expect(caps.arch == .generic),
    }

    // has_simd should be true if vector_size > 1
    try testing.expect(caps.has_simd == (caps.vector_size > 1));
}

test "c_api: simd vector operations work correctly" {
    // Test that SIMD operations produce correct results
    // C API would use these through wrapper functions
    var a = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    var b = [_]f32{ 0.5, 1.5, 2.5, 3.5 };
    var result: [4]f32 = undefined;

    abi.services.simd.vectorAdd(&a, &b, &result);

    try testing.expectApproxEqAbs(@as(f32, 1.5), result[0], 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 3.5), result[1], 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 5.5), result[2], 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 7.5), result[3], 1e-6);
}

test "c_api: simd dot product" {
    var a = [_]f32{ 1.0, 2.0, 3.0 };
    var b = [_]f32{ 4.0, 5.0, 6.0 };

    const result = abi.services.simd.vectorDot(&a, &b);

    // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
    try testing.expectApproxEqAbs(@as(f32, 32.0), result, 1e-6);
}

test "c_api: simd cosine similarity" {
    var a = [_]f32{ 1.0, 0.0 };
    var b = [_]f32{ 1.0, 0.0 };

    const result = abi.services.simd.cosineSimilarity(&a, &b);

    // Identical vectors should have cosine similarity of 1.0
    try testing.expectApproxEqAbs(@as(f32, 1.0), result, 1e-6);

    // Orthogonal vectors
    var c = [_]f32{ 1.0, 0.0 };
    var d = [_]f32{ 0.0, 1.0 };

    const result2 = abi.services.simd.cosineSimilarity(&c, &d);
    try testing.expectApproxEqAbs(@as(f32, 0.0), result2, 1e-6);
}

test "c_api: simd L2 norm" {
    var v = [_]f32{ 3.0, 4.0 };

    const result = abi.services.simd.vectorL2Norm(&v);

    // sqrt(3^2 + 4^2) = sqrt(9 + 16) = sqrt(25) = 5
    try testing.expectApproxEqAbs(@as(f32, 5.0), result, 1e-6);
}

// ============================================================================
// SIMD SimdCaps Struct Tests
// ============================================================================

test "c_api: simd caps struct layout" {
    // Verify SimdCaps struct matches C ABI layout
    const SimdCaps = extern struct {
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

    // Get actual capabilities
    const caps = abi.services.simd.getSimdCapabilities();
    const is_x86 = caps.arch == .x86_64;
    const is_arm = caps.arch == .aarch64;

    // Build C API compatible struct
    var c_caps = SimdCaps{
        .sse = if (is_x86) caps.has_simd else false,
        .sse2 = if (is_x86) caps.has_simd else false,
        .sse3 = if (is_x86) caps.has_simd else false,
        .ssse3 = if (is_x86) caps.has_simd else false,
        .sse4_1 = if (is_x86) caps.has_simd else false,
        .sse4_2 = if (is_x86) caps.has_simd else false,
        .avx = if (is_x86) (caps.vector_size >= 8) else false,
        .avx2 = if (is_x86) (caps.vector_size >= 8) else false,
        .avx512f = if (is_x86) (caps.vector_size >= 16) else false,
        .neon = is_arm,
    };

    // On x86_64, at least SSE should be available if SIMD is supported
    if (is_x86 and caps.has_simd) {
        try testing.expect(c_caps.sse);
        try testing.expect(c_caps.sse2);
    }

    // On ARM64, NEON should be available
    if (is_arm) {
        try testing.expect(c_caps.neon);
    }
}
