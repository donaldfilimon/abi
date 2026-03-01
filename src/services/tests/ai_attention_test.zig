//! AI Attention Mechanism Tests — Scale Factors, Causal Mask, Flash Attention
//!
//! Tests mathematical correctness of the attention implementation.

const std = @import("std");
const abi = @import("abi");
const build_options = @import("build_options");

const attention = if (build_options.enable_llm) abi.features.ai.llm.ops.attention else struct {};
const activations = if (build_options.enable_llm) abi.features.ai.llm.ops.activations else struct {};

// ============================================================================
// Comptime Scale Factor Tests
// ============================================================================

test "attention: comptime scale factors are 1/sqrt(d)" {
    if (!build_options.enable_llm) return error.SkipZigTest;

    // Verify pre-computed constants match formula
    try std.testing.expectApproxEqAbs(
        1.0 / @sqrt(@as(f32, 32.0)),
        attention.ComptimeScales.SCALE_32,
        1e-7,
    );
    try std.testing.expectApproxEqAbs(
        1.0 / @sqrt(@as(f32, 64.0)),
        attention.ComptimeScales.SCALE_64,
        1e-7,
    );
    try std.testing.expectApproxEqAbs(
        1.0 / @sqrt(@as(f32, 128.0)),
        attention.ComptimeScales.SCALE_128,
        1e-7,
    );
}

test "attention: getScale falls back to runtime for non-standard dims" {
    if (!build_options.enable_llm) return error.SkipZigTest;

    const scale_48 = attention.ComptimeScales.getScale(48);
    try std.testing.expectApproxEqAbs(
        1.0 / @sqrt(@as(f32, 48.0)),
        scale_48,
        1e-7,
    );

    // Known dimension should use pre-computed
    const scale_64 = attention.ComptimeScales.getScale(64);
    try std.testing.expectEqual(attention.ComptimeScales.SCALE_64, scale_64);
}

test "attention: ScaledAttention comptime instantiation" {
    if (!build_options.enable_llm) return error.SkipZigTest;

    // Verify comptime instantiation produces correct constants
    const A64 = attention.ScaledAttention(64);
    try std.testing.expectEqual(@as(u32, 64), A64.HEAD_DIM);
    try std.testing.expectApproxEqAbs(
        1.0 / @sqrt(@as(f32, 64.0)),
        A64.SCALE,
        1e-7,
    );
}

// ============================================================================
// Causal Mask Tests
// ============================================================================

test "attention: causal mask blocks future positions" {
    if (!build_options.enable_llm) return error.SkipZigTest;
    const allocator = std.testing.allocator;

    // 3-token sequence, head_dim=2
    // Q = [1,0; 0,1; 1,1], K = V = [1,0; 0,1; 1,1]
    const q = [_]f32{ 1, 0, 0, 1, 1, 1 };
    const k = [_]f32{ 1, 0, 0, 1, 1, 1 };
    const v = [_]f32{ 1, 0, 0, 1, 0.5, 0.5 };
    var output: [6]f32 = undefined;

    try attention.scaledDotProductAttention(
        allocator,
        &q,
        &k,
        &v,
        &output,
        3, // seq_len
        3, // kv_len
        2, // head_dim
        true, // causal
    );

    // Row 0 (token 0): can only see position 0
    // Output should be V[0] = [1.0, 0.0]
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), output[0], 0.01);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), output[1], 0.01);

    // Row 2 (token 2): can see all positions
    // Output should be weighted sum of all V rows
    try std.testing.expect(output[4] > 0);
}

test "attention: non-causal sees all positions" {
    if (!build_options.enable_llm) return error.SkipZigTest;
    const allocator = std.testing.allocator;

    // Two positions with orthogonal Q,K so attention is spread
    const q = [_]f32{ 1, 0, 0, 1 };
    const k = [_]f32{ 1, 0, 0, 1 };
    const v = [_]f32{ 10, 0, 0, 10 };
    var output_causal: [4]f32 = undefined;
    var output_non_causal: [4]f32 = undefined;

    try attention.scaledDotProductAttention(
        allocator,
        &q,
        &k,
        &v,
        &output_causal,
        2,
        2,
        2,
        true,
    );
    try attention.scaledDotProductAttention(
        allocator,
        &q,
        &k,
        &v,
        &output_non_causal,
        2,
        2,
        2,
        false,
    );

    // Row 0 should differ: causal only sees V[0], non-causal sees V[0]+V[1]
    // Causal row 0: output ≈ V[0] = [10, 0]
    // Non-causal row 0: output = weighted sum of V[0] and V[1]
    try std.testing.expect(output_non_causal[1] > output_causal[1]);
}

// ============================================================================
// Flash Attention Tests
// ============================================================================

test "attention: flash matches standard for small input" {
    if (!build_options.enable_llm) return error.SkipZigTest;
    const allocator = std.testing.allocator;

    const q = [_]f32{ 1, 0, 0, 1 };
    const k = [_]f32{ 1, 0, 0, 1 };
    const v = [_]f32{ 2, 3, 4, 5 };
    var output_std: [4]f32 = undefined;
    var output_flash: [4]f32 = undefined;

    try attention.scaledDotProductAttention(
        allocator,
        &q,
        &k,
        &v,
        &output_std,
        2,
        2,
        2,
        false,
    );
    try attention.flashAttention(
        allocator,
        &q,
        &k,
        &v,
        &output_flash,
        2,
        2,
        2,
        .{ .causal = false, .block_size_q = 1, .block_size_kv = 1 },
    );

    for (0..4) |i| {
        try std.testing.expectApproxEqAbs(output_std[i], output_flash[i], 1e-4);
    }
}

test "attention: flash attention with varied block sizes" {
    if (!build_options.enable_llm) return error.SkipZigTest;
    const allocator = std.testing.allocator;

    const seq_len: u32 = 4;
    const head_dim: u32 = 4;
    const size = @as(usize, seq_len) * head_dim;

    var q = try allocator.alloc(f32, size);
    defer allocator.free(q);
    var k = try allocator.alloc(f32, size);
    defer allocator.free(k);
    var v = try allocator.alloc(f32, size);
    defer allocator.free(v);
    const output_ref = try allocator.alloc(f32, size);
    defer allocator.free(output_ref);

    for (0..size) |i| {
        q[i] = @as(f32, @floatFromInt(i % 5)) * 0.2;
        k[i] = @as(f32, @floatFromInt((i + 2) % 5)) * 0.2;
        v[i] = @as(f32, @floatFromInt((i + 4) % 5)) * 0.2;
    }

    // Reference: standard attention
    try attention.scaledDotProductAttention(
        allocator,
        q,
        k,
        v,
        output_ref,
        seq_len,
        seq_len,
        head_dim,
        false,
    );

    // Test block sizes: 1, 2, 4 (== seq_len)
    const block_sizes = [_]u32{ 1, 2, 4 };
    for (block_sizes) |bs| {
        const output_flash = try allocator.alloc(f32, size);
        defer allocator.free(output_flash);

        try attention.flashAttention(
            allocator,
            q,
            k,
            v,
            output_flash,
            seq_len,
            seq_len,
            head_dim,
            .{ .causal = false, .block_size_q = bs, .block_size_kv = bs },
        );

        for (0..size) |i| {
            try std.testing.expectApproxEqAbs(output_ref[i], output_flash[i], 1e-3);
        }
    }
}

// ============================================================================
// Softmax Tests
// ============================================================================

test "attention: softmax output sums to 1" {
    if (!build_options.enable_llm) return error.SkipZigTest;

    var logits = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    activations.softmaxInPlace(&logits);

    var sum: f32 = 0;
    for (logits) |p| {
        sum += p;
        try std.testing.expect(p >= 0);
        try std.testing.expect(p <= 1.0);
    }
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), sum, 1e-5);
}

test "attention: softmax preserves ordering" {
    if (!build_options.enable_llm) return error.SkipZigTest;

    var logits = [_]f32{ 1.0, 3.0, 2.0, 5.0 };
    activations.softmaxInPlace(&logits);

    // Highest logit (5.0 at index 3) should have highest probability
    try std.testing.expect(logits[3] > logits[1]);
    try std.testing.expect(logits[1] > logits[2]);
    try std.testing.expect(logits[2] > logits[0]);
}

test "attention: softmax numerically stable with large values" {
    if (!build_options.enable_llm) return error.SkipZigTest;

    // Large values that would overflow naive exp()
    var logits = [_]f32{ 1000.0, 1001.0, 999.0, 1000.5 };
    activations.softmaxInPlace(&logits);

    // Should not produce NaN or Inf
    var sum: f32 = 0;
    for (logits) |p| {
        try std.testing.expect(!std.math.isNan(p));
        try std.testing.expect(!std.math.isInf(p));
        sum += p;
    }
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), sum, 1e-5);
}

// ============================================================================
// AttentionConfig Tests
// ============================================================================

test "attention: config from model dimensions" {
    if (!build_options.enable_llm) return error.SkipZigTest;

    // Llama-7B config: hidden=4096, heads=32, kv_heads=32
    const config = attention.AttentionConfig.fromModel(4096, 32, 32);
    try std.testing.expectEqual(@as(u32, 128), config.head_dim);
    try std.testing.expect(config.causal);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), config.dropout, 1e-7);

    // GQA config: Llama-2 70B: hidden=8192, heads=64, kv_heads=8
    const gqa_config = attention.AttentionConfig.fromModel(8192, 64, 8);
    try std.testing.expectEqual(@as(u32, 128), gqa_config.head_dim);
    try std.testing.expectEqual(@as(u32, 8), gqa_config.num_kv_heads);
}
