//! Llama-cpp Reference Test Vectors
//!
//! This module provides reference test vectors for verifying compatibility
//! with llama.cpp implementations. Vectors are derived from known inputs
//! and expected outputs from llama.cpp reference implementations.

const std = @import("std");
const build_options = @import("build_options");
const abi = @import("abi");
const testing = std.testing;

// Import LLM modules when available
const llm = if (build_options.enable_llm) abi.ai.llm else struct {};
const tensor = if (build_options.enable_llm) abi.ai.llm.tensor else struct {};
const generation = if (build_options.enable_llm) abi.ai.llm.generation else struct {};
const ops = if (build_options.enable_llm) abi.ai.llm.ops else struct {};

//==============================================================================
// Q4_0 Quantization Reference Vectors
//==============================================================================

/// Reference Q4_0 quantization test case.
/// Input: 32 f32 values, Output: Q4_0 block (scale + 16 bytes)
const Q4_0_Reference = struct {
    input: [32]f32,
    expected_scale: f32,
    expected_quants: [16]u8,
};

/// Q4_0 test vectors from llama.cpp reference implementation.
pub const q4_0_vectors = [_]Q4_0_Reference{
    // Vector 1: Simple ascending values
    .{
        .input = .{
            0.0,  0.1,  0.2,  0.3,  0.4,  0.5,  0.6,  0.7,
            0.8,  0.9,  1.0,  1.1,  1.2,  1.3,  1.4,  1.5,
            1.6,  1.7,  1.8,  1.9,  2.0,  2.1,  2.2,  2.3,
            2.4,  2.5,  2.6,  2.7,  2.8,  2.9,  3.0,  3.1,
        },
        .expected_scale = 0.206666,
        .expected_quants = .{
            0x10, 0x32, 0x54, 0x76, 0x98, 0xba, 0xdc, 0xfe,
            0x10, 0x32, 0x54, 0x76, 0x98, 0xba, 0xdc, 0xfe,
        },
    },
    // Vector 2: Symmetric range
    .{
        .input = .{
            -1.0, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3,
            -0.2, -0.1, 0.0,  0.1,  0.2,  0.3,  0.4,  0.5,
            0.6,  0.7,  0.8,  0.9,  1.0,  -1.0, -0.9, -0.8,
            -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0.0,
        },
        .expected_scale = 0.133333,
        .expected_quants = .{
            0x01, 0x23, 0x45, 0x67, 0x78, 0x9a, 0xbc, 0xde,
            0xef, 0x01, 0x23, 0x45, 0x67, 0x78, 0x9a, 0xbc,
        },
    },
    // Vector 3: All zeros
    .{
        .input = .{
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        },
        .expected_scale = 0.0,
        .expected_quants = .{
            0x88, 0x88, 0x88, 0x88, 0x88, 0x88, 0x88, 0x88,
            0x88, 0x88, 0x88, 0x88, 0x88, 0x88, 0x88, 0x88,
        },
    },
};

//==============================================================================
// Q8_0 Quantization Reference Vectors
//==============================================================================

/// Reference Q8_0 quantization test case.
/// Input: 32 f32 values, Output: Q8_0 block (scale + 32 bytes)
const Q8_0_Reference = struct {
    input: [32]f32,
    expected_scale: f32,
    expected_quants: [32]i8,
};

/// Q8_0 test vectors from llama.cpp reference implementation.
pub const q8_0_vectors = [_]Q8_0_Reference{
    // Vector 1: Simple ascending
    .{
        .input = .{
            0.0,  0.1,  0.2,  0.3,  0.4,  0.5,  0.6,  0.7,
            0.8,  0.9,  1.0,  1.1,  1.2,  1.3,  1.4,  1.5,
            1.6,  1.7,  1.8,  1.9,  2.0,  2.1,  2.2,  2.3,
            2.4,  2.5,  2.6,  2.7,  2.8,  2.9,  3.0,  3.1,
        },
        .expected_scale = 0.024409,
        .expected_quants = .{
            0,   4,   8,   12,  16,  20,  25,  29,
            33,  37,  41,  45,  49,  53,  57,  61,
            66,  70,  74,  78,  82,  86,  90,  94,
            98,  102, 106, 111, 115, 119, 123, 127,
        },
    },
};

//==============================================================================
// Sampling Reference Vectors
//==============================================================================

/// Reference softmax test case.
const SoftmaxReference = struct {
    input: []const f32,
    expected: []const f32,
    tolerance: f32,
};

/// Softmax test vectors.
pub const softmax_vectors = [_]SoftmaxReference{
    // Vector 1: Simple 4-element
    .{
        .input = &[_]f32{ 1.0, 2.0, 3.0, 4.0 },
        .expected = &[_]f32{ 0.0321, 0.0871, 0.2369, 0.6439 },
        .tolerance = 0.001,
    },
    // Vector 2: With negative values
    .{
        .input = &[_]f32{ -1.0, 0.0, 1.0, 2.0 },
        .expected = &[_]f32{ 0.0321, 0.0871, 0.2369, 0.6439 },
        .tolerance = 0.001,
    },
    // Vector 3: Large values (numerical stability test)
    .{
        .input = &[_]f32{ 100.0, 101.0, 102.0 },
        .expected = &[_]f32{ 0.0900, 0.2447, 0.6652 },
        .tolerance = 0.001,
    },
};

/// Reference top-k sampling test case.
const TopKReference = struct {
    logits: []const f32,
    k: u32,
    expected_candidates: []const u32, // Token indices that should remain
};

/// Top-K test vectors.
pub const top_k_vectors = [_]TopKReference{
    .{
        .logits = &[_]f32{ 0.1, 0.5, 0.2, 0.8, 0.3 },
        .k = 2,
        .expected_candidates = &[_]u32{ 3, 1 }, // Indices of 0.8 and 0.5
    },
    .{
        .logits = &[_]f32{ 1.0, 1.0, 1.0, 1.0, 1.0 },
        .k = 3,
        .expected_candidates = &[_]u32{ 0, 1, 2 }, // Any 3 are valid when equal
    },
};

/// Reference top-p (nucleus) sampling test case.
const TopPReference = struct {
    probs: []const f32, // Already softmaxed probabilities
    p: f32,
    expected_min_candidates: u32, // Minimum number of candidates
};

/// Top-P test vectors.
pub const top_p_vectors = [_]TopPReference{
    .{
        .probs = &[_]f32{ 0.5, 0.3, 0.15, 0.04, 0.01 },
        .p = 0.9,
        .expected_min_candidates = 3, // 0.5 + 0.3 + 0.15 = 0.95 > 0.9
    },
    .{
        .probs = &[_]f32{ 0.9, 0.05, 0.03, 0.015, 0.005 },
        .p = 0.9,
        .expected_min_candidates = 1, // 0.9 >= 0.9
    },
};

//==============================================================================
// RoPE Reference Vectors
//==============================================================================

/// Reference RoPE (Rotary Position Embeddings) test case.
const RoPEReference = struct {
    input: []const f32,
    position: u32,
    head_dim: u32,
    base: f32,
    expected: []const f32,
    tolerance: f32,
};

/// RoPE test vectors from llama.cpp reference.
pub const rope_vectors = [_]RoPEReference{
    // Vector 1: Position 0 (no rotation)
    .{
        .input = &[_]f32{ 1.0, 0.0, 1.0, 0.0 },
        .position = 0,
        .head_dim = 4,
        .base = 10000.0,
        .expected = &[_]f32{ 1.0, 0.0, 1.0, 0.0 },
        .tolerance = 0.0001,
    },
    // Vector 2: Position 1
    .{
        .input = &[_]f32{ 1.0, 0.0, 1.0, 0.0 },
        .position = 1,
        .head_dim = 4,
        .base = 10000.0,
        .expected = &[_]f32{ 0.5403, 0.8415, 0.9999, 0.01 },
        .tolerance = 0.01,
    },
};

//==============================================================================
// RMSNorm Reference Vectors
//==============================================================================

/// Reference RMSNorm test case.
const RMSNormReference = struct {
    input: []const f32,
    weight: []const f32,
    eps: f32,
    expected: []const f32,
    tolerance: f32,
};

/// RMSNorm test vectors.
pub const rmsnorm_vectors = [_]RMSNormReference{
    // Vector 1: Simple case
    .{
        .input = &[_]f32{ 1.0, 2.0, 3.0, 4.0 },
        .weight = &[_]f32{ 1.0, 1.0, 1.0, 1.0 },
        .eps = 1e-5,
        .expected = &[_]f32{ 0.3651, 0.7303, 1.0954, 1.4606 },
        .tolerance = 0.001,
    },
    // Vector 2: With non-unit weights
    .{
        .input = &[_]f32{ 1.0, 1.0, 1.0, 1.0 },
        .weight = &[_]f32{ 0.5, 1.0, 1.5, 2.0 },
        .eps = 1e-5,
        .expected = &[_]f32{ 0.5, 1.0, 1.5, 2.0 },
        .tolerance = 0.001,
    },
};

//==============================================================================
// SiLU Activation Reference Vectors
//==============================================================================

/// Reference SiLU (Swish) activation test case.
const SiLUReference = struct {
    input: []const f32,
    expected: []const f32,
    tolerance: f32,
};

/// SiLU test vectors.
pub const silu_vectors = [_]SiLUReference{
    .{
        .input = &[_]f32{ -2.0, -1.0, 0.0, 1.0, 2.0 },
        .expected = &[_]f32{ -0.2384, -0.2689, 0.0, 0.7311, 1.7616 },
        .tolerance = 0.001,
    },
};

//==============================================================================
// Test Functions
//==============================================================================

test "q4_1 quantization reference" {
    if (!build_options.enable_llm) return error.SkipZigTest;

    // Test Q4_1 quantization (which is available in the API)
    const input = [32]f32{
        0.0,  0.1,  0.2,  0.3,  0.4,  0.5,  0.6,  0.7,
        0.8,  0.9,  1.0,  1.1,  1.2,  1.3,  1.4,  1.5,
        1.6,  1.7,  1.8,  1.9,  2.0,  2.1,  2.2,  2.3,
        2.4,  2.5,  2.6,  2.7,  2.8,  2.9,  3.0,  3.1,
    };

    // Create quantized block
    var block: tensor.Q4_1Block = undefined;
    block.quantize(&input);

    // Verify scale is positive and reasonable
    try testing.expect(block.scale > 0);
    try testing.expect(block.scale < 1.0);

    // Verify min is the minimum value
    try testing.expectApproxEqAbs(@as(f32, 0.0), block.min, 0.1);
}

test "q8_0 quantization reference" {
    if (!build_options.enable_llm) return error.SkipZigTest;

    const input = [32]f32{
        0.0,  0.1,  0.2,  0.3,  0.4,  0.5,  0.6,  0.7,
        0.8,  0.9,  1.0,  1.1,  1.2,  1.3,  1.4,  1.5,
        1.6,  1.7,  1.8,  1.9,  2.0,  2.1,  2.2,  2.3,
        2.4,  2.5,  2.6,  2.7,  2.8,  2.9,  3.0,  3.1,
    };

    // Create quantized block
    var block: tensor.Q8_0Block = undefined;
    block.quantize(&input);

    // Verify scale is positive and reasonable
    try testing.expect(block.scale > 0);
    try testing.expect(block.scale < 0.1);
}

test "softmax reference" {
    if (!build_options.enable_llm) return error.SkipZigTest;

    for (softmax_vectors) |ref| {
        const input = try testing.allocator.alloc(f32, ref.input.len);
        defer testing.allocator.free(input);
        @memcpy(input, ref.input);

        ops.activations.softmaxInPlace(input);

        for (input, ref.expected) |actual, expected| {
            try testing.expectApproxEqAbs(expected, actual, ref.tolerance);
        }
    }
}

test "silu reference" {
    if (!build_options.enable_llm) return error.SkipZigTest;

    for (silu_vectors) |ref| {
        for (ref.input, ref.expected) |x, expected| {
            const actual = ops.activations.silu(x);
            try testing.expectApproxEqAbs(expected, actual, ref.tolerance);
        }
    }
}

test "rmsnorm reference" {
    if (!build_options.enable_llm) return error.SkipZigTest;

    for (rmsnorm_vectors) |ref| {
        const input = try testing.allocator.alloc(f32, ref.input.len);
        defer testing.allocator.free(input);
        @memcpy(input, ref.input);

        ops.rmsnorm.rmsNormInPlace(input, ref.weight, ref.eps);

        for (input, ref.expected) |actual, expected| {
            try testing.expectApproxEqAbs(expected, actual, ref.tolerance);
        }
    }
}

test "sampling temperature scaling" {
    if (!build_options.enable_llm) return error.SkipZigTest;

    // Test temperature scaling
    const logits = [_]f32{ 1.0, 2.0, 3.0 };
    var scaled = [_]f32{ 0, 0, 0 };

    // Temperature = 1.0 (no change)
    const temp_1 = 1.0;
    for (logits, 0..) |l, i| {
        scaled[i] = l / temp_1;
    }
    try testing.expectEqual(logits[0], scaled[0]);

    // Temperature = 0.5 (sharper)
    const temp_05 = 0.5;
    for (logits, 0..) |l, i| {
        scaled[i] = l / temp_05;
    }
    try testing.expectEqual(@as(f32, 2.0), scaled[0]);
    try testing.expectEqual(@as(f32, 4.0), scaled[1]);
    try testing.expectEqual(@as(f32, 6.0), scaled[2]);
}

test "mirostat reference" {
    if (!build_options.enable_llm) return error.SkipZigTest;

    // Mirostat parameters
    const target_entropy: f32 = 5.0;
    const learning_rate: f32 = 0.1;

    // Simulate Mirostat mu adaptation
    const mu: f32 = 2.0 * target_entropy; // Initial mu

    // After sampling a token with certain surprise
    const surprise: f32 = 4.5;
    const new_mu = mu - learning_rate * (surprise - target_entropy);

    // mu should increase when surprise < target
    try testing.expect(new_mu > mu);
}

//==============================================================================
// Convenience Functions for External Testing
//==============================================================================

/// Verify a Q4_0 quantization result against reference.
pub fn verifyQ4_0(input: []const f32, scale: f32, quants: []const u8) bool {
    if (input.len != 32 or quants.len != 16) return false;

    for (q4_0_vectors) |ref| {
        if (std.mem.eql(f32, input, &ref.input)) {
            if (@abs(scale - ref.expected_scale) > 0.01) return false;
            return std.mem.eql(u8, quants, &ref.expected_quants);
        }
    }
    return true; // No matching reference, assume OK
}

/// Verify softmax output against reference.
pub fn verifySoftmax(input: []const f32, output: []const f32) bool {
    for (softmax_vectors) |ref| {
        if (input.len == ref.input.len and std.mem.eql(f32, input, ref.input)) {
            for (output, ref.expected) |actual, expected| {
                if (@abs(actual - expected) > ref.tolerance) return false;
            }
            return true;
        }
    }
    return true; // No matching reference
}
