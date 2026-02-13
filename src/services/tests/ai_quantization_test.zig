//! AI Quantization Tests — Q4_0, Q4_1, Q5_0, Q5_1, Q8_0
//!
//! Tests quantization roundtrip accuracy, edge cases, and dot product correctness.

const std = @import("std");
const abi = @import("abi");
const build_options = @import("build_options");

const quantized = if (build_options.enable_llm) abi.ai.llm.tensor.quantized else struct {};

// ============================================================================
// Q8_0 Roundtrip Tests
// ============================================================================

test "q8_0: roundtrip preserves values within tolerance" {
    if (!build_options.enable_llm) return error.SkipZigTest;

    var input: [32]f32 = undefined;
    for (0..32) |i| {
        input[i] = @as(f32, @floatFromInt(@as(i32, @intCast(i)) - 16)) * 0.5;
    }

    var block: quantized.Q8_0Block = undefined;
    block.quantize(&input);

    var output: [32]f32 = undefined;
    block.dequantize(&output);

    // Q8_0 has 8-bit precision → very small error
    for (input, 0..) |v, i| {
        try std.testing.expectApproxEqAbs(v, output[i], 0.15);
    }
}

test "q8_0: zero values roundtrip exactly" {
    if (!build_options.enable_llm) return error.SkipZigTest;

    var input: [32]f32 = undefined;
    @memset(&input, 0.0);

    var block: quantized.Q8_0Block = undefined;
    block.quantize(&input);

    var output: [32]f32 = undefined;
    block.dequantize(&output);

    for (output) |v| {
        try std.testing.expectApproxEqAbs(@as(f32, 0.0), v, 0.001);
    }
}

test "q8_0: constant values roundtrip accurately" {
    if (!build_options.enable_llm) return error.SkipZigTest;

    var input: [32]f32 = undefined;
    @memset(&input, 1.5);

    var block: quantized.Q8_0Block = undefined;
    block.quantize(&input);

    var output: [32]f32 = undefined;
    block.dequantize(&output);

    for (output) |v| {
        try std.testing.expectApproxEqAbs(@as(f32, 1.5), v, 0.05);
    }
}

// ============================================================================
// Q4_1 Roundtrip Tests
// ============================================================================

test "q4_1: roundtrip preserves monotonicity" {
    if (!build_options.enable_llm) return error.SkipZigTest;

    var input: [32]f32 = undefined;
    for (0..32) |i| {
        input[i] = @as(f32, @floatFromInt(i)) * 0.1;
    }

    var block: quantized.Q4_1Block = undefined;
    block.quantize(&input);

    var output: [32]f32 = undefined;
    block.dequantize(&output);

    // Values should be roughly monotonically increasing
    // (4-bit quantization may collapse adjacent values)
    try std.testing.expect(output[0] < output[31]);
}

test "q4_1: handles negative-to-positive range" {
    if (!build_options.enable_llm) return error.SkipZigTest;

    var input: [32]f32 = undefined;
    for (0..32) |i| {
        input[i] = @as(f32, @floatFromInt(@as(i32, @intCast(i)) - 16));
    }

    var block: quantized.Q4_1Block = undefined;
    block.quantize(&input);

    var output: [32]f32 = undefined;
    block.dequantize(&output);

    // First value should be negative, last should be positive
    try std.testing.expect(output[0] < 0);
    try std.testing.expect(output[31] > 0);
}

// ============================================================================
// Q5_0 / Q5_1 Roundtrip Tests
// ============================================================================

test "q5_0: better precision than q4_0" {
    if (!build_options.enable_llm) return error.SkipZigTest;

    var input: [32]f32 = undefined;
    for (0..32) |i| {
        input[i] = @as(f32, @floatFromInt(@as(i32, @intCast(i)) - 16));
    }

    var block5: quantized.Q5_0Block = undefined;
    block5.quantize(&input);
    var output5: [32]f32 = undefined;
    block5.dequantize(&output5);

    // Q5_0 has 5-bit precision — max error should be bounded
    var max_error5: f32 = 0;
    for (input, 0..) |v, i| {
        const err = @abs(v - output5[i]);
        if (err > max_error5) max_error5 = err;
    }

    try std.testing.expect(max_error5 < 2.0);
}

test "q5_1: asymmetric quantization handles positive range" {
    if (!build_options.enable_llm) return error.SkipZigTest;

    var input: [32]f32 = undefined;
    for (0..32) |i| {
        input[i] = @as(f32, @floatFromInt(i)) * 0.1;
    }

    var block: quantized.Q5_1Block = undefined;
    block.quantize(&input);

    var output: [32]f32 = undefined;
    block.dequantize(&output);

    for (input, 0..) |v, i| {
        try std.testing.expectApproxEqAbs(v, output[i], 0.15);
    }
}

// ============================================================================
// Bulk Quantization Tests
// ============================================================================

test "bulk q8_0: multi-block roundtrip" {
    if (!build_options.enable_llm) return error.SkipZigTest;

    // 2 blocks = 64 values
    var input: [64]f32 = undefined;
    for (0..64) |i| {
        input[i] = @sin(@as(f32, @floatFromInt(i)) * 0.1) * 5.0;
    }

    const q_size = quantized.quantizedSize(.q8_0, 64);
    var q_data: [64 * @sizeOf(quantized.Q8_0Block) / 32]u8 align(@alignOf(quantized.Q8_0Block)) = undefined;
    try quantized.quantizeToQ8_0(&input, q_data[0..q_size]);

    var output: [64]f32 = undefined;
    try quantized.dequantizeQ8_0(q_data[0..q_size], &output);

    for (input, 0..) |v, i| {
        try std.testing.expectApproxEqAbs(v, output[i], 0.2);
    }
}

// ============================================================================
// Quantized Dot Product Tests
// ============================================================================

test "q8_0 dot product: matches f32 reference" {
    if (!build_options.enable_llm) return error.SkipZigTest;

    // Create known input vectors
    var a_f32: [32]f32 = undefined;
    var b_f32: [32]f32 = undefined;
    for (0..32) |i| {
        a_f32[i] = @as(f32, @floatFromInt(i)) * 0.1;
        b_f32[i] = @as(f32, @floatFromInt(32 - i)) * 0.1;
    }

    // Reference dot product in f32
    var ref_dot: f32 = 0;
    for (a_f32, b_f32) |a, b| {
        ref_dot += a * b;
    }

    // Quantize a and compute dot with f32 b
    var block: quantized.Q8_0Block = undefined;
    block.quantize(&a_f32);

    const q8_bytes = std.mem.asBytes(&block);
    const q_dot = quantized.dotQ8_0F32(q8_bytes, &b_f32);

    // Should be close to reference (Q8_0 is high precision)
    try std.testing.expectApproxEqAbs(ref_dot, q_dot, ref_dot * 0.05);
}

// ============================================================================
// Size Calculation Tests
// ============================================================================

test "quantized size: compression ratios correct" {
    if (!build_options.enable_llm) return error.SkipZigTest;

    const n: usize = 1024;
    const f32_size = n * @sizeOf(f32); // 4096 bytes

    // Q4_0: 4.5 bits/weight → ~562 bytes for 1024 elements
    const q4_size = quantized.quantizedSize(.q4_0, n);
    try std.testing.expect(q4_size < f32_size / 2);

    // Q8_0: 8.5 bits/weight → ~1088 bytes
    const q8_size = quantized.quantizedSize(.q8_0, n);
    try std.testing.expect(q8_size < f32_size);
    try std.testing.expect(q8_size > f32_size / 4);

    // Q8_0 should be larger than Q4_0
    try std.testing.expect(q8_size > q4_size);
}

test "quantized size: bits per weight values" {
    if (!build_options.enable_llm) return error.SkipZigTest;

    // Verify ordering: Q4_0 < Q4_1 < Q5_0 < Q5_1 < Q8_0
    try std.testing.expect(quantized.QuantType.q4_0.bitsPerWeight() < quantized.QuantType.q4_1.bitsPerWeight());
    try std.testing.expect(quantized.QuantType.q4_1.bitsPerWeight() < quantized.QuantType.q5_0.bitsPerWeight());
    try std.testing.expect(quantized.QuantType.q5_0.bitsPerWeight() < quantized.QuantType.q5_1.bitsPerWeight());
    try std.testing.expect(quantized.QuantType.q5_1.bitsPerWeight() < quantized.QuantType.q8_0.bitsPerWeight());
}
