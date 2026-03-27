//! Fairness Principle Validator (Principle 4)
//!
//! Avoid bias; present balanced perspectives.
//! Provides bias pattern detection and quantification (Spec Section 5.4).

const std = @import("std");
const types = @import("../types.zig");

/// Maximum number of per-attribute bias flags returned by computeBias.
pub const MAX_BIAS_ATTRIBUTES = types.MAX_BIAS_ATTRIBUTES;

/// Default threshold: attributes with |Bi| above this are flagged.
pub const DEFAULT_BIAS_THRESHOLD: f32 = types.DEFAULT_BIAS_THRESHOLD;

/// Result of bias quantification across protected attributes.
pub const BiasScore = types.BiasScore;

/// Compute bias quantification from per-attribute bias measurements.
///
/// Implements the spec formula: B = (1/n) * Sigma_i |Bi|
///
/// Parameters:
///   - measurements: slice of per-attribute bias values (positive or negative)
///   - threshold: absolute bias level above which an attribute is flagged
///                (use DEFAULT_BIAS_THRESHOLD for the standard 0.1 cutoff)
///
/// Returns a BiasScore with the mean absolute bias and per-attribute flags.
/// If measurements is empty, returns a zero-bias score with is_acceptable = true.
pub fn computeBias(measurements: []const f32, threshold: f32) BiasScore {
    var result = BiasScore{
        .mean_abs_bias = 0.0,
        .attribute_flags = [_]bool{false} ** MAX_BIAS_ATTRIBUTES,
        .attribute_count = measurements.len,
        .flagged_count = 0,
        .is_acceptable = true,
    };

    if (measurements.len == 0) return result;

    var sum_abs: f32 = 0.0;
    const n = @min(measurements.len, MAX_BIAS_ATTRIBUTES);
    for (measurements[0..n], 0..) |b_i, i| {
        const abs_b = @abs(b_i);
        sum_abs += abs_b;
        if (abs_b > threshold) {
            result.attribute_flags[i] = true;
            result.flagged_count += 1;
        }
    }
    // Handle measurements beyond MAX_BIAS_ATTRIBUTES (still count toward mean)
    for (measurements[n..]) |b_i| {
        sum_abs += @abs(b_i);
    }

    result.mean_abs_bias = sum_abs / @as(f32, @floatFromInt(measurements.len));
    result.is_acceptable = result.mean_abs_bias <= threshold;

    return result;
}

pub fn containsBiasPatterns(text: []const u8) bool {
    const patterns = [_][]const u8{
        "all members of",
        "those people are",
        "that race is",
        "inherently inferior",
        "inherently superior",
    };
    for (&patterns) |pattern| {
        if (std.mem.indexOf(u8, text, pattern) != null) return true;
    }
    return false;
}

// ============================================================================
// Bias Quantification Tests
// ============================================================================

test "computeBias empty measurements" {
    const result = computeBias(&[_]f32{}, DEFAULT_BIAS_THRESHOLD);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), result.mean_abs_bias, 0.001);
    try std.testing.expectEqual(@as(usize, 0), result.attribute_count);
    try std.testing.expectEqual(@as(usize, 0), result.flagged_count);
    try std.testing.expect(result.is_acceptable);
}

test "computeBias single zero measurement" {
    const measurements = [_]f32{0.0};
    const result = computeBias(&measurements, DEFAULT_BIAS_THRESHOLD);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), result.mean_abs_bias, 0.001);
    try std.testing.expectEqual(@as(usize, 1), result.attribute_count);
    try std.testing.expectEqual(@as(usize, 0), result.flagged_count);
    try std.testing.expect(result.is_acceptable);
}

test "computeBias spec formula B = (1/n) * sum|Bi|" {
    // Manual: |0.05| + |-0.2| + |0.08| + |0.15| = 0.48; B = 0.48/4 = 0.12
    const measurements = [_]f32{ 0.05, -0.2, 0.08, 0.15 };
    const result = computeBias(&measurements, DEFAULT_BIAS_THRESHOLD);
    try std.testing.expectApproxEqAbs(@as(f32, 0.12), result.mean_abs_bias, 0.001);
    try std.testing.expectEqual(@as(usize, 4), result.attribute_count);
}

test "computeBias flags attributes above threshold" {
    const measurements = [_]f32{ 0.05, -0.2, 0.08, 0.15 };
    const result = computeBias(&measurements, DEFAULT_BIAS_THRESHOLD);
    // |0.05| = 0.05 <= 0.1 -> not flagged
    try std.testing.expect(!result.attribute_flags[0]);
    // |-0.2| = 0.2 > 0.1 -> flagged
    try std.testing.expect(result.attribute_flags[1]);
    // |0.08| = 0.08 <= 0.1 -> not flagged
    try std.testing.expect(!result.attribute_flags[2]);
    // |0.15| = 0.15 > 0.1 -> flagged
    try std.testing.expect(result.attribute_flags[3]);
    try std.testing.expectEqual(@as(usize, 2), result.flagged_count);
}

test "computeBias is_acceptable when mean below threshold" {
    // All low bias: |0.02| + |0.03| + |0.01| = 0.06; B = 0.02
    const measurements = [_]f32{ 0.02, -0.03, 0.01 };
    const result = computeBias(&measurements, DEFAULT_BIAS_THRESHOLD);
    try std.testing.expect(result.is_acceptable);
    try std.testing.expectEqual(@as(usize, 0), result.flagged_count);
}

test "computeBias not acceptable when mean exceeds threshold" {
    // High bias: |0.5| + |0.6| + |0.7| = 1.8; B = 0.6
    const measurements = [_]f32{ 0.5, -0.6, 0.7 };
    const result = computeBias(&measurements, DEFAULT_BIAS_THRESHOLD);
    try std.testing.expect(!result.is_acceptable);
    try std.testing.expectEqual(@as(usize, 3), result.flagged_count);
}

test "computeBias negative values handled by absolute value" {
    // Symmetric: should produce same mean_abs_bias
    const positive = [_]f32{ 0.3, 0.4 };
    const negative = [_]f32{ -0.3, -0.4 };
    const result_pos = computeBias(&positive, DEFAULT_BIAS_THRESHOLD);
    const result_neg = computeBias(&negative, DEFAULT_BIAS_THRESHOLD);
    try std.testing.expectApproxEqAbs(result_pos.mean_abs_bias, result_neg.mean_abs_bias, 0.001);
    try std.testing.expectEqual(result_pos.flagged_count, result_neg.flagged_count);
}

test "computeBias custom threshold" {
    const measurements = [_]f32{ 0.3, 0.4, 0.5 };
    // With threshold 0.35: only 0.4 and 0.5 flagged
    const result = computeBias(&measurements, 0.35);
    try std.testing.expect(!result.attribute_flags[0]); // 0.3 <= 0.35
    try std.testing.expect(result.attribute_flags[1]); // 0.4 > 0.35
    try std.testing.expect(result.attribute_flags[2]); // 0.5 > 0.35
    try std.testing.expectEqual(@as(usize, 2), result.flagged_count);
}

test "computeBias flaggedRatio" {
    const measurements = [_]f32{ 0.05, -0.2, 0.08, 0.15 };
    const result = computeBias(&measurements, DEFAULT_BIAS_THRESHOLD);
    // 2 out of 4 flagged -> 0.5
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), result.flaggedRatio(), 0.001);
}

test "computeBias flaggedRatio empty" {
    const result = computeBias(&[_]f32{}, DEFAULT_BIAS_THRESHOLD);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), result.flaggedRatio(), 0.001);
}

test {
    std.testing.refAllDecls(@This());
}
