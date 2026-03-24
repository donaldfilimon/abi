//! Fairness Principle Validator (Principle 4)
//!
//! Avoid bias; present balanced perspectives.
//! Provides bias pattern detection and quantification (Spec Section 5.4).

const std = @import("std");

/// Maximum number of per-attribute bias flags returned by computeBias.
pub const MAX_BIAS_ATTRIBUTES = 32;

/// Default threshold: attributes with |Bi| above this are flagged.
pub const DEFAULT_BIAS_THRESHOLD: f32 = 0.1;

/// Result of bias quantification across protected attributes.
/// Implements: B = (1/n) * Sigma |Bi|
pub const BiasScore = struct {
    /// Mean absolute bias across all attributes.
    mean_abs_bias: f32,
    /// Per-attribute flags: true if |Bi| exceeds the threshold.
    attribute_flags: [MAX_BIAS_ATTRIBUTES]bool,
    /// Number of attributes that were evaluated.
    attribute_count: usize,
    /// Number of attributes that exceeded the threshold.
    flagged_count: usize,
    /// Whether overall bias is within acceptable limits.
    is_acceptable: bool,

    /// Return the fraction of attributes that are flagged.
    pub fn flaggedRatio(self: *const BiasScore) f32 {
        if (self.attribute_count == 0) return 0.0;
        return @as(f32, @floatFromInt(self.flagged_count)) / @as(f32, @floatFromInt(self.attribute_count));
    }
};

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

test {
    std.testing.refAllDecls(@This());
}
