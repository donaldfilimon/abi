//! Utilities for model training.

const std = @import("std");

/// Xavier uniform initialization.
pub fn initializeXavier(weights: []f32) void {
    const fan_in = @sqrt(@as(f32, @floatFromInt(weights.len)));
    const limit = @sqrt(6.0) / fan_in;

    var rng = std.Random.DefaultPrng.init(@as(u64, @intCast(weights.len)));
    for (weights) |*w| {
        w.* = (rng.random().float(f32) * 2.0 - 1.0) * limit;
    }
}
