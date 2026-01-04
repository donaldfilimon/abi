//! Gradient aggregation utilities for training pipelines.
const std = @import("std");

pub const GradientError = error{
    InvalidGradientSize,
    EmptyAggregation,
};

pub const GradientAccumulator = struct {
    allocator: std.mem.Allocator,
    sum: []f32,
    count: usize,

    /// Initialize a gradient accumulator for a fixed gradient size.
    /// @param allocator Memory allocator for allocations
    /// @param size Length of gradient vectors to aggregate
    /// @return Initialized GradientAccumulator
    pub fn init(allocator: std.mem.Allocator, size: usize) !GradientAccumulator {
        const sum = try allocator.alloc(f32, size);
        @memset(sum, 0);
        return .{
            .allocator = allocator,
            .sum = sum,
            .count = 0,
        };
    }

    /// Release resources owned by the accumulator.
    pub fn deinit(self: *GradientAccumulator) void {
        self.allocator.free(self.sum);
        self.* = undefined;
    }

    /// Reset the accumulator to an empty state.
    pub fn reset(self: *GradientAccumulator) void {
        @memset(self.sum, 0);
        self.count = 0;
    }

    /// Add a gradient into the accumulator.
    /// @param gradient Gradient values to accumulate
    /// @return Error if gradient size mismatches
    pub fn add(self: *GradientAccumulator, gradient: []const f32) GradientError!void {
        if (gradient.len != self.sum.len) return GradientError.InvalidGradientSize;
        for (self.sum, gradient) |*acc, value| {
            acc.* += value;
        }
        self.count += 1;
    }

    /// Compute the average gradient into a freshly allocated buffer.
    /// @param allocator Memory allocator for the returned slice
    /// @return Averaged gradient slice owned by caller
    pub fn average(
        self: *const GradientAccumulator,
        allocator: std.mem.Allocator,
    ) GradientError![]f32 {
        if (self.count == 0) return GradientError.EmptyAggregation;
        const output = try allocator.alloc(f32, self.sum.len);
        const divisor = @as(f32, @floatFromInt(self.count));
        for (self.sum, 0..) |value, i| {
            output[i] = value / divisor;
        }
        return output;
    }

    /// Apply the averaged gradient to the provided weights in-place.
    /// @param weights Model weights to update
    /// @param learning_rate Step size for gradient update
    /// @return Error if gradients are empty or sizes mismatch
    pub fn apply(
        self: *GradientAccumulator,
        weights: []f32,
        learning_rate: f32,
    ) GradientError!void {
        if (self.count == 0) return GradientError.EmptyAggregation;
        if (weights.len != self.sum.len) return GradientError.InvalidGradientSize;
        const scale = learning_rate / @as(f32, @floatFromInt(self.count));
        for (weights, self.sum) |*weight, grad_sum| {
            weight.* -= grad_sum * scale;
        }
    }
};

test "gradient accumulator averages values" {
    var accumulator = try GradientAccumulator.init(std.testing.allocator, 3);
    defer accumulator.deinit();

    try accumulator.add(&.{ 1.0, 2.0, 3.0 });
    try accumulator.add(&.{ 3.0, 4.0, 5.0 });

    const avg = try accumulator.average(std.testing.allocator);
    defer std.testing.allocator.free(avg);

    try std.testing.expectEqualSlices(f32, &.{ 2.0, 3.0, 4.0 }, avg);
}

test "gradient accumulator apply updates weights" {
    var accumulator = try GradientAccumulator.init(std.testing.allocator, 2);
    defer accumulator.deinit();

    try accumulator.add(&.{ 1.0, 1.0 });
    try accumulator.add(&.{ 3.0, 3.0 });

    var weights = [_]f32{ 10.0, 20.0 };
    try accumulator.apply(&weights, 0.1);

    try std.testing.expectApproxEqAbs(@as(f32, 9.8), weights[0], 0.0001);
    try std.testing.expectApproxEqAbs(@as(f32, 19.8), weights[1], 0.0001);
}

test "gradient accumulator rejects mismatched sizes" {
    var accumulator = try GradientAccumulator.init(std.testing.allocator, 2);
    defer accumulator.deinit();

    try std.testing.expectError(
        GradientError.InvalidGradientSize,
        accumulator.add(&.{ 1.0, 2.0, 3.0 }),
    );
}
