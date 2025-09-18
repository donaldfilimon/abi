//! Statistics - Statistical calculation utilities
//!
//! This module provides statistical calculation utilities for:
//! - Exponential moving averages
//! - Running statistics
//! - Online statistical computations

const std = @import("std");

/// Exponential moving average calculator
pub const ExponentialMovingAverage = struct {
    const Self = @This();

    /// Current EMA value
    value: f32,
    /// Smoothing factor (alpha)
    alpha: f32,
    /// Whether this is the first value
    first_value: bool,
    /// Memory allocator
    allocator: std.mem.Allocator,

    /// Initialize a new EMA calculator
    pub fn init(allocator: std.mem.Allocator, alpha: f32) !*Self {
        if (alpha <= 0.0 or alpha > 1.0) return error.InvalidAlpha;

        const ema = try allocator.create(Self);
        ema.* = Self{
            .value = 0.0,
            .alpha = alpha,
            .first_value = true,
            .allocator = allocator,
        };
        return ema;
    }

    /// Deinitialize the EMA calculator
    pub fn deinit(self: *Self) void {
        self.allocator.destroy(self);
    }

    /// Add a new value and update the EMA
    pub fn update(self: *Self, new_value: f32) void {
        if (self.first_value) {
            self.value = new_value;
            self.first_value = false;
        } else {
            self.value = self.alpha * new_value + (1.0 - self.alpha) * self.value;
        }
    }

    /// Get current EMA value
    pub fn get(self: *Self) f32 {
        return self.value;
    }

    /// Reset the EMA calculator
    pub fn reset(self: *Self) void {
        self.value = 0.0;
        self.first_value = true;
    }

    /// Get the smoothing factor
    pub fn getAlpha(self: *Self) f32 {
        return self.alpha;
    }

    /// Set a new smoothing factor
    pub fn setAlpha(self: *Self, alpha: f32) !void {
        if (alpha <= 0.0 or alpha > 1.0) return error.InvalidAlpha;
        self.alpha = alpha;
    }
};

/// Running statistics calculator
pub const RunningStats = struct {
    const Self = @This();

    /// Number of values seen
    count: usize,
    /// Sum of all values
    sum: f32,
    /// Sum of squares
    sum_squares: f32,
    /// Minimum value
    min_val: f32,
    /// Maximum value
    max_val: f32,
    /// Memory allocator
    allocator: std.mem.Allocator,

    /// Initialize a new running statistics calculator
    pub fn init(allocator: std.mem.Allocator) !*Self {
        const stats = try allocator.create(Self);
        stats.* = Self{
            .count = 0,
            .sum = 0.0,
            .sum_squares = 0.0,
            .min_val = std.math.inf(f32),
            .max_val = -std.math.inf(f32),
            .allocator = allocator,
        };
        return stats;
    }

    /// Deinitialize the calculator
    pub fn deinit(self: *Self) void {
        self.allocator.destroy(self);
    }

    /// Add a new value
    pub fn update(self: *Self, value: f32) void {
        self.count += 1;
        self.sum += value;
        self.sum_squares += value * value;
        self.min_val = @min(self.min_val, value);
        self.max_val = @max(self.max_val, value);
    }

    /// Get the mean
    pub fn mean(self: *Self) f32 {
        if (self.count == 0) return 0.0;
        return self.sum / @as(f32, @floatFromInt(self.count));
    }

    /// Get the variance
    pub fn variance(self: *Self) f32 {
        if (self.count < 2) return 0.0;
        const mean_val = self.mean();
        const mean_square = mean_val * mean_val;
        const square_mean = self.sum_squares / @as(f32, @floatFromInt(self.count));
        return square_mean - mean_square;
    }

    /// Get the standard deviation
    pub fn stdDev(self: *Self) f32 {
        return @sqrt(self.variance());
    }

    /// Get the minimum value
    pub fn min(self: *Self) f32 {
        if (self.count == 0) return 0.0;
        return self.min_val;
    }

    /// Get the maximum value
    pub fn max(self: *Self) f32 {
        if (self.count == 0) return 0.0;
        return self.max_val;
    }

    /// Get the range (max - min)
    pub fn range(self: *Self) f32 {
        if (self.count == 0) return 0.0;
        return self.max_val - self.min_val;
    }

    /// Reset all statistics
    pub fn reset(self: *Self) void {
        self.count = 0;
        self.sum = 0.0;
        self.sum_squares = 0.0;
        self.min_val = std.math.inf(f32);
        self.max_val = -std.math.inf(f32);
    }

    /// Get a summary of all statistics
    pub fn summary(self: *Self) struct {
        count: usize,
        mean: f32,
        std_dev: f32,
        min: f32,
        max: f32,
        range: f32,
    } {
        return .{
            .count = self.count,
            .mean = self.mean(),
            .std_dev = self.stdDev(),
            .min = self.min(),
            .max = self.max(),
            .range = self.range(),
        };
    }
};

/// Online variance calculator using Welford's method
pub const OnlineVariance = struct {
    const Self = @This();

    /// Number of values seen
    count: usize,
    /// Current mean
    mean: f32,
    /// Sum of squared differences from current mean
    m2: f32,
    /// Memory allocator
    allocator: std.mem.Allocator,

    /// Initialize a new online variance calculator
    pub fn init(allocator: std.mem.Allocator) !*Self {
        const variance = try allocator.create(Self);
        variance.* = Self{
            .count = 0,
            .mean = 0.0,
            .m2 = 0.0,
            .allocator = allocator,
        };
        return variance;
    }

    /// Deinitialize the calculator
    pub fn deinit(self: *Self) void {
        self.allocator.destroy(self);
    }

    /// Add a new value
    pub fn update(self: *Self, value: f32) void {
        self.count += 1;
        const delta = value - self.mean;
        self.mean += delta / @as(f32, @floatFromInt(self.count));
        const delta2 = value - self.mean;
        self.m2 += delta * delta2;
    }

    /// Get the current mean
    pub fn getMean(self: *Self) f32 {
        if (self.count == 0) return 0.0;
        return self.mean;
    }

    /// Get the current variance
    pub fn getVariance(self: *Self) f32 {
        if (self.count < 2) return 0.0;
        return self.m2 / @as(f32, @floatFromInt(self.count - 1));
    }

    /// Get the current standard deviation
    pub fn getStdDev(self: *Self) f32 {
        return @sqrt(self.getVariance());
    }

    /// Reset the calculator
    pub fn reset(self: *Self) void {
        self.count = 0;
        self.mean = 0.0;
        self.m2 = 0.0;
    }
};
