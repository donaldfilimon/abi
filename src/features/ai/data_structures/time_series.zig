//! Time Series - Time-based data structures
//!
//! This module provides time series data structures for:
//! - Time-stamped data storage
//! - Efficient time-based queries
//! - Automatic data aging and cleanup

const std = @import("std");

/// Time series data point
pub const TimeSeriesPoint = struct {
    timestamp: i64,
    value: f32,
};

/// Time series buffer for storing time-stamped data
pub const TimeSeriesBuffer = struct {
    const Self = @This();

    /// Data points
    points: std.ArrayList(TimeSeriesPoint),
    /// Maximum capacity
    capacity: usize,
    /// Memory allocator
    allocator: std.mem.Allocator,

    /// Initialize a new time series buffer
    pub fn init(allocator: std.mem.Allocator, capacity: usize) !*Self {
        const buffer = try allocator.create(Self);
        buffer.* = Self{
            .points = try std.ArrayList(TimeSeriesPoint).initCapacity(allocator, capacity),
            .capacity = capacity,
            .allocator = allocator,
        };
        return buffer;
    }

    /// Deinitialize the buffer
    pub fn deinit(self: *Self) void {
        self.points.deinit();
        self.allocator.destroy(self);
    }

    /// Add a data point
    pub fn addPoint(self: *Self, timestamp: i64, value: f32) !void {
        const point = TimeSeriesPoint{
            .timestamp = timestamp,
            .value = value,
        };

        try self.points.append(point);

        // Remove oldest points if over capacity
        while (self.points.items.len > self.capacity) {
            _ = self.points.orderedRemove(0);
        }
    }

    /// Get value at specific timestamp (exact match)
    pub fn getValueAt(self: *Self, timestamp: i64) ?f32 {
        for (self.points.items) |point| {
            if (point.timestamp == timestamp) {
                return point.value;
            }
        }
        return null;
    }

    /// Get interpolated value at timestamp
    pub fn getInterpolatedValueAt(self: *Self, timestamp: i64) ?f32 {
        if (self.points.items.len == 0) return null;
        if (self.points.items.len == 1) return self.points.items[0].value;

        // Find surrounding points
        var prev_point: ?TimeSeriesPoint = null;
        var next_point: ?TimeSeriesPoint = null;

        for (self.points.items) |point| {
            if (point.timestamp <= timestamp) {
                prev_point = point;
            } else {
                next_point = point;
                break;
            }
        }

        if (prev_point == null) return next_point.?.value;
        if (next_point == null) return prev_point.?.value;

        // Linear interpolation
        const prev = prev_point.?;
        const next = next_point.?;
        const time_diff = @as(f32, @floatFromInt(next.timestamp - prev.timestamp));
        const value_diff = next.value - prev.value;
        const time_ratio = @as(f32, @floatFromInt(timestamp - prev.timestamp)) / time_diff;

        return prev.value + value_diff * time_ratio;
    }

    /// Get values in time range
    pub fn getValuesInRange(self: *Self, start_time: i64, end_time: i64) !std.ArrayList(TimeSeriesPoint) {
        var result = std.ArrayList(TimeSeriesPoint){};
        for (self.points.items) |point| {
            if (point.timestamp >= start_time and point.timestamp <= end_time) {
                try result.append(self.allocator, point);
            }
        }
        return result;
    }

    /// Calculate simple moving average
    pub fn simpleMovingAverage(self: *Self, window_size: usize) !std.ArrayList(f32) {
        if (window_size == 0 or self.points.items.len < window_size) {
            return std.ArrayList(f32){};
        }

        var result = try std.ArrayList(f32).initCapacity(self.allocator, self.points.items.len - window_size + 1);

        for (0..self.points.items.len - window_size + 1) |i| {
            var sum: f32 = 0.0;
            for (0..window_size) |j| {
                sum += self.points.items[i + j].value;
            }
            try result.append(sum / @as(f32, @floatFromInt(window_size)));
        }

        return result;
    }

    /// Get statistics for the time series
    pub fn getStats(self: *Self) struct {
        count: usize,
        min_value: f32,
        max_value: f32,
        avg_value: f32,
        time_span: i64,
    } {
        if (self.points.items.len == 0) {
            return .{
                .count = 0,
                .min_value = 0.0,
                .max_value = 0.0,
                .avg_value = 0.0,
                .time_span = 0,
            };
        }

        var min_val = self.points.items[0].value;
        var max_val = self.points.items[0].value;
        var sum: f32 = 0.0;

        for (self.points.items) |point| {
            min_val = @min(min_val, point.value);
            max_val = @max(max_val, point.value);
            sum += point.value;
        }

        const earliest = self.points.items[0].timestamp;
        const latest = self.points.items[self.points.items.len - 1].timestamp;

        return .{
            .count = self.points.items.len,
            .min_value = min_val,
            .max_value = max_val,
            .avg_value = sum / @as(f32, @floatFromInt(self.points.items.len)),
            .time_span = latest - earliest,
        };
    }
};

/// Time series data structure (alias for TimeSeriesBuffer for backward compatibility)
pub const TimeSeries = TimeSeriesBuffer;
