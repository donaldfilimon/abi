//! Metrics collector
//!
//! Bounded ring buffer for performance metrics.

const std = @import("std");

pub const MetricsCollector = struct {
    buffer: []f64,
    index: std.atomic.Value(usize),
    count: std.atomic.Value(usize),

    pub fn init(allocator: std.mem.Allocator, size: usize) !MetricsCollector {
        const buffer = try allocator.alloc(f64, size);
        @memset(buffer, 0.0);

        return .{
            .buffer = buffer,
            .index = std.atomic.Value(usize).init(0),
            .count = std.atomic.Value(usize).init(0),
        };
    }

    pub fn deinit(self: *MetricsCollector, allocator: std.mem.Allocator) void {
        allocator.free(self.buffer);
        self.* = undefined;
    }

    pub fn record(self: *MetricsCollector, value: f64) void {
        const idx = self.index.fetchAdd(1, .monotonic) % self.buffer.len;
        self.buffer[idx] = value;
        _ = self.count.fetchAdd(1, .monotonic);
    }
};
