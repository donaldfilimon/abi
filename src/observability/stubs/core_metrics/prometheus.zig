//! Prometheus Text Format Export (Stub)
//!
//! Stub implementation of metrics export in Prometheus format.
//! All write operations are no-ops when observability is disabled.

const std = @import("std");

/// Stub MetricWriter - all operations are no-ops.
pub const MetricWriter = struct {
    const Self = @This();
    output: std.ArrayListUnmanaged(u8) = .empty,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) Self {
        return .{ .allocator = allocator };
    }

    pub fn deinit(_: *Self) void {}

    pub fn writeCounter(_: *Self, _: []const u8, _: []const u8, _: u64, _: ?[]const u8) !void {}

    pub fn writeGauge(_: *Self, _: []const u8, _: []const u8, _: anytype, _: ?[]const u8) !void {}

    pub fn writeHistogram(
        _: *Self,
        _: []const u8,
        _: []const u8,
        comptime _: usize,
        _: anytype,
        _: ?[]const u8,
    ) !void {}

    pub fn finish(self: *Self) ![]u8 {
        return try self.allocator.alloc(u8, 0);
    }

    pub fn clear(_: *Self) void {}
};
