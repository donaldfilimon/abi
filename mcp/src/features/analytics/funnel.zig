//! Funnel Tracking
//!
//! Track progression through named conversion funnels.

const std = @import("std");
const types = @import("types.zig");

/// Track progression through a named funnel.
pub const Funnel = struct {
    name: []const u8,
    steps: std.ArrayListUnmanaged(Step) = .empty,
    allocator: std.mem.Allocator,

    pub const Step = types.FunnelStep;

    pub fn init(allocator: std.mem.Allocator, name: []const u8) Funnel {
        return .{ .name = name, .allocator = allocator };
    }

    pub fn deinit(self: *Funnel) void {
        self.steps.deinit(self.allocator);
        self.* = undefined;
    }

    /// Add a step to the funnel.
    pub fn addStep(self: *Funnel, step_name: []const u8) !void {
        try self.steps.append(self.allocator, .{ .name = step_name });
    }

    /// Record a user reaching a step.
    pub fn recordStep(self: *Funnel, step_index: usize) void {
        if (step_index < self.steps.items.len) {
            _ = self.steps.items[step_index].count.fetchAdd(1, .monotonic);
        }
    }

    /// Get step counts for analysis.
    pub fn getStepCounts(self: *const Funnel, buffer: []u64) []u64 {
        const len = @min(buffer.len, self.steps.items.len);
        for (0..len) |i| {
            buffer[i] = self.steps.items[i].count.load(.monotonic);
        }
        return buffer[0..len];
    }
};
