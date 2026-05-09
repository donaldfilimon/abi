//! Self-Improvement Stub Module
//!
//! Stub implementation of the self-improvement system when AI is disabled.

const std = @import("std");
const types = @import("self_improve_types.zig");

pub const ImprovementStatus = types.ImprovementStatus;
pub const ImprovementPriority = types.ImprovementPriority;
pub const Improvement = types.Improvement;
pub const ResponseMetrics = types.ResponseMetrics;
pub const Trend = types.Trend;
pub const PerformanceReport = types.PerformanceReport;

pub const SelfImprover = struct {
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) SelfImprover {
        return .{ .allocator = allocator };
    }

    pub fn deinit(_: *SelfImprover) void {}

    pub fn recordMetrics(_: *SelfImprover, _: ResponseMetrics) !void {}

    pub fn recordFeedback(_: *SelfImprover, _: bool) void {}

    pub fn evaluateResponse(_: *SelfImprover, _: []const u8, _: []const u8) ResponseMetrics {
        return .{};
    }

    pub fn getReport(_: *const SelfImprover) PerformanceReport {
        return .{
            .total_interactions = 0,
            .avg_quality = 0,
            .positive_feedback_count = 0,
            .negative_feedback_count = 0,
            .tool_usage_count = 0,
            .avg_tool_success_rate = 0,
            .improvement_count = 0,
            .trend = .stable,
        };
    }

    pub fn getLatestMetrics(_: *const SelfImprover) ?ResponseMetrics {
        return null;
    }

    pub fn getImprovements(_: *const SelfImprover) []const Improvement {
        return &[_]Improvement{};
    }
};

test {
    std.testing.refAllDecls(@This());
}
