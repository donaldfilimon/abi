//! Self-Improvement Stub Module
//!
//! Stub implementation of the self-improvement system when AI is disabled.

const std = @import("std");

pub const ImprovementStatus = enum {
    proposed,
    approved,
    applied,
    rejected,
};

pub const ImprovementPriority = enum {
    low,
    medium,
    high,
    critical,
};

pub const Improvement = struct {
    description: []const u8,
    target_file: []const u8,
    suggested_change: []const u8,
    confidence: f32,
    priority: ImprovementPriority,
    status: ImprovementStatus,
    created_at: i64,
};

pub const ResponseMetrics = struct {
    coherence: f32 = 0,
    relevance: f32 = 0,
    completeness: f32 = 0,
    clarity: f32 = 0,
    overall: f32 = 0,
    tool_calls_made: usize = 0,
    tool_success_rate: f32 = 0,
    response_length: usize = 0,
};

pub const Trend = enum {
    improving,
    stable,
    declining,
};

pub const PerformanceReport = struct {
    total_interactions: usize,
    avg_quality: f32,
    positive_feedback_count: usize,
    negative_feedback_count: usize,
    tool_usage_count: usize,
    avg_tool_success_rate: f32,
    improvement_count: usize,
    trend: Trend,
};

pub const SelfImprover = struct {
    allocator: std.mem.Allocator,

    const Self = @This();

    pub fn init(allocator: std.mem.Allocator) Self {
        return .{ .allocator = allocator };
    }

    pub fn deinit(_: *Self) void {}

    pub fn recordMetrics(_: *Self, _: ResponseMetrics) !void {}

    pub fn recordFeedback(_: *Self, _: bool) void {}

    pub fn evaluateResponse(_: *Self, _: []const u8, _: []const u8) ResponseMetrics {
        return .{};
    }

    pub fn getReport(_: *const Self) PerformanceReport {
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

    pub fn getLatestMetrics(_: *const Self) ?ResponseMetrics {
        return null;
    }

    pub fn getImprovements(_: *const Self) []const Improvement {
        return &[_]Improvement{};
    }
};

test {
    std.testing.refAllDecls(@This());
}
