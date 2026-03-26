//! Shared types for the self-improvement module.
//!
//! Used by both self_improve.zig (mod) and self_improve_stub.zig (stub)
//! to prevent type drift between enabled and disabled paths.

const std = @import("std");

/// Status of a proposed improvement.
pub const ImprovementStatus = enum {
    proposed,
    approved,
    applied,
    rejected,
};

/// Priority level for an improvement.
pub const ImprovementPriority = enum {
    low,
    medium,
    high,
    critical,
};

/// A proposed improvement to the agent's behavior or code.
pub const Improvement = struct {
    description: []const u8,
    target_file: []const u8,
    suggested_change: []const u8,
    confidence: f32,
    priority: ImprovementPriority,
    status: ImprovementStatus,
    created_at: i64,
};

/// Quality metrics for a single response.
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

/// Performance trend direction.
pub const Trend = enum {
    improving,
    stable,
    declining,
};

/// Aggregated performance metrics over time.
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

test {
    std.testing.refAllDecls(@This());
}
