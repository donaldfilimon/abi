//! Learning facade stub for disabled AI builds.

const std = @import("std");

pub const FeedbackKind = enum {
    positive,
    negative,
    neutral,

    pub fn fromString(_: []const u8) ?FeedbackKind {
        return null;
    }

    pub fn label(self: FeedbackKind) []const u8 {
        return switch (self) {
            .positive => "positive",
            .negative => "negative",
            .neutral => "neutral",
        };
    }
};

pub const Interaction = struct {
    prompt: []const u8,
    response: []const u8,
    profile: []const u8 = "unknown",
    backend: []const u8 = "unknown",
    latency_ms: f32 = 0,
    selected_model: []const u8 = "",
    quality_score: ?f32 = null,
    wdbx_block_id: ?u64 = null,
    route_reason: []const u8 = "",
    retrieval_hits: usize = 0,
    constitution_passed: bool = true,
    used_fallback_provider: bool = false,
};

pub const LearningReport = struct {
    total_interactions: usize = 0,
    avg_quality: f32 = 0,
    positive_feedback_count: usize = 0,
    negative_feedback_count: usize = 0,
    stored_events: usize = 0,
    auto_retrain_enabled: bool = false,
};

pub const LearningRuntime = struct {
    pub fn init(_: std.mem.Allocator) !LearningRuntime {
        return .{};
    }
    pub fn deinit(_: *LearningRuntime) void {}
    pub fn recordInteraction(_: *LearningRuntime, _: Interaction) !void {}
    pub fn recordFeedback(_: *LearningRuntime, _: FeedbackKind, _: ?[]const u8) !void {}
    pub fn report(_: *LearningRuntime) LearningReport {
        return .{};
    }
    pub fn maybeTriggerRetrain(_: *LearningRuntime) !bool {
        return false;
    }
    pub fn forceRetrain(_: *LearningRuntime) !bool {
        return false;
    }
    pub fn exportArtifacts(_: *LearningRuntime, _: []const u8) !usize {
        return 0;
    }
};

test {
    std.testing.refAllDecls(@This());
}
