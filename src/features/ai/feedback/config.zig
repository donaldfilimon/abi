//! Configuration types for the Feedback Collection System.

const std = @import("std");

/// Configuration for the feedback system.
pub const FeedbackConfig = struct {
    /// Maximum number of feedback entries to retain in memory.
    max_entries: u32 = 10_000,
    /// Whether to enable real-time analysis of incoming feedback.
    enable_analysis: bool = true,
    /// Minimum number of entries before analysis results are considered reliable.
    min_analysis_threshold: u32 = 10,
    /// Whether to track per-persona satisfaction scores.
    track_persona_scores: bool = true,
    /// Maximum length of text feedback (in bytes).
    max_text_length: u32 = 2048,

    pub fn defaults() FeedbackConfig {
        return .{};
    }
};

test {
    std.testing.refAllDecls(@This());
}

test "FeedbackConfig defaults" {
    const cfg = FeedbackConfig.defaults();
    try std.testing.expect(cfg.max_entries == 10_000);
    try std.testing.expect(cfg.enable_analysis);
}
