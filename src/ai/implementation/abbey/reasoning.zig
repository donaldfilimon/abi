//! Abbey Reasoning System
//!
//! Provides structured chain-of-thought reasoning with:
//! - Step-by-step reasoning tracking
//! - Confidence calibration at each step
//! - Research triggers when confidence is low
//! - Reasoning summaries for transparency

const std = @import("std");
const core_types = @import("../../core/types.zig");

// Re-export canonical types from core for consistency
pub const ConfidenceLevel = core_types.ConfidenceLevel;
pub const Confidence = core_types.Confidence;

// Zig 0.16 compatible time function
fn getTimestampNs() i128 {
    return @intCast(std.time.milliTimestamp() * std.time.ns_per_ms);
}

/// Helper to create a display string for confidence (extends core Confidence)
pub fn confidenceToDisplayString(conf: Confidence, allocator: std.mem.Allocator) ![]u8 {
    return std.fmt.allocPrint(allocator, "{s} ({d:.0}%): {s}", .{
        @tagName(conf.level),
        conf.score * 100,
        conf.reasoning,
    });
}

/// Type of reasoning step
pub const StepType = enum {
    /// Initial assessment of the query
    assessment,
    /// Breaking down the problem
    decomposition,
    /// Retrieving relevant information
    retrieval,
    /// Analyzing retrieved information
    analysis,
    /// Synthesizing conclusions
    synthesis,
    /// Indicating research is needed
    research,
    /// Validating the response
    validation,
    /// Formulating the final response
    response,

    pub fn toString(self: StepType) []const u8 {
        return switch (self) {
            .assessment => "Assessment",
            .decomposition => "Decomposition",
            .retrieval => "Retrieval",
            .analysis => "Analysis",
            .synthesis => "Synthesis",
            .research => "Research Needed",
            .validation => "Validation",
            .response => "Response Formation",
        };
    }

    pub fn getEmoji(self: StepType) []const u8 {
        return switch (self) {
            .assessment => "[?]",
            .decomposition => "[/]",
            .retrieval => "[>]",
            .analysis => "[~]",
            .synthesis => "[+]",
            .research => "[!]",
            .validation => "[v]",
            .response => "[=]",
        };
    }
};

/// A single reasoning step
pub const ReasoningStep = struct {
    step_type: StepType,
    description: []const u8,
    confidence: Confidence,
    timestamp_ns: i128,
    duration_ns: u64 = 0,

    /// Format step for display
    pub fn format(self: ReasoningStep, allocator: std.mem.Allocator) ![]u8 {
        return std.fmt.allocPrint(allocator, "{s} {s}: {s} (confidence: {t})", .{
            self.step_type.getEmoji(),
            self.step_type.toString(),
            self.description,
            self.confidence.level,
        });
    }
};

/// Chain of reasoning for a single query
pub const ReasoningChain = struct {
    allocator: std.mem.Allocator,
    query: []const u8,
    steps: std.ArrayListUnmanaged(ReasoningStep),
    start_time_ns: i128,
    finalized: bool = false,
    overall_confidence: ?Confidence = null,

    const Self = @This();

    /// Initialize a new reasoning chain
    pub fn init(allocator: std.mem.Allocator, query: []const u8) Self {
        return .{
            .allocator = allocator,
            .query = query,
            .steps = .{},
            .start_time_ns = getTimestampNs(),
        };
    }

    /// Clean up
    pub fn deinit(self: *Self) void {
        self.steps.deinit(self.allocator);
    }

    /// Add a reasoning step
    pub fn addStep(self: *Self, step_type: StepType, description: []const u8, confidence: Confidence) !void {
        if (self.finalized) return error.ChainFinalized;

        try self.steps.append(self.allocator, .{
            .step_type = step_type,
            .description = description,
            .confidence = confidence,
            .timestamp_ns = getTimestampNs(),
        });
    }

    /// Finalize the chain and calculate overall confidence
    pub fn finalize(self: *Self) !void {
        if (self.finalized) return;
        self.finalized = true;

        if (self.steps.items.len == 0) {
            self.overall_confidence = .{
                .level = .unknown,
                .score = 0.0,
                .reasoning = "No reasoning steps recorded",
            };
            return;
        }

        // Calculate weighted average confidence
        var total_score: f32 = 0;
        var weight_sum: f32 = 0;
        var lowest_level: ConfidenceLevel = .certain;

        for (self.steps.items, 0..) |step, i| {
            // Later steps are weighted more heavily
            const weight: f32 = @as(f32, @floatFromInt(i + 1));
            total_score += step.confidence.score * weight;
            weight_sum += weight;

            // Track the lowest confidence level (higher enum value = lower confidence)
            if (@intFromEnum(step.confidence.level) > @intFromEnum(lowest_level)) {
                lowest_level = step.confidence.level;
            }
        }

        const avg_score = if (weight_sum > 0) total_score / weight_sum else 0;

        // Use ConfidenceLevel.fromScore for consistent level determination
        const score_based_level = ConfidenceLevel.fromScore(avg_score);

        // Determine final level: use the worse of score-based or lowest step level
        const final_level: ConfidenceLevel = if (@intFromEnum(lowest_level) > @intFromEnum(score_based_level))
            lowest_level
        else
            score_based_level;

        self.overall_confidence = .{
            .level = final_level,
            .score = avg_score,
            .reasoning = if (lowest_level == .unknown)
                "Contains steps with unknown confidence"
            else if (lowest_level.needsResearch())
                "Contains steps requiring verification"
            else
                "Reasoning chain completed successfully",
        };
    }

    /// Get overall confidence (auto-finalizes if needed)
    pub fn getOverallConfidence(self: *Self) Confidence {
        if (!self.finalized) {
            self.finalize() catch {};
        }
        return self.overall_confidence orelse .{
            .level = .unknown,
            .score = 0.0,
            .reasoning = "Unable to determine confidence",
        };
    }

    /// Get step count
    pub fn stepCount(self: *const Self) usize {
        return self.steps.items.len;
    }

    /// Check if research was triggered
    pub fn researchTriggered(self: *const Self) bool {
        for (self.steps.items) |step| {
            if (step.step_type == .research) return true;
        }
        return false;
    }

    /// Get a summary of the reasoning
    pub fn getSummary(self: *Self, allocator: std.mem.Allocator) ![]u8 {
        if (!self.finalized) {
            try self.finalize();
        }

        var result = std.ArrayListUnmanaged(u8){};
        errdefer result.deinit(allocator);

        try result.appendSlice(allocator, "Reasoning Summary:\n");
        try result.appendSlice(allocator, "─────────────────\n");

        for (self.steps.items) |step| {
            try result.appendSlice(allocator, step.step_type.getEmoji());
            try result.appendSlice(allocator, " ");
            try result.appendSlice(allocator, step.description);
            try result.appendSlice(allocator, "\n");
        }

        if (self.overall_confidence) |conf| {
            try result.appendSlice(allocator, "\nOverall: ");
            try result.appendSlice(allocator, @tagName(conf.level));
            try result.appendSlice(allocator, " confidence\n");
        }

        return result.toOwnedSlice(allocator);
    }

    /// Get reasoning as JSON-compatible structure
    pub fn toJson(self: *Self, allocator: std.mem.Allocator) ![]u8 {
        var result = std.ArrayListUnmanaged(u8){};
        errdefer result.deinit(allocator);

        try result.appendSlice(allocator, "{\"query\":\"");
        // Escape the query
        for (self.query) |c| {
            switch (c) {
                '"' => try result.appendSlice(allocator, "\\\""),
                '\\' => try result.appendSlice(allocator, "\\\\"),
                '\n' => try result.appendSlice(allocator, "\\n"),
                else => try result.append(allocator, c),
            }
        }
        try result.appendSlice(allocator, "\",\"steps\":[");

        for (self.steps.items, 0..) |step, i| {
            if (i > 0) try result.appendSlice(allocator, ",");
            try result.appendSlice(allocator, "{\"type\":\"");
            try result.appendSlice(allocator, step.step_type.toString());
            try result.appendSlice(allocator, "\",\"confidence\":\"");
            try result.appendSlice(allocator, @tagName(step.confidence.level));
            try result.appendSlice(allocator, "\"}");
        }

        try result.appendSlice(allocator, "]}");
        return result.toOwnedSlice(allocator);
    }
};

// ============================================================================
// Tests
// ============================================================================

test "confidence level" {
    // Using core types: .certain/.high/.medium/.low/.uncertain/.unknown
    const high_conf = Confidence{ .level = .high, .score = 0.9, .reasoning = "test" };
    try std.testing.expect(!high_conf.level.needsResearch());

    const low_conf = Confidence{ .level = .low, .score = 0.45, .reasoning = "test" };
    try std.testing.expect(low_conf.level.needsResearch());
}

test "reasoning chain" {
    const allocator = std.testing.allocator;

    var chain = ReasoningChain.init(allocator, "What is Zig?");
    defer chain.deinit();

    // Use core ConfidenceLevel values (.certain, .high, .medium, .low, .uncertain, .unknown)
    try chain.addStep(.assessment, "Analyzing query", .{ .level = .high, .score = 0.9, .reasoning = "Common topic" });
    try chain.addStep(.retrieval, "Retrieving Zig information", .{ .level = .high, .score = 0.85, .reasoning = "Well-documented" });
    try chain.addStep(.response, "Formulating response", .{ .level = .high, .score = 0.9, .reasoning = "Clear answer" });

    try std.testing.expectEqual(@as(usize, 3), chain.stepCount());

    try chain.finalize();
    try std.testing.expect(chain.finalized);

    const conf = chain.getOverallConfidence();
    // Expect high confidence based on weighted average
    try std.testing.expect(conf.score >= 0.8);
}

test "research trigger" {
    const allocator = std.testing.allocator;

    var chain = ReasoningChain.init(allocator, "Latest news?");
    defer chain.deinit();

    try chain.addStep(.assessment, "Analyzing query", .{ .level = .low, .score = 0.3, .reasoning = "Time-sensitive" });
    try chain.addStep(.research, "Research required", .{ .level = .low, .score = 0.2, .reasoning = "Need current data" });

    try std.testing.expect(chain.researchTriggered());
}
