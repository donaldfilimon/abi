//! Abbey Enhanced Reasoning Module
//!
//! Provides step-by-step reasoning capabilities for Abbey, enabling
//! transparent and explainable thought processes. Integrates with
//! emotion awareness for empathetic problem-solving.
//!
//! Features:
//! - Multi-step reasoning chains
//! - Confidence tracking per step
//! - Emotion-aware reasoning adaptation
//! - Memory context integration
//! - Step formatting and presentation

const std = @import("std");
const emotion = @import("emotion.zig");
const core_types = @import("../../core/types.zig");

/// A single step in the reasoning chain.
pub const ReasoningStep = struct {
    /// Step number (1-indexed).
    step_number: usize,
    /// Title/summary of this step.
    title: []const u8,
    /// Detailed explanation.
    explanation: []const u8,
    /// Confidence in this step (0.0 - 1.0).
    confidence: f32,
    /// Type of reasoning used.
    reasoning_type: ReasoningType,
    /// Evidence or sources supporting this step.
    evidence: ?[]const u8 = null,
    /// Whether this step involves uncertainty.
    involves_uncertainty: bool = false,
    /// Whether this step requires user validation.
    needs_validation: bool = false,
};

/// Types of reasoning approaches.
pub const ReasoningType = enum {
    /// Breaking down complex problems.
    decomposition,
    /// Drawing from similar cases.
    analogy,
    /// Logical if-then reasoning.
    deduction,
    /// Building from specific to general.
    induction,
    /// Eliminating alternatives.
    elimination,
    /// Testing hypotheses.
    hypothesis_testing,
    /// Pattern recognition.
    pattern_matching,
    /// Combining information.
    synthesis,
    /// Considering user emotions.
    emotional_consideration,
    /// Presenting options.
    option_analysis,

    pub fn getDescription(self: ReasoningType) []const u8 {
        return switch (self) {
            .decomposition => "Breaking down the problem",
            .analogy => "Drawing from similar situations",
            .deduction => "Following logical implications",
            .induction => "Building from observations",
            .elimination => "Ruling out alternatives",
            .hypothesis_testing => "Testing possibilities",
            .pattern_matching => "Recognizing patterns",
            .synthesis => "Combining information",
            .emotional_consideration => "Considering emotional context",
            .option_analysis => "Analyzing available options",
        };
    }
};

/// Complete reasoning chain from query to conclusion.
pub const ReasoningChain = struct {
    /// The original query being reasoned about.
    query: []const u8,
    /// Ordered list of reasoning steps.
    steps: std.ArrayList(ReasoningStep),
    /// Overall confidence in the chain.
    overall_confidence: f32,
    /// Final conclusion/answer.
    conclusion: ?[]const u8,
    /// Time taken to reason (ms).
    reasoning_time_ms: u64,
    /// Whether the reasoning was emotion-aware.
    emotion_aware: bool,
    /// Emotional context if available.
    emotional_context: ?emotion.EmotionalResponse,

    const Self = @This();

    /// Initialize a new reasoning chain.
    pub fn init(allocator: std.mem.Allocator, query: []const u8) Self {
        return .{
            .query = query,
            .steps = std.ArrayList(ReasoningStep).init(allocator),
            .overall_confidence = 1.0,
            .conclusion = null,
            .reasoning_time_ms = 0,
            .emotion_aware = false,
            .emotional_context = null,
        };
    }

    /// Deinitialize and free resources.
    pub fn deinit(self: *Self, allocator: std.mem.Allocator) void {
        for (self.steps.items) |step| {
            allocator.free(step.title);
            allocator.free(step.explanation);
            if (step.evidence) |ev| allocator.free(ev);
        }
        self.steps.deinit();
        if (self.conclusion) |c| allocator.free(c);
    }

    /// Add a step to the chain.
    pub fn addStep(self: *Self, step: ReasoningStep) !void {
        try self.steps.append(step);
        // Update overall confidence (product of step confidences)
        self.overall_confidence = @min(self.overall_confidence, step.confidence);
    }

    /// Set the final conclusion.
    pub fn setConclusion(self: *Self, conclusion: []const u8) void {
        self.conclusion = conclusion;
    }

    /// Get the number of steps.
    pub fn stepCount(self: *const Self) usize {
        return self.steps.items.len;
    }

    /// Check if any step needs validation.
    pub fn needsValidation(self: *const Self) bool {
        for (self.steps.items) |step| {
            if (step.needs_validation) return true;
        }
        return false;
    }
};

/// Configuration for the reasoning engine.
pub const ReasoningConfig = struct {
    /// Maximum number of reasoning steps.
    max_steps: usize = 10,
    /// Minimum confidence to continue reasoning.
    min_confidence_threshold: f32 = 0.3,
    /// Whether to include emotional considerations.
    emotion_aware: bool = true,
    /// Whether to show intermediate steps.
    show_intermediate_steps: bool = true,
    /// Format for step presentation.
    output_format: OutputFormat = .detailed,

    pub const OutputFormat = enum {
        /// Full explanation for each step.
        detailed,
        /// Brief summary of steps.
        summary,
        /// Just the conclusion.
        conclusion_only,
        /// Bullet points.
        bullet_points,
    };
};

/// Memory context for reasoning.
pub const MemoryContext = struct {
    /// Previous interactions relevant to this query.
    relevant_history: []const []const u8 = &.{},
    /// User preferences learned over time.
    user_preferences: ?UserPreferences = null,
    /// Domain knowledge available.
    domain_knowledge: []const []const u8 = &.{},
    /// Current conversation topics.
    active_topics: []const []const u8 = &.{},

    pub const UserPreferences = struct {
        prefers_detail: bool = false,
        prefers_examples: bool = true,
        technical_level: TechnicalLevel = .intermediate,
    };

    pub const TechnicalLevel = enum {
        beginner,
        intermediate,
        advanced,
        expert,
    };
};

/// The enhanced reasoning engine.
pub const ReasoningEngine = struct {
    allocator: std.mem.Allocator,
    config: ReasoningConfig,

    const Self = @This();

    /// Initialize the reasoning engine.
    pub fn init(allocator: std.mem.Allocator) Self {
        return initWithConfig(allocator, .{});
    }

    /// Initialize with custom configuration.
    pub fn initWithConfig(allocator: std.mem.Allocator, config: ReasoningConfig) Self {
        return .{
            .allocator = allocator,
            .config = config,
        };
    }

    /// Perform reasoning on a query.
    pub fn reason(
        self: *Self,
        query: []const u8,
        context: MemoryContext,
        emotional_context: ?emotion.EmotionalResponse,
    ) !ReasoningChain {
        var timer = std.time.Timer.start() catch {
            return error.TimerFailed;
        };

        var chain = ReasoningChain.init(self.allocator, query);
        chain.emotion_aware = self.config.emotion_aware and emotional_context != null;
        chain.emotional_context = emotional_context;

        // Step 1: Understand the query
        try chain.addStep(.{
            .step_number = 1,
            .title = try self.allocator.dupe(u8, "Understanding the question"),
            .explanation = try self.allocator.dupe(u8, "Analyzing the query to identify key components and intent."),
            .confidence = 0.95,
            .reasoning_type = .decomposition,
        });

        // Step 2: Consider emotional context if emotion-aware
        if (chain.emotion_aware) {
            if (emotional_context) |emo| {
                const emo_step = try self.createEmotionalStep(emo, chain.stepCount() + 1);
                try chain.addStep(emo_step);
            }
        }

        // Step 3: Recall relevant context
        if (context.relevant_history.len > 0 or context.domain_knowledge.len > 0) {
            try chain.addStep(.{
                .step_number = chain.stepCount() + 1,
                .title = try self.allocator.dupe(u8, "Recalling relevant context"),
                .explanation = try self.allocator.dupe(u8, "Drawing from conversation history and domain knowledge."),
                .confidence = 0.9,
                .reasoning_type = .pattern_matching,
            });
        }

        // Step 4: Analyze the problem
        try chain.addStep(.{
            .step_number = chain.stepCount() + 1,
            .title = try self.allocator.dupe(u8, "Analyzing the problem"),
            .explanation = try self.allocator.dupe(u8, "Breaking down the problem into manageable components."),
            .confidence = 0.85,
            .reasoning_type = .decomposition,
        });

        // Step 5: Formulate approach
        try chain.addStep(.{
            .step_number = chain.stepCount() + 1,
            .title = try self.allocator.dupe(u8, "Formulating approach"),
            .explanation = try self.allocator.dupe(u8, "Determining the best method to address this query."),
            .confidence = 0.8,
            .reasoning_type = .synthesis,
        });

        chain.reasoning_time_ms = timer.read() / std.time.ns_per_ms;

        return chain;
    }

    /// Create a reasoning step for emotional consideration.
    fn createEmotionalStep(self: *Self, emo: emotion.EmotionalResponse, step_num: usize) !ReasoningStep {
        var explanation_buf = std.ArrayList(u8).init(self.allocator);
        defer explanation_buf.deinit();

        try explanation_buf.appendSlice("User appears to be feeling ");
        try explanation_buf.appendSlice(@tagName(emo.primary_emotion));
        try explanation_buf.appendSlice(". ");

        if (emo.needs_special_care) {
            try explanation_buf.appendSlice("Special care needed. ");
        }

        try explanation_buf.appendSlice("Adjusting response tone to be ");
        try explanation_buf.appendSlice(emo.suggested_tone.getDescription());
        try explanation_buf.appendSlice(".");

        return .{
            .step_number = step_num,
            .title = try self.allocator.dupe(u8, "Considering emotional context"),
            .explanation = try explanation_buf.toOwnedSlice(),
            .confidence = 0.9,
            .reasoning_type = .emotional_consideration,
        };
    }

    /// Format reasoning steps for output.
    pub fn formatSteps(self: *Self, chain: *const ReasoningChain) ![]const u8 {
        var result = std.ArrayList(u8).init(self.allocator);
        errdefer result.deinit();

        switch (self.config.output_format) {
            .detailed => {
                for (chain.steps.items) |step| {
                    try std.fmt.format(result.writer(), "**Step {d}: {s}**\n", .{ step.step_number, step.title });
                    try result.appendSlice(step.explanation);
                    try result.appendSlice("\n\n");
                }
            },
            .summary => {
                try result.appendSlice("Reasoning: ");
                for (chain.steps.items, 0..) |step, i| {
                    try result.appendSlice(step.title);
                    if (i < chain.steps.items.len - 1) {
                        try result.appendSlice(" -> ");
                    }
                }
                try result.append('\n');
            },
            .bullet_points => {
                for (chain.steps.items) |step| {
                    try std.fmt.format(result.writer(), "- {s}\n", .{step.title});
                }
            },
            .conclusion_only => {
                if (chain.conclusion) |conclusion| {
                    try result.appendSlice(conclusion);
                }
            },
        }

        // Add confidence indicator
        if (self.config.output_format != .conclusion_only) {
            try std.fmt.format(result.writer(), "\nConfidence: {d:.0}%\n", .{chain.overall_confidence * 100});
        }

        return result.toOwnedSlice();
    }

    /// Get reasoning step recommendations based on query type.
    pub fn getRecommendedSteps(self: *const Self, query: []const u8) []const ReasoningType {
        _ = self;
        // Analyze query to recommend reasoning approaches
        const lower = query;

        // Check for question patterns
        if (std.mem.indexOf(u8, lower, "why") != null) {
            return &[_]ReasoningType{ .decomposition, .deduction, .synthesis };
        }
        if (std.mem.indexOf(u8, lower, "how") != null) {
            return &[_]ReasoningType{ .decomposition, .pattern_matching, .synthesis };
        }
        if (std.mem.indexOf(u8, lower, "which") != null or std.mem.indexOf(u8, lower, "should i") != null) {
            return &[_]ReasoningType{ .option_analysis, .elimination, .synthesis };
        }
        if (std.mem.indexOf(u8, lower, "compare") != null) {
            return &[_]ReasoningType{ .analogy, .option_analysis, .synthesis };
        }
        if (std.mem.indexOf(u8, lower, "debug") != null or std.mem.indexOf(u8, lower, "error") != null) {
            return &[_]ReasoningType{ .hypothesis_testing, .elimination, .pattern_matching };
        }

        // Default reasoning flow
        return &[_]ReasoningType{ .decomposition, .pattern_matching, .synthesis };
    }
};

/// Utility to combine reasoning from multiple sources.
pub fn combineReasoningChains(
    allocator: std.mem.Allocator,
    chains: []const ReasoningChain,
    primary_index: usize,
) !ReasoningChain {
    if (chains.len == 0) return error.NoChains;

    var combined = ReasoningChain.init(allocator, chains[primary_index].query);

    // Add steps from primary chain
    for (chains[primary_index].steps.items) |step| {
        try combined.addStep(.{
            .step_number = combined.stepCount() + 1,
            .title = try allocator.dupe(u8, step.title),
            .explanation = try allocator.dupe(u8, step.explanation),
            .confidence = step.confidence,
            .reasoning_type = step.reasoning_type,
            .evidence = if (step.evidence) |ev| try allocator.dupe(u8, ev) else null,
            .involves_uncertainty = step.involves_uncertainty,
            .needs_validation = step.needs_validation,
        });
    }

    // Average confidence across all chains
    var total_confidence: f32 = 0;
    for (chains) |chain| {
        total_confidence += chain.overall_confidence;
    }
    combined.overall_confidence = total_confidence / @as(f32, @floatFromInt(chains.len));

    return combined;
}

// Tests

test "reasoning engine initialization" {
    const engine = ReasoningEngine.init(std.testing.allocator);
    try std.testing.expectEqual(@as(usize, 10), engine.config.max_steps);
    try std.testing.expect(engine.config.emotion_aware);
}

test "reason without emotional context" {
    var engine = ReasoningEngine.init(std.testing.allocator);

    var chain = try engine.reason("How do I fix this bug?", .{}, null);
    defer chain.deinit(std.testing.allocator);

    try std.testing.expect(chain.stepCount() >= 3);
    try std.testing.expect(!chain.emotion_aware);
}

test "reason with emotional context" {
    var engine = ReasoningEngine.init(std.testing.allocator);

    const emo = emotion.EmotionalResponse{
        .primary_emotion = .frustrated,
        .intensity = 0.8,
        .suggested_tone = .empathetic,
        .empathy_level = 0.9,
        .needs_special_care = true,
    };

    var chain = try engine.reason("Why doesn't this work?", .{}, emo);
    defer chain.deinit(std.testing.allocator);

    try std.testing.expect(chain.emotion_aware);
    try std.testing.expect(chain.emotional_context != null);
}

test "format steps detailed" {
    var engine = ReasoningEngine.initWithConfig(std.testing.allocator, .{ .output_format = .detailed });

    var chain = try engine.reason("Test query", .{}, null);
    defer chain.deinit(std.testing.allocator);

    const formatted = try engine.formatSteps(&chain);
    defer std.testing.allocator.free(formatted);

    try std.testing.expect(std.mem.indexOf(u8, formatted, "Step") != null);
}

test "format steps summary" {
    var engine = ReasoningEngine.initWithConfig(std.testing.allocator, .{ .output_format = .summary });

    var chain = try engine.reason("Test query", .{}, null);
    defer chain.deinit(std.testing.allocator);

    const formatted = try engine.formatSteps(&chain);
    defer std.testing.allocator.free(formatted);

    try std.testing.expect(std.mem.indexOf(u8, formatted, "->") != null);
}

test "recommended steps for why questions" {
    const engine = ReasoningEngine.init(std.testing.allocator);
    const steps = engine.getRecommendedSteps("Why does this happen?");

    try std.testing.expectEqual(ReasoningType.decomposition, steps[0]);
}

test "reasoning chain confidence" {
    var chain = ReasoningChain.init(std.testing.allocator, "test");
    defer chain.deinit(std.testing.allocator);

    try chain.addStep(.{
        .step_number = 1,
        .title = try std.testing.allocator.dupe(u8, "Step 1"),
        .explanation = try std.testing.allocator.dupe(u8, "Explanation"),
        .confidence = 0.9,
        .reasoning_type = .decomposition,
    });

    try chain.addStep(.{
        .step_number = 2,
        .title = try std.testing.allocator.dupe(u8, "Step 2"),
        .explanation = try std.testing.allocator.dupe(u8, "Explanation"),
        .confidence = 0.8,
        .reasoning_type = .synthesis,
    });

    // Overall confidence should be minimum of steps
    try std.testing.expectApproxEqAbs(@as(f32, 0.8), chain.overall_confidence, 0.01);
}
