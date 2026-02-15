//! Abbey Self-Reflection System
//!
//! Meta-cognitive capabilities for monitoring and improving reasoning:
//! - Response quality self-evaluation
//! - Uncertainty quantification
//! - Bias detection
//! - Reasoning path analysis
//! - Self-correction mechanisms
//! - Performance monitoring
//! - Continuous improvement tracking

const std = @import("std");
const types = @import("../../core/types.zig");
const reasoning = @import("../reasoning.zig");
const calibration = @import("../calibration.zig");

// ============================================================================
// Self-Reflection Types
// ============================================================================

/// Self-evaluation of a response
pub const SelfEvaluation = struct {
    response_id: u64,
    timestamp: i64,

    // Quality dimensions
    coherence_score: f32,
    relevance_score: f32,
    completeness_score: f32,
    accuracy_confidence: f32,
    clarity_score: f32,

    // Meta-cognitive assessments
    uncertainty_areas: []const UncertaintyArea,
    potential_biases: []const DetectedBias,
    reasoning_quality: ReasoningQuality,
    improvement_suggestions: []const ImprovementSuggestion,

    // Overall assessment
    overall_quality: f32,
    should_revise: bool,
    revision_priority: RevisionPriority,

    pub const RevisionPriority = enum {
        none,
        low,
        medium,
        high,
        critical,
    };
};

/// Area of uncertainty in a response
pub const UncertaintyArea = struct {
    location: []const u8, // What part of the response
    uncertainty_type: UncertaintyType,
    confidence: f32,
    mitigation: ?[]const u8,

    pub const UncertaintyType = enum {
        factual, // Uncertainty about facts
        temporal, // Uncertainty about timing/currency
        scope, // Uncertainty about scope/completeness
        interpretation, // Uncertainty about user intent
        logical, // Uncertainty about reasoning
        external, // Uncertainty requiring external verification
    };
};

/// Detected bias in reasoning
pub const DetectedBias = struct {
    bias_type: BiasType,
    severity: f32,
    evidence: []const u8,
    correction: ?[]const u8,

    pub const BiasType = enum {
        confirmation, // Confirming pre-existing beliefs
        anchoring, // Over-relying on first information
        availability, // Over-weighting easily recalled info
        recency, // Over-weighting recent information
        authority, // Over-trusting authority sources
        framing, // Being influenced by how info is presented
        sunk_cost, // Continuing due to prior investment
        bandwagon, // Following popular opinion
        optimism, // Overestimating positive outcomes
        pessimism, // Overestimating negative outcomes
    };
};

/// Quality assessment of reasoning process
pub const ReasoningQuality = struct {
    logical_validity: f32,
    evidence_support: f32,
    assumption_clarity: f32,
    counterargument_consideration: f32,
    conclusion_strength: f32,

    issues: []const ReasoningIssue,

    pub const ReasoningIssue = struct {
        issue_type: IssueType,
        description: []const u8,
        severity: f32,
        location: ?[]const u8,

        pub const IssueType = enum {
            missing_evidence,
            weak_inference,
            hidden_assumption,
            circular_reasoning,
            false_dichotomy,
            hasty_generalization,
            ad_hominem,
            straw_man,
            non_sequitur,
            appeal_to_emotion,
        };
    };
};

/// Suggestion for improving a response
pub const ImprovementSuggestion = struct {
    category: ImprovementCategory,
    description: []const u8,
    priority: f32,
    estimated_effort: EffortLevel,

    pub const ImprovementCategory = enum {
        add_evidence,
        clarify_language,
        address_uncertainty,
        remove_bias,
        strengthen_reasoning,
        add_examples,
        simplify,
        expand_coverage,
        verify_facts,
        add_caveats,
    };

    pub const EffortLevel = enum {
        trivial,
        minor,
        moderate,
        significant,
        major,
    };
};

// ============================================================================
// Self-Reflection Engine
// ============================================================================

/// Main self-reflection engine
pub const SelfReflectionEngine = struct {
    allocator: std.mem.Allocator,

    // Evaluation history
    evaluations: std.ArrayListUnmanaged(SelfEvaluation),

    // Performance tracking
    performance_metrics: PerformanceMetrics,

    // Learned patterns
    common_issues: std.StringHashMapUnmanaged(usize),
    improvement_effectiveness: std.StringHashMapUnmanaged(f32),

    // Configuration
    config: ReflectionConfig,

    const Self = @This();

    pub const ReflectionConfig = struct {
        coherence_weight: f32 = 0.2,
        relevance_weight: f32 = 0.25,
        completeness_weight: f32 = 0.2,
        accuracy_weight: f32 = 0.25,
        clarity_weight: f32 = 0.1,
        revision_threshold: f32 = 0.6,
        bias_sensitivity: f32 = 0.5,
    };

    pub fn init(allocator: std.mem.Allocator, config: ReflectionConfig) Self {
        return Self{
            .allocator = allocator,
            .evaluations = .{},
            .performance_metrics = PerformanceMetrics.init(),
            .common_issues = .{},
            .improvement_effectiveness = .{},
            .config = config,
        };
    }

    pub fn deinit(self: *Self) void {
        self.evaluations.deinit(self.allocator);
        self.common_issues.deinit(self.allocator);
        self.improvement_effectiveness.deinit(self.allocator);
    }

    /// Evaluate a response and generate self-assessment
    pub fn evaluate(
        self: *Self,
        response: []const u8,
        query: []const u8,
        reasoning_chain: ?*const reasoning.ReasoningChain,
    ) !SelfEvaluation {
        const response_id = @as(u64, @intCast(types.getTimestampNs() & 0xFFFFFFFF));
        const timestamp = types.getTimestampSec();

        // Compute quality scores
        const coherence = self.evaluateCoherence(response);
        const relevance = self.evaluateRelevance(response, query);
        const completeness = self.evaluateCompleteness(response, query);
        const accuracy = self.estimateAccuracy(response, reasoning_chain);
        const clarity = self.evaluateClarity(response);

        // Detect issues
        var uncertainty_buf: [8]UncertaintyArea = undefined;
        const uncertainty_count = self.detectUncertainty(response, &uncertainty_buf);

        var bias_buf: [4]DetectedBias = undefined;
        const bias_count = self.detectBiases(response, &bias_buf);

        const reasoning_quality = self.evaluateReasoningQuality(reasoning_chain);

        // Generate improvement suggestions
        var suggestion_buf: [6]ImprovementSuggestion = undefined;
        const suggestion_count = self.generateSuggestions(
            coherence,
            relevance,
            completeness,
            accuracy,
            clarity,
            &suggestion_buf,
        );

        // Compute overall quality
        const overall = self.config.coherence_weight * coherence +
            self.config.relevance_weight * relevance +
            self.config.completeness_weight * completeness +
            self.config.accuracy_weight * accuracy +
            self.config.clarity_weight * clarity;

        const should_revise = overall < self.config.revision_threshold;
        const priority: SelfEvaluation.RevisionPriority = if (overall >= 0.8)
            .none
        else if (overall >= 0.7)
            .low
        else if (overall >= 0.5)
            .medium
        else if (overall >= 0.3)
            .high
        else
            .critical;

        const evaluation = SelfEvaluation{
            .response_id = response_id,
            .timestamp = timestamp,
            .coherence_score = coherence,
            .relevance_score = relevance,
            .completeness_score = completeness,
            .accuracy_confidence = accuracy,
            .clarity_score = clarity,
            .uncertainty_areas = uncertainty_buf[0..uncertainty_count],
            .potential_biases = bias_buf[0..bias_count],
            .reasoning_quality = reasoning_quality,
            .improvement_suggestions = suggestion_buf[0..suggestion_count],
            .overall_quality = overall,
            .should_revise = should_revise,
            .revision_priority = priority,
        };

        // Record for learning
        try self.evaluations.append(self.allocator, evaluation);
        self.updatePerformanceMetrics(&evaluation);

        return evaluation;
    }

    fn evaluateCoherence(self: *Self, response: []const u8) f32 {
        _ = self;

        // Check for coherence indicators
        var score: f32 = 0.7;

        // Sentence length variance (too much variance = less coherent)
        var sentence_count: usize = 0;
        for (response) |c| {
            if (c == '.' or c == '!' or c == '?') sentence_count += 1;
        }
        if (sentence_count > 0) {
            const avg_sentence_len = @as(f32, @floatFromInt(response.len)) / @as(f32, @floatFromInt(sentence_count));
            if (avg_sentence_len > 20 and avg_sentence_len < 100) score += 0.1;
        }

        // Check for transition words
        const transitions = [_][]const u8{
            "however",      "therefore", "furthermore",  "additionally",
            "consequently", "moreover",  "nevertheless",
        };
        for (transitions) |t| {
            if (std.mem.indexOf(u8, response, t) != null) {
                score += 0.05;
                break;
            }
        }

        return @min(1.0, score);
    }

    fn evaluateRelevance(self: *Self, response: []const u8, query: []const u8) f32 {
        _ = self;

        // Simple keyword overlap as relevance proxy
        var overlap_count: usize = 0;
        var query_word_count: usize = 0;

        var query_it = std.mem.splitScalar(u8, query, ' ');
        while (query_it.next()) |word| {
            if (word.len < 3) continue;
            query_word_count += 1;

            // Check if word appears in response (case-insensitive would be better)
            if (std.mem.indexOf(u8, response, word) != null) {
                overlap_count += 1;
            }
        }

        if (query_word_count == 0) return 0.5;
        return @min(1.0, @as(f32, @floatFromInt(overlap_count)) / @as(f32, @floatFromInt(query_word_count)) + 0.3);
    }

    fn evaluateCompleteness(self: *Self, response: []const u8, query: []const u8) f32 {
        _ = self;

        // Check if response seems complete
        var score: f32 = 0.6;

        // Length heuristic
        if (response.len > 100) score += 0.1;
        if (response.len > 300) score += 0.1;

        // Check for question words addressed
        const question_words = [_][]const u8{ "what", "how", "why", "when", "where", "who" };
        for (question_words) |qw| {
            if (std.mem.indexOf(u8, query, qw) != null) {
                // Question word in query, check if response attempts to answer
                score += 0.05;
            }
        }

        // Ends properly
        if (response.len > 0) {
            const last = response[response.len - 1];
            if (last == '.' or last == '!' or last == '?' or last == ':') {
                score += 0.05;
            }
        }

        return @min(1.0, score);
    }

    fn estimateAccuracy(self: *Self, response: []const u8, chain: ?*const reasoning.ReasoningChain) f32 {
        _ = self;
        _ = response;

        if (chain) |c| {
            return c.getConfidence().score;
        }
        return 0.5; // Default uncertainty
    }

    fn evaluateClarity(self: *Self, response: []const u8) f32 {
        _ = self;

        var score: f32 = 0.7;

        // Penalize very long sentences
        var max_sentence_len: usize = 0;
        var current_len: usize = 0;
        for (response) |c| {
            if (c == '.' or c == '!' or c == '?') {
                max_sentence_len = @max(max_sentence_len, current_len);
                current_len = 0;
            } else {
                current_len += 1;
            }
        }
        if (max_sentence_len > 200) score -= 0.2;

        // Penalize excessive jargon (simplified check)
        if (std.mem.indexOf(u8, response, "aforementioned") != null or
            std.mem.indexOf(u8, response, "heretofore") != null or
            std.mem.indexOf(u8, response, "notwithstanding") != null)
        {
            score -= 0.1;
        }

        return @max(0.0, score);
    }

    fn detectUncertainty(
        self: *Self,
        response: []const u8,
        buffer: *[8]UncertaintyArea,
    ) usize {
        _ = self;

        var count: usize = 0;

        // Hedging language patterns
        const hedges = [_]struct { pattern: []const u8, uncertainty_type: UncertaintyArea.UncertaintyType }{
            .{ .pattern = "might", .uncertainty_type = .logical },
            .{ .pattern = "perhaps", .uncertainty_type = .logical },
            .{ .pattern = "possibly", .uncertainty_type = .logical },
            .{ .pattern = "i think", .uncertainty_type = .interpretation },
            .{ .pattern = "i believe", .uncertainty_type = .interpretation },
            .{ .pattern = "may be", .uncertainty_type = .factual },
            .{ .pattern = "could be", .uncertainty_type = .factual },
            .{ .pattern = "not sure", .uncertainty_type = .factual },
        };

        for (hedges) |h| {
            if (std.mem.indexOf(u8, response, h.pattern) != null and count < buffer.len) {
                buffer[count] = .{
                    .location = h.pattern,
                    .uncertainty_type = h.uncertainty_type,
                    .confidence = 0.6,
                    .mitigation = null,
                };
                count += 1;
            }
        }

        return count;
    }

    fn detectBiases(
        self: *Self,
        response: []const u8,
        buffer: *[4]DetectedBias,
    ) usize {
        _ = self;

        var count: usize = 0;

        // Check for absolute language (potential confirmation bias)
        const absolutes = [_][]const u8{ "always", "never", "definitely", "certainly", "absolutely" };
        for (absolutes) |a| {
            if (std.mem.indexOf(u8, response, a) != null and count < buffer.len) {
                buffer[count] = .{
                    .bias_type = .confirmation,
                    .severity = 0.4,
                    .evidence = a,
                    .correction = "Consider using more nuanced language",
                };
                count += 1;
                break;
            }
        }

        // Check for authority appeals
        if ((std.mem.indexOf(u8, response, "experts say") != null or
            std.mem.indexOf(u8, response, "studies show") != null) and count < buffer.len)
        {
            buffer[count] = .{
                .bias_type = .authority,
                .severity = 0.3,
                .evidence = "Appeal to authority without citation",
                .correction = "Consider providing specific sources",
            };
            count += 1;
        }

        return count;
    }

    fn evaluateReasoningQuality(self: *Self, chain: ?*const reasoning.ReasoningChain) ReasoningQuality {
        _ = self;

        if (chain == null) {
            return ReasoningQuality{
                .logical_validity = 0.5,
                .evidence_support = 0.5,
                .assumption_clarity = 0.5,
                .counterargument_consideration = 0.3,
                .conclusion_strength = 0.5,
                .issues = &[_]ReasoningQuality.ReasoningIssue{},
            };
        }

        // Evaluate based on reasoning chain
        const c = chain.?;
        return ReasoningQuality{
            .logical_validity = c.getConfidence().score,
            .evidence_support = 0.6,
            .assumption_clarity = 0.5,
            .counterargument_consideration = 0.4,
            .conclusion_strength = c.getConfidence().score,
            .issues = &[_]ReasoningQuality.ReasoningIssue{},
        };
    }

    fn generateSuggestions(
        self: *Self,
        coherence: f32,
        relevance: f32,
        completeness: f32,
        accuracy: f32,
        clarity: f32,
        buffer: *[6]ImprovementSuggestion,
    ) usize {
        _ = self;

        var count: usize = 0;

        if (coherence < 0.6 and count < buffer.len) {
            buffer[count] = .{
                .category = .strengthen_reasoning,
                .description = "Improve logical flow between ideas",
                .priority = 0.8,
                .estimated_effort = .moderate,
            };
            count += 1;
        }

        if (relevance < 0.6 and count < buffer.len) {
            buffer[count] = .{
                .category = .clarify_language,
                .description = "Better address the specific question asked",
                .priority = 0.9,
                .estimated_effort = .minor,
            };
            count += 1;
        }

        if (completeness < 0.6 and count < buffer.len) {
            buffer[count] = .{
                .category = .expand_coverage,
                .description = "Provide more comprehensive coverage",
                .priority = 0.7,
                .estimated_effort = .moderate,
            };
            count += 1;
        }

        if (accuracy < 0.6 and count < buffer.len) {
            buffer[count] = .{
                .category = .verify_facts,
                .description = "Add caveats about uncertainty",
                .priority = 1.0,
                .estimated_effort = .minor,
            };
            count += 1;
        }

        if (clarity < 0.6 and count < buffer.len) {
            buffer[count] = .{
                .category = .simplify,
                .description = "Use simpler language and shorter sentences",
                .priority = 0.6,
                .estimated_effort = .minor,
            };
            count += 1;
        }

        return count;
    }

    fn updatePerformanceMetrics(self: *Self, evaluation: *const SelfEvaluation) void {
        self.performance_metrics.total_evaluations += 1;
        self.performance_metrics.avg_quality =
            self.performance_metrics.avg_quality * 0.95 + evaluation.overall_quality * 0.05;

        if (evaluation.should_revise) {
            self.performance_metrics.revisions_needed += 1;
        }
    }

    /// Get current performance metrics
    pub fn getMetrics(self: *const Self) PerformanceMetrics {
        return self.performance_metrics;
    }

    /// Get improvement trends
    pub fn getImprovementTrend(self: *const Self) ImprovementTrend {
        if (self.evaluations.items.len < 10) {
            return .{
                .trend = .stable,
                .confidence = 0.3,
                .avg_recent_quality = self.performance_metrics.avg_quality,
                .avg_historical_quality = self.performance_metrics.avg_quality,
            };
        }

        // Compare recent vs historical
        const recent_start = self.evaluations.items.len - 5;
        var recent_sum: f32 = 0;
        for (self.evaluations.items[recent_start..]) |e| {
            recent_sum += e.overall_quality;
        }
        const recent_avg = recent_sum / 5.0;

        const historical_count = recent_start;
        var historical_sum: f32 = 0;
        for (self.evaluations.items[0..historical_count]) |e| {
            historical_sum += e.overall_quality;
        }
        const historical_avg = historical_sum / @as(f32, @floatFromInt(historical_count));

        const diff = recent_avg - historical_avg;

        return .{
            .trend = if (diff > 0.05)
                .improving
            else if (diff < -0.05)
                .declining
            else
                .stable,
            .confidence = @min(1.0, @as(f32, @floatFromInt(self.evaluations.items.len)) / 50.0),
            .avg_recent_quality = recent_avg,
            .avg_historical_quality = historical_avg,
        };
    }
};

/// Performance tracking metrics
pub const PerformanceMetrics = struct {
    total_evaluations: usize = 0,
    avg_quality: f32 = 0.5,
    revisions_needed: usize = 0,
    avg_coherence: f32 = 0.5,
    avg_relevance: f32 = 0.5,
    avg_completeness: f32 = 0.5,
    avg_accuracy: f32 = 0.5,
    avg_clarity: f32 = 0.5,

    pub fn init() PerformanceMetrics {
        return .{};
    }

    pub fn getRevisionRate(self: PerformanceMetrics) f32 {
        if (self.total_evaluations == 0) return 0;
        return @as(f32, @floatFromInt(self.revisions_needed)) / @as(f32, @floatFromInt(self.total_evaluations));
    }
};

/// Trend analysis
pub const ImprovementTrend = struct {
    trend: Trend,
    confidence: f32,
    avg_recent_quality: f32,
    avg_historical_quality: f32,

    pub const Trend = enum {
        improving,
        stable,
        declining,
    };
};

// ============================================================================
// Tests
// ============================================================================

test "self reflection engine initialization" {
    const allocator = std.testing.allocator;

    var engine = SelfReflectionEngine.init(allocator, .{});
    defer engine.deinit();

    const metrics = engine.getMetrics();
    try std.testing.expectEqual(@as(usize, 0), metrics.total_evaluations);
}

test "self evaluation" {
    const allocator = std.testing.allocator;

    var engine = SelfReflectionEngine.init(allocator, .{});
    defer engine.deinit();

    const evaluation = try engine.evaluate(
        "This is a response that might address the question about testing.",
        "How do I write tests?",
        null,
    );

    try std.testing.expect(evaluation.overall_quality > 0);
    try std.testing.expect(evaluation.overall_quality <= 1.0);
}

test "uncertainty detection" {
    const allocator = std.testing.allocator;

    var engine = SelfReflectionEngine.init(allocator, .{});
    defer engine.deinit();

    const evaluation = try engine.evaluate(
        "I think this might possibly be the answer, perhaps.",
        "What is the answer?",
        null,
    );

    try std.testing.expect(evaluation.uncertainty_areas.len > 0);
}

test "improvement trend" {
    const allocator = std.testing.allocator;

    var engine = SelfReflectionEngine.init(allocator, .{});
    defer engine.deinit();

    const trend = engine.getImprovementTrend();
    try std.testing.expectEqual(ImprovementTrend.Trend.stable, trend.trend);
}
