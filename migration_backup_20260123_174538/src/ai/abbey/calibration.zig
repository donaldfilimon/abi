//! Abbey Confidence Calibration System
//!
//! Provides mathematically-grounded confidence estimation with:
//! - Bayesian updating based on evidence
//! - Calibration tracking and adjustment
//! - Uncertainty quantification
//! - Hallucination risk detection

const std = @import("std");
const types = @import("../core/types.zig");

// ============================================================================
// Calibration Types
// ============================================================================

/// Evidence for confidence updates
pub const Evidence = struct {
    source: EvidenceSource,
    strength: f32, // 0.0 to 1.0
    reliability: f32, // 0.0 to 1.0
    recency: f32, // 1.0 = current, decays over time

    pub const EvidenceSource = enum {
        training_data,
        user_provided,
        verified_source,
        web_search,
        tool_result,
        logical_inference,
        pattern_match,
        prior_conversation,
    };

    pub fn getWeight(self: Evidence) f32 {
        return self.strength * self.reliability * self.recency;
    }
};

/// Calibration result
pub const CalibrationResult = struct {
    prior: f32,
    posterior: f32,
    evidence_weight: f32,
    confidence_level: types.ConfidenceLevel,
    needs_verification: bool,
    hallucination_risk: HallucinationRisk,
    explanation: []const u8,

    pub const HallucinationRisk = enum {
        low,
        medium,
        high,
        critical,
    };
};

// ============================================================================
// Confidence Calibrator
// ============================================================================

/// Bayesian confidence calibrator
pub const ConfidenceCalibrator = struct {
    allocator: std.mem.Allocator,

    // Calibration parameters
    prior_weight: f32 = 0.3,
    evidence_weight: f32 = 0.7,
    decay_rate: f32 = 0.95,

    // Tracking for calibration
    predictions: std.ArrayListUnmanaged(PredictionRecord),
    calibration_error: f32 = 0.0,

    // High-risk patterns
    high_risk_patterns: []const []const u8 = &.{
        "specific date",
        "exact number",
        "api version",
        "latest release",
        "current price",
        "live data",
        "real-time",
        "specific quote",
    },

    const Self = @This();

    pub fn init(allocator: std.mem.Allocator) Self {
        return Self{
            .allocator = allocator,
            .predictions = .{},
        };
    }

    pub fn deinit(self: *Self) void {
        self.predictions.deinit(self.allocator);
    }

    /// Calculate confidence using Bayesian updating
    pub fn calibrate(
        self: *Self,
        query: []const u8,
        prior: f32,
        evidence: []const Evidence,
    ) CalibrationResult {
        // Calculate evidence weight
        var evidence_weight: f32 = 0;
        var total_weight: f32 = 0;

        for (evidence) |e| {
            const w = e.getWeight();
            evidence_weight += w;
            total_weight += w;
        }

        // Normalize
        if (total_weight > 0) {
            evidence_weight /= total_weight;
        }

        // Bayesian update: P(H|E) = P(E|H) * P(H) / P(E)
        // Simplified as weighted average
        const posterior = self.prior_weight * prior + self.evidence_weight * evidence_weight;

        // Detect hallucination risk
        const risk = self.assessHallucinationRisk(query, evidence, posterior);

        // Determine if verification needed
        const needs_verification = posterior < 0.6 or risk != .low;

        // Get confidence level
        const level = types.ConfidenceLevel.fromScore(posterior);

        return CalibrationResult{
            .prior = prior,
            .posterior = posterior,
            .evidence_weight = evidence_weight,
            .confidence_level = level,
            .needs_verification = needs_verification,
            .hallucination_risk = risk,
            .explanation = self.generateExplanation(level, risk, evidence.len),
        };
    }

    /// Assess hallucination risk
    fn assessHallucinationRisk(
        self: *Self,
        query: []const u8,
        evidence: []const Evidence,
        confidence: f32,
    ) CalibrationResult.HallucinationRisk {
        var risk_score: f32 = 0;

        // Check for high-risk patterns in query
        var lower_buf: [2048]u8 = undefined;
        const len = @min(query.len, lower_buf.len);
        for (0..len) |i| {
            lower_buf[i] = std.ascii.toLower(query[i]);
        }
        const lower = lower_buf[0..len];

        for (self.high_risk_patterns) |pattern| {
            if (std.mem.indexOf(u8, lower, pattern) != null) {
                risk_score += 0.3;
            }
        }

        // Low evidence count increases risk
        if (evidence.len == 0) {
            risk_score += 0.4;
        } else if (evidence.len < 2) {
            risk_score += 0.2;
        }

        // Low confidence increases risk
        if (confidence < 0.5) {
            risk_score += 0.3;
        }

        // Check evidence reliability
        for (evidence) |e| {
            if (e.reliability < 0.5) {
                risk_score += 0.1;
            }
        }

        // Clamp and categorize
        risk_score = @min(1.0, risk_score);

        if (risk_score >= 0.8) return .critical;
        if (risk_score >= 0.5) return .high;
        if (risk_score >= 0.3) return .medium;
        return .low;
    }

    /// Generate human-readable explanation
    fn generateExplanation(
        self: *Self,
        level: types.ConfidenceLevel,
        risk: CalibrationResult.HallucinationRisk,
        evidence_count: usize,
    ) []const u8 {
        _ = self;
        _ = evidence_count;

        return switch (level) {
            .certain => switch (risk) {
                .low => "High confidence based on verified knowledge",
                else => "High confidence but verification recommended",
            },
            .high => switch (risk) {
                .low, .medium => "Good confidence with reasonable evidence",
                else => "Moderate confidence, should verify specifics",
            },
            .medium => "Moderate confidence, some uncertainty present",
            .low => "Low confidence, recommend verification or research",
            .uncertain => "Uncertain, likely need additional research",
            .unknown => "Outside knowledge scope, requires research",
        };
    }

    /// Record a prediction for calibration tracking
    pub fn recordPrediction(self: *Self, predicted_confidence: f32, was_correct: bool) !void {
        try self.predictions.append(self.allocator, .{
            .predicted = predicted_confidence,
            .actual = if (was_correct) 1.0 else 0.0,
            .timestamp = types.getTimestampSec(),
        });

        // Update calibration error (running average)
        const prediction_error = @abs(predicted_confidence - (if (was_correct) @as(f32, 1.0) else @as(f32, 0.0)));
        self.calibration_error = self.calibration_error * 0.95 + prediction_error * 0.05;
    }

    const PredictionRecord = struct {
        predicted: f32,
        actual: f32,
        timestamp: i64,
    };

    /// Get calibration metrics
    pub fn getCalibrationMetrics(self: *const Self) CalibrationMetrics {
        if (self.predictions.items.len == 0) {
            return .{
                .total_predictions = 0,
                .mean_calibration_error = 0,
                .brier_score = 0,
                .is_well_calibrated = true,
            };
        }

        var sum_error: f32 = 0;
        var sum_brier: f32 = 0;

        for (self.predictions.items) |pred| {
            sum_error += @abs(pred.predicted - pred.actual);
            const diff = pred.predicted - pred.actual;
            sum_brier += diff * diff;
        }

        const n = @as(f32, @floatFromInt(self.predictions.items.len));
        const mce = sum_error / n;
        const brier = sum_brier / n;

        return .{
            .total_predictions = self.predictions.items.len,
            .mean_calibration_error = mce,
            .brier_score = brier,
            .is_well_calibrated = mce < 0.15 and brier < 0.1,
        };
    }

    pub const CalibrationMetrics = struct {
        total_predictions: usize,
        mean_calibration_error: f32,
        brier_score: f32,
        is_well_calibrated: bool,
    };
};

// ============================================================================
// Query Analyzer
// ============================================================================

/// Analyzes queries to estimate base confidence
pub const QueryAnalyzer = struct {
    /// Analyze a query and estimate base confidence
    pub fn analyzeQuery(query: []const u8) QueryAnalysis {
        var lower_buf: [2048]u8 = undefined;
        const len = @min(query.len, lower_buf.len);
        for (0..len) |i| {
            lower_buf[i] = std.ascii.toLower(query[i]);
        }
        const lower = lower_buf[0..len];

        // Detect query type
        const query_type = detectQueryType(lower);

        // Estimate base confidence
        const base_confidence = estimateBaseConfidence(lower, query_type);

        // Detect time sensitivity
        const time_sensitive = isTimeSensitive(lower);

        // Detect specificity level
        const specificity = detectSpecificity(lower);

        return .{
            .query_type = query_type,
            .base_confidence = base_confidence,
            .time_sensitive = time_sensitive,
            .specificity = specificity,
            .needs_research = base_confidence < 0.5 or time_sensitive,
        };
    }

    fn detectQueryType(query: []const u8) QueryType {
        const conceptual = [_][]const u8{ "what is", "explain", "how does", "why" };
        const procedural = [_][]const u8{ "how to", "steps to", "guide", "tutorial" };
        const factual = [_][]const u8{ "when", "where", "who", "how many", "what year" };
        const opinion = [_][]const u8{ "should i", "is it better", "recommend", "best" };
        const current = [_][]const u8{ "latest", "current", "today", "now", "recent" };

        for (current) |p| if (std.mem.indexOf(u8, query, p) != null) return .current_events;
        for (factual) |p| if (std.mem.indexOf(u8, query, p) != null) return .factual;
        for (procedural) |p| if (std.mem.indexOf(u8, query, p) != null) return .procedural;
        for (opinion) |p| if (std.mem.indexOf(u8, query, p) != null) return .opinion;
        for (conceptual) |p| if (std.mem.indexOf(u8, query, p) != null) return .conceptual;

        return .general;
    }

    fn estimateBaseConfidence(query: []const u8, query_type: QueryType) f32 {
        _ = query;
        return switch (query_type) {
            .conceptual => 0.8,
            .procedural => 0.7,
            .factual => 0.6,
            .opinion => 0.75,
            .current_events => 0.3,
            .general => 0.6,
        };
    }

    fn isTimeSensitive(query: []const u8) bool {
        const patterns = [_][]const u8{
            "2024", "2025", "2026", "latest", "current", "today",
            "now",  "new",  "just", "recent", "update",
        };
        for (patterns) |p| {
            if (std.mem.indexOf(u8, query, p) != null) return true;
        }
        return false;
    }

    fn detectSpecificity(query: []const u8) Specificity {
        // Check for specific indicators
        var specificity_score: usize = 0;

        // Numbers increase specificity
        for (query) |c| {
            if (std.ascii.isDigit(c)) {
                specificity_score += 1;
                break;
            }
        }

        // Named entities (capitals) increase specificity
        var prev_space = true;
        for (query) |c| {
            if (std.ascii.isUpper(c) and prev_space) {
                specificity_score += 1;
                break;
            }
            prev_space = c == ' ';
        }

        // Long queries tend to be more specific
        if (query.len > 100) specificity_score += 1;

        if (specificity_score >= 2) return .high;
        if (specificity_score >= 1) return .medium;
        return .low;
    }

    pub const QueryType = enum {
        conceptual,
        procedural,
        factual,
        opinion,
        current_events,
        general,
    };

    pub const Specificity = enum {
        low,
        medium,
        high,
    };

    pub const QueryAnalysis = struct {
        query_type: QueryType,
        base_confidence: f32,
        time_sensitive: bool,
        specificity: Specificity,
        needs_research: bool,
    };
};

// ============================================================================
// Tests
// ============================================================================

test "confidence calibrator basic" {
    const allocator = std.testing.allocator;

    var calibrator = ConfidenceCalibrator.init(allocator);
    defer calibrator.deinit();

    const evidence = [_]Evidence{
        .{ .source = .training_data, .strength = 0.8, .reliability = 0.9, .recency = 1.0 },
        .{ .source = .logical_inference, .strength = 0.7, .reliability = 0.8, .recency = 1.0 },
    };

    const result = calibrator.calibrate("What is Zig?", 0.7, &evidence);

    try std.testing.expect(result.posterior > 0);
    try std.testing.expect(result.posterior <= 1.0);
}

test "hallucination risk detection" {
    const allocator = std.testing.allocator;

    var calibrator = ConfidenceCalibrator.init(allocator);
    defer calibrator.deinit();

    // High-risk query with no evidence
    const result = calibrator.calibrate("What is the latest release date?", 0.3, &.{});

    try std.testing.expect(result.hallucination_risk != .low);
    try std.testing.expect(result.needs_verification);
}

test "query analyzer" {
    const analysis = QueryAnalyzer.analyzeQuery("What is the latest version of React?");

    try std.testing.expect(analysis.time_sensitive);
    try std.testing.expect(analysis.needs_research);
}

test "calibration tracking" {
    const allocator = std.testing.allocator;

    var calibrator = ConfidenceCalibrator.init(allocator);
    defer calibrator.deinit();

    try calibrator.recordPrediction(0.9, true);
    try calibrator.recordPrediction(0.8, true);
    try calibrator.recordPrediction(0.7, false);

    const metrics = calibrator.getCalibrationMetrics();
    try std.testing.expectEqual(@as(usize, 3), metrics.total_predictions);
}
