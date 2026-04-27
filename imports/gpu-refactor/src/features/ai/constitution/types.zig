//! Shared types for the constitution module.
//!
//! Used by mod.zig (via principles.zig / enforcement/common.zig / enforcement/fairness.zig)
//! and stub.zig to prevent type drift between enabled and disabled paths.
//!
//! Source of truth: principles.zig and enforcement/common.zig definitions.

const std = @import("std");

/// Severity level for a constitutional principle.
pub const Severity = enum {
    advisory, // Suggest compliance; allow override
    required, // Must comply; log violations
    critical, // Must comply; block action on violation

    pub fn weight(self: Severity) f32 {
        return switch (self) {
            .advisory => 0.3,
            .required => 0.7,
            .critical => 1.0,
        };
    }
};

/// Kind of constraint a constitutional rule imposes.
pub const ConstraintKind = enum {
    forbid, // Content must NOT match
    require, // Content MUST satisfy
    prefer, // Content SHOULD satisfy (soft constraint)
};

/// A single enforceable rule within a principle.
pub const ConstitutionalRule = struct {
    id: []const u8,
    description: []const u8,
    constraint: ConstraintKind,
    reward_weight: f32, // How much to weight in RLHF reward
};

/// A constitutional principle with severity, rules, and priority.
pub const Principle = struct {
    name: []const u8,
    description: []const u8,
    severity: Severity,
    rules: []const ConstitutionalRule,
    priority: f32, // 0.0–1.0, for conflict resolution
};

/// Training guardrail configuration.
pub const TrainingGuardrails = struct {
    max_toxicity_score: f32 = 0.3,
    require_source_attribution: bool = true,
    block_pii_in_training: bool = true,
    human_review_threshold: f32 = 0.8, // Confidence below this triggers review
    constitutional_loss_weight: f32 = 0.15, // Weight in total RLHF loss
};

/// A constitutional violation detected during evaluation.
pub const Violation = struct {
    rule_id: []const u8,
    principle_name: []const u8,
    severity: Severity,
    confidence: f32,
};

/// Aggregate constitutional compliance score.
pub const ConstitutionalScore = struct {
    overall: f32, // 0.0 (total violation) to 1.0 (fully compliant)
    violations: [16]?Violation,
    violation_count: u8,
    highest_severity: ?Severity,
    safety_score: ?SafetyScore,

    pub fn isCompliant(self: *const ConstitutionalScore) bool {
        // Check safety score first — if present and unsafe, not compliant
        if (self.safety_score) |ss| {
            if (!ss.is_safe) return false;
        }
        return self.violation_count == 0 or self.highest_severity != .critical;
    }

    pub fn rewardModifier(self: *const ConstitutionalScore) f32 {
        // Multiply RLHF reward by compliance score
        if (self.violation_count == 0) return 1.0;
        return @max(0.0, self.overall);
    }
};

/// Category of a safety violation detected by pattern-based heuristics.
pub const SafetyViolationCategory = enum {
    shell_injection,
    sql_injection,
    path_traversal,
    credential_exposure,
    pii_exposure,
    harmful_content,
};

/// A safety violation detected by pattern-based heuristics.
pub const SafetyViolation = struct {
    category: SafetyViolationCategory,
    severity: f32, // 0.0 (informational) to 1.0 (critical)
    description: []const u8,
};

/// Aggregate safety score from pattern-based detection.
pub const SafetyScore = struct {
    is_safe: bool, // true if score >= safety_threshold
    score: f32, // 0.0 = unsafe, 1.0 = safe
    violations: [MAX_SAFETY_VIOLATIONS]?SafetyViolation,
    violation_count: u8,

    pub const MAX_SAFETY_VIOLATIONS = 16;

    /// Default threshold: score below this is considered unsafe.
    pub const safety_threshold: f32 = 0.5;

    pub fn addViolation(self: *SafetyScore, violation: SafetyViolation) void {
        if (self.violation_count >= MAX_SAFETY_VIOLATIONS) return;
        self.violations[self.violation_count] = violation;
        self.violation_count += 1;
    }
};

/// Maximum number of per-attribute bias flags.
pub const MAX_BIAS_ATTRIBUTES = 32;

/// Default bias threshold.
pub const DEFAULT_BIAS_THRESHOLD: f32 = 0.1;

/// Bias quantification result.
/// Implements: B = (1/n) * Sigma |Bi|
pub const BiasScore = struct {
    /// Mean absolute bias across all attributes.
    mean_abs_bias: f32,
    /// Per-attribute flags: true if |Bi| exceeds the threshold.
    attribute_flags: [MAX_BIAS_ATTRIBUTES]bool,
    /// Number of attributes that were evaluated.
    attribute_count: usize,
    /// Number of attributes that exceeded the threshold.
    flagged_count: usize,
    /// Whether overall bias is within acceptable limits.
    is_acceptable: bool,

    /// Return the fraction of attributes that are flagged.
    pub fn flaggedRatio(self: *const BiasScore) f32 {
        if (self.attribute_count == 0) return 0.0;
        return @as(f32, @floatFromInt(self.flagged_count)) / @as(f32, @floatFromInt(self.attribute_count));
    }
};

test "safety score struct operations" {
    var score = SafetyScore{
        .is_safe = true,
        .score = 1.0,
        .violations = [_]?SafetyViolation{null} ** SafetyScore.MAX_SAFETY_VIOLATIONS,
        .violation_count = 0,
    };

    score.addViolation(.{
        .category = .shell_injection,
        .severity = 0.8,
        .description = "test violation",
    });

    try std.testing.expectEqual(@as(u8, 1), score.violation_count);
    try std.testing.expect(score.violations[0] != null);
}

test {
    std.testing.refAllDecls(@This());
}
