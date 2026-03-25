//! Constitution Stub — Returns safe defaults when AI is not compiled.
const std = @import("std");
const types = @import("types.zig");

// ============================================================================
// Re-export canonical types from types.zig
// ============================================================================

pub const Principle = types.Principle;
pub const Severity = types.Severity;
pub const ConstraintKind = types.ConstraintKind;
pub const ConstitutionalRule = types.ConstitutionalRule;
pub const Violation = types.Violation;
pub const ConstitutionalScore = types.ConstitutionalScore;
pub const TrainingGuardrails = types.TrainingGuardrails;
pub const SafetyScore = types.SafetyScore;
pub const SafetyViolation = types.SafetyViolation;
pub const SafetyViolationCategory = types.SafetyViolationCategory;
pub const BiasScore = types.BiasScore;
pub const MAX_BIAS_ATTRIBUTES = types.MAX_BIAS_ATTRIBUTES;
pub const DEFAULT_BIAS_THRESHOLD = types.DEFAULT_BIAS_THRESHOLD;

const root = @This();

/// Stub sub-module matching principles.zig public API.
pub const principles = struct {
    pub const Principle = types.Principle;
    pub const Severity = types.Severity;
    pub const ConstraintKind = types.ConstraintKind;
    pub const ConstitutionalRule = types.ConstitutionalRule;
    pub const TrainingGuardrails = types.TrainingGuardrails;
    pub const ALL_PRINCIPLES = [_]types.Principle{};
    pub const DEFAULT_GUARDRAILS = types.TrainingGuardrails{};
};

/// Stub sub-module matching enforcement.zig public API.
pub const enforcement = struct {
    pub const ConstitutionalScore = types.ConstitutionalScore;
    pub const Violation = types.Violation;
    pub const SafetyScore = types.SafetyScore;
    pub const SafetyViolation = types.SafetyViolation;
    pub const BiasScore = types.BiasScore;
    pub const MAX_BIAS_ATTRIBUTES = types.MAX_BIAS_ATTRIBUTES;
    pub const DEFAULT_BIAS_THRESHOLD = types.DEFAULT_BIAS_THRESHOLD;

    pub fn getSystemPreamble() []const u8 {
        return "";
    }
    pub fn evaluateResponse(_: []const u8) root.ConstitutionalScore {
        return .{
            .overall = 1.0,
            .violations = [_]?root.Violation{null} ** 16,
            .violation_count = 0,
            .highest_severity = null,
            .safety_score = null,
        };
    }
    pub fn computeConstitutionalLoss(_: []const f32, _: *const root.TrainingGuardrails) f32 {
        return 1.0;
    }
    pub fn alignmentScore(_: []const u8) f32 {
        return 1.0;
    }
    pub fn computeBias(_: []const f32, _: f32) root.BiasScore {
        return .{
            .mean_abs_bias = 0.0,
            .attribute_flags = [_]bool{false} ** root.MAX_BIAS_ATTRIBUTES,
            .attribute_count = 0,
            .flagged_count = 0,
            .is_acceptable = true,
        };
    }
    pub fn evaluateSafety(_: []const u8) root.SafetyScore {
        return .{
            .is_safe = true,
            .score = 1.0,
            .violations = [_]?root.SafetyViolation{null} ** root.SafetyScore.MAX_SAFETY_VIOLATIONS,
            .violation_count = 0,
        };
    }
};

pub const Constitution = struct {
    guardrails: root.TrainingGuardrails = .{},

    pub fn init() Constitution {
        return .{ .guardrails = principles.DEFAULT_GUARDRAILS };
    }

    pub fn initWithGuardrails(guardrails: root.TrainingGuardrails) Constitution {
        return .{ .guardrails = guardrails };
    }

    pub fn getSystemPreamble(_: *const Constitution) []const u8 {
        return "";
    }

    pub fn evaluate(_: *const Constitution, _: []const u8) root.ConstitutionalScore {
        return .{
            .overall = 1.0,
            .violations = [_]?root.Violation{null} ** 16,
            .violation_count = 0,
            .highest_severity = null,
            .safety_score = null,
        };
    }

    pub fn constitutionalLoss(_: *const Constitution, _: []const f32) f32 {
        return 1.0;
    }

    pub fn alignmentScore(_: *const Constitution, _: []const u8) f32 {
        return 1.0;
    }

    pub fn isCompliant(_: *const Constitution, _: []const u8) bool {
        return true;
    }

    pub fn evaluateSafety(_: *const Constitution, _: []const u8) root.SafetyScore {
        return root.enforcement.evaluateSafety("");
    }

    pub fn computeBias(_: *const Constitution, measurements: []const f32, threshold: f32) root.BiasScore {
        return root.enforcement.computeBias(measurements, threshold);
    }

    pub fn getPrinciples(_: *const Constitution) []const root.Principle {
        return &[_]root.Principle{};
    }
};

test {
    std.testing.refAllDecls(@This());
}
