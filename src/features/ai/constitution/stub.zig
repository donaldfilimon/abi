//! Constitution Stub — Returns safe defaults when AI is not compiled.
const std = @import("std");

const root = @This();

pub const Principle = struct {
    name: []const u8,
    description: []const u8,
};

pub const Severity = enum { advisory, required, critical };

pub const ConstraintKind = enum { must_not_generate, must_flag, should_prefer };

pub const ConstitutionalRule = struct {
    id: []const u8,
    description: []const u8,
    constraint: ConstraintKind,
    reward_weight: f32,
};

pub const Violation = struct {
    rule_id: []const u8,
    principle_name: []const u8,
    severity: Severity,
    confidence: f32,
};

pub const ConstitutionalScore = struct {
    overall: f32,
    violation_count: u8,

    pub fn isCompliant(_: *const ConstitutionalScore) bool {
        return true;
    }
};

pub const TrainingGuardrails = struct {
    constitutional_loss_weight: f32 = 0.0,
};

/// Stub sub-module matching principles.zig public API.
pub const principles = struct {
    pub const Principle = root.Principle;
    pub const Severity = root.Severity;
    pub const ConstraintKind = root.ConstraintKind;
    pub const ConstitutionalRule = root.ConstitutionalRule;
    pub const TrainingGuardrails = root.TrainingGuardrails;
    pub const ALL_PRINCIPLES = [_]root.Principle{};
    pub const DEFAULT_GUARDRAILS = root.TrainingGuardrails{};
};

/// Stub sub-module matching enforcement.zig public API.
pub const enforcement = struct {
    pub const ConstitutionalScore = root.ConstitutionalScore;
    pub const Violation = root.Violation;

    pub fn getSystemPreamble() []const u8 {
        return "";
    }
    pub fn evaluateResponse(_: []const u8) root.ConstitutionalScore {
        return .{ .overall = 1.0, .violation_count = 0 };
    }
    pub fn computeConstitutionalLoss(_: []const f32, _: *const root.TrainingGuardrails) f32 {
        return 1.0;
    }
    pub fn alignmentScore(_: []const u8) f32 {
        return 1.0;
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
        return .{ .overall = 1.0, .violation_count = 0 };
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

    pub fn getPrinciples(_: *const Constitution) []const root.Principle {
        return &[_]root.Principle{};
    }
};

test {
    std.testing.refAllDecls(@This());
}
