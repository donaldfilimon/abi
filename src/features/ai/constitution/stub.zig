//! Constitution Stub — Returns FeatureDisabled when AI is not compiled.

pub const Principle = struct {
    name: []const u8,
    description: []const u8,
};

pub const Severity = enum { advisory, required, critical };

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

pub const Constitution = struct {
    pub fn init() Constitution {
        return .{};
    }

    pub fn getSystemPreamble(_: *const Constitution) []const u8 {
        return "";
    }

    pub fn evaluate(_: *const Constitution, _: []const u8) ConstitutionalScore {
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

    pub fn getPrinciples(_: *const Constitution) []const Principle {
        return &[_]Principle{};
    }
};
