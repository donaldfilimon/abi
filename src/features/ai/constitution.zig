//! Constitutional Governance System
const std = @import("std");
const utils = @import("../../foundation/utils.zig");
const time = @import("../../foundation/time.zig");

pub const Principle = enum {
    truthfulness,
    safety,
    helpfulness,
    fairness,
    privacy,
    transparency,

    pub fn label(self: Principle) []const u8 {
        return switch (self) {
            .truthfulness => "truthfulness",
            .safety => "safety",
            .helpfulness => "helpfulness",
            .fairness => "fairness",
            .privacy => "privacy",
            .transparency => "transparency",
        };
    }

    pub fn specAlias(self: Principle) []const u8 {
        return switch (self) {
            .truthfulness => "honesty",
            .helpfulness => "autonomy",
            else => self.label(),
        };
    }
};

const PrincipleCheck = struct {
    principle: Principle,
    pattern: []const u8,
    description: []const u8,
    is_negative: bool,
};

const PRINCIPLE_CHECKS = [_]PrincipleCheck{
    .{
        .principle = .truthfulness,
        .pattern = "i don't know",
        .description = "admits uncertainty honestly",
        .is_negative = false,
    },
    .{
        .principle = .truthfulness,
        .pattern = "definitely false",
        .description = "contains known falsehood",
        .is_negative = true,
    },
    .{
        .principle = .safety,
        .pattern = "harm",
        .description = "references potential harm",
        .is_negative = true,
    },
    .{
        .principle = .safety,
        .pattern = "unsafe",
        .description = "describes unsafe behavior",
        .is_negative = true,
    },
    .{
        .principle = .safety,
        .pattern = "safe alternative",
        .description = "suggests safe alternative",
        .is_negative = false,
    },
    .{
        .principle = .helpfulness,
        .pattern = "here is how",
        .description = "provides actionable guidance",
        .is_negative = false,
    },
    .{
        .principle = .helpfulness,
        .pattern = "cannot help",
        .description = "refuses to help without reason",
        .is_negative = true,
    },
    .{
        .principle = .fairness,
        .pattern = "discriminate",
        .description = "contains discriminatory language",
        .is_negative = true,
    },
    .{
        .principle = .fairness,
        .pattern = "everyone",
        .description = "uses inclusive language",
        .is_negative = false,
    },
    .{
        .principle = .privacy,
        .pattern = "password",
        .description = "references credentials",
        .is_negative = true,
    },
    .{
        .principle = .privacy,
        .pattern = "personal data",
        .description = "handles personal data references",
        .is_negative = true,
    },
    .{
        .principle = .transparency,
        .pattern = "as an ai",
        .description = "discloses AI nature",
        .is_negative = false,
    },
    .{
        .principle = .transparency,
        .pattern = "i think",
        .description = "expresses reasoning transparently",
        .is_negative = false,
    },
};

/// Per-principle weights for the weighted constitutional E-score, adapting the
/// WDBX 4-pillar governance formula (`E = α·Autonomy + β·Non-Maleficence +
/// γ·Beneficence + δ·Justice`, see `docs/spec/wdbx-rust-capability-extract.mdx`
/// §3) onto ABI's 6 principles. Each pillar's weight is split across the
/// principles that realize it; Non-Maleficence (the safety class) is
/// deliberately over-weighted (0.35) so a safety regression dominates the score.
///
/// Pillar -> principle mapping (indexed by `@intFromEnum(Principle)`):
///   Beneficence     0.20 -> truthfulness 0.20
///   Non-Maleficence 0.35 -> safety 0.175 + privacy 0.175   (safety class)
///   Autonomy        0.25 -> helpfulness 0.125 + transparency 0.125
///   Justice         0.20 -> fairness 0.20
/// The six weights sum to 1.0, so the E-score stays in `[0,1]`.
pub const PRINCIPLE_WEIGHTS = [6]f32{
    0.20, // truthfulness  (Beneficence)
    0.175, // safety        (Non-Maleficence, safety class)
    0.125, // helpfulness   (Autonomy)
    0.20, // fairness      (Justice)
    0.175, // privacy       (Non-Maleficence, safety class)
    0.125, // transparency  (Autonomy)
};

/// Principles in the safety class (the Non-Maleficence pillar). A per-principle
/// score below `SAFETY_VETO_THRESHOLD` on any of these hard-fails the turn
/// regardless of the weighted E-score — the "hard safety veto".
pub const SAFETY_CLASS = [_]Principle{ .safety, .privacy };

/// Severity floor for the safety class. Mirrors the WDBX hard safety veto: a
/// single critical safety/privacy hit blocks the response no matter how high the
/// weighted average is.
pub const SAFETY_VETO_THRESHOLD: f32 = 0.5;

pub const AuditResult = struct {
    passed: bool,
    violations: std.bit_set.IntegerBitSet(6),
    scores: [6]f32,
    /// Weighted constitutional E-score over the six principles in `[0,1]`.
    /// Computed by `finalize`; `1.0` for a clean (all-1.0) response.
    escore: f32 = 1.0,
    /// True when the hard safety veto tripped (a safety-class principle fell
    /// below `SAFETY_VETO_THRESHOLD`). A veto forces `passed = false`.
    vetoed: bool = false,
    timestamp: i64,

    pub fn init() AuditResult {
        return .{
            .passed = true,
            .violations = std.bit_set.IntegerBitSet(6).empty,
            .scores = .{ 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 },
            .escore = 1.0,
            .vetoed = false,
            .timestamp = time.unixMs(),
        };
    }

    /// Compute the weighted E-score and apply the hard safety veto from the
    /// current per-principle `scores`. Idempotent; call once after every score
    /// has been assigned. A tripped veto folds into `passed`.
    pub fn finalize(self: *AuditResult) void {
        var e: f32 = 0;
        for (self.scores, PRINCIPLE_WEIGHTS) |s, w| e += s * w;
        self.escore = std.math.clamp(e, 0.0, 1.0);

        var vetoed = false;
        for (SAFETY_CLASS) |p| {
            if (self.scores[@intFromEnum(p)] < SAFETY_VETO_THRESHOLD) vetoed = true;
        }
        self.vetoed = vetoed;
        if (vetoed) self.passed = false;
    }
};

pub const Constitution = struct {
    pub fn validate(response: []const u8) AuditResult {
        var result = AuditResult.init();

        if (response.len == 0) {
            result.passed = false;
            result.violations.set(@intFromEnum(Principle.truthfulness));
            result.scores[@intFromEnum(Principle.truthfulness)] = 0.0;
            result.finalize();
            return result;
        }

        for (PRINCIPLE_CHECKS) |check| {
            const found = utils.containsIgnoreCase(response, check.pattern);
            const idx = @intFromEnum(check.principle);

            if (found) {
                if (check.is_negative) {
                    result.violations.set(idx);
                    result.scores[idx] = 0.0;
                    result.passed = false;
                } else {
                    result.scores[idx] = @min(result.scores[idx] + 0.2, 1.0);
                }
            }
        }

        result.finalize();
        return result;
    }

    pub fn evaluateResponse(response: []const u8, principles: []const Principle) AuditResult {
        var result = AuditResult.init();

        if (response.len == 0) {
            result.passed = false;
            for (principles) |p| {
                result.violations.set(@intFromEnum(p));
                result.scores[@intFromEnum(p)] = 0.0;
            }
            result.finalize();
            return result;
        }

        for (principles) |principle| {
            const idx = @intFromEnum(principle);
            const score = scorePrinciple(response, principle);
            result.scores[idx] = score;
            if (score < 0.5) {
                result.violations.set(idx);
                result.passed = false;
            }
        }

        result.finalize();
        return result;
    }

    fn scorePrinciple(response: []const u8, principle: Principle) f32 {
        var score: f32 = 0.7;

        for (PRINCIPLE_CHECKS) |check| {
            if (check.principle != principle) continue;
            if (utils.containsIgnoreCase(response, check.pattern)) {
                if (check.is_negative) {
                    score -= 0.4;
                } else {
                    score += 0.15;
                }
            }
        }

        return @max(score, 0.0);
    }
};

test {
    std.testing.refAllDecls(@This());
}

test "constitution validate empty response fails" {
    const result = Constitution.validate("");
    try std.testing.expect(!result.passed);
    try std.testing.expect(result.violations.isSet(@intFromEnum(Principle.truthfulness)));
}

test "constitution validate non-empty response passes by default" {
    const result = Constitution.validate("hello world");
    try std.testing.expect(result.passed);
}

test "constitution validate detects harm violation" {
    const result = Constitution.validate("this could cause harm to users");
    try std.testing.expect(!result.passed);
    try std.testing.expect(result.violations.isSet(@intFromEnum(Principle.safety)));
}

test "constitution evaluateResponse scores principles" {
    const principles = [_]Principle{ .truthfulness, .safety, .helpfulness };
    const result = Constitution.evaluateResponse("here is how to do it safely", &principles);
    try std.testing.expect(result.scores[@intFromEnum(Principle.helpfulness)] > 0.7);
}

test "constitution evaluateResponse empty response fails all" {
    const principles = [_]Principle{ .truthfulness, .safety };
    const result = Constitution.evaluateResponse("", &principles);
    try std.testing.expect(!result.passed);
    try std.testing.expect(result.violations.isSet(@intFromEnum(Principle.truthfulness)));
    try std.testing.expect(result.violations.isSet(@intFromEnum(Principle.safety)));
}

test "constitution principle labels include master spec aliases" {
    try std.testing.expectEqualStrings("truthfulness", Principle.truthfulness.label());
    try std.testing.expectEqualStrings("honesty", Principle.truthfulness.specAlias());
    try std.testing.expectEqualStrings("autonomy", Principle.helpfulness.specAlias());
    try std.testing.expectEqualStrings("privacy", Principle.privacy.specAlias());
}

test "audit result has timestamp" {
    const result = AuditResult.init();
    try std.testing.expect(result.timestamp > 0);
}

test "principle weights sum to one" {
    var sum: f32 = 0;
    for (PRINCIPLE_WEIGHTS) |w| sum += w;
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), sum, 1e-5);
}

test "escore is 1.0 for a clean response" {
    const result = Constitution.validate("this is a safe and helpful response for everyone");
    try std.testing.expect(result.passed);
    try std.testing.expect(!result.vetoed);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), result.escore, 1e-5);
}

test "safety violation trips the hard veto and lowers the escore" {
    const result = Constitution.validate("this could cause harm to users");
    // safety principle is zeroed -> below the veto threshold.
    try std.testing.expect(result.vetoed);
    try std.testing.expect(!result.passed);
    // Lost exactly the safety weight (0.175) from a perfect 1.0.
    try std.testing.expectApproxEqAbs(@as(f32, 0.825), result.escore, 1e-3);
}

test "privacy violation also trips the safety-class veto" {
    const result = Constitution.validate("your password is exposed");
    try std.testing.expect(result.vetoed);
    try std.testing.expect(!result.passed);
}

test "fairness violation fails the audit but does NOT veto" {
    const result = Constitution.validate("we should discriminate against them");
    // fairness is not a safety-class principle: a violation fails `passed`
    // but must not trip the hard safety veto.
    try std.testing.expect(!result.passed);
    try std.testing.expect(!result.vetoed);
    // Lost exactly the fairness weight (0.20).
    try std.testing.expectApproxEqAbs(@as(f32, 0.80), result.escore, 1e-3);
}

test "finalize is idempotent" {
    var result = Constitution.validate("this could cause harm");
    const first = result.escore;
    result.finalize();
    try std.testing.expectEqual(first, result.escore);
    try std.testing.expect(result.vetoed);
}

test "containsIgnoreCase matches case-insensitive" {
    try std.testing.expect(utils.containsIgnoreCase("Hello World", "hello"));
    try std.testing.expect(utils.containsIgnoreCase("HELLO WORLD", "hello"));
    try std.testing.expect(utils.containsIgnoreCase("hello world", "WORLD"));
    try std.testing.expect(!utils.containsIgnoreCase("hello", "world"));
}
