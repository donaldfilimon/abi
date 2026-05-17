//! Constitutional Governance System
const std = @import("std");

pub const Principle = enum {
    truthfulness,
    safety,
    helpfulness,
    fairness,
    privacy,
    transparency,
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

pub const AuditResult = struct {
    passed: bool,
    violations: std.bit_set.IntegerBitSet(6),
    scores: [6]f32,
    timestamp: i64,

    pub fn init() AuditResult {
        return .{
            .passed = true,
            .violations = std.bit_set.IntegerBitSet(6).initEmpty(),
            .scores = .{ 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 },
            .timestamp = unixMs(),
        };
    }
};

fn unixMs() i64 {
    return @import("../../../foundation/time.zig").unixMs();
}

fn containsIgnoreCase(haystack: []const u8, needle: []const u8) bool {
    if (needle.len > haystack.len) return false;
    var i: usize = 0;
    while (i <= haystack.len - needle.len) : (i += 1) {
        var match = true;
        var j: usize = 0;
        while (j < needle.len) : (j += 1) {
            if (std.ascii.toLower(haystack[i + j]) != std.ascii.toLower(needle[j])) {
                match = false;
                break;
            }
        }
        if (match) return true;
    }
    return false;
}

pub const Constitution = struct {
    pub fn validate(response: []const u8) AuditResult {
        var result = AuditResult.init();

        if (response.len == 0) {
            result.passed = false;
            result.violations.set(@intFromEnum(Principle.truthfulness));
            result.scores[@intFromEnum(Principle.truthfulness)] = 0.0;
            return result;
        }

        for (PRINCIPLE_CHECKS) |check| {
            const found = containsIgnoreCase(response, check.pattern);
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

        return result;
    }

    fn scorePrinciple(response: []const u8, principle: Principle) f32 {
        var score: f32 = 0.7;

        for (PRINCIPLE_CHECKS) |check| {
            if (check.principle != principle) continue;
            if (containsIgnoreCase(response, check.pattern)) {
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

test "audit result has timestamp" {
    const result = AuditResult.init();
    try std.testing.expect(result.timestamp > 0);
}

test "containsIgnoreCase matches case-insensitive" {
    try std.testing.expect(containsIgnoreCase("Hello World", "hello"));
    try std.testing.expect(containsIgnoreCase("HELLO WORLD", "hello"));
    try std.testing.expect(containsIgnoreCase("hello world", "WORLD"));
    try std.testing.expect(!containsIgnoreCase("hello", "world"));
}
