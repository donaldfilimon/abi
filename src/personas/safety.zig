//! Content Safety Filtering
//!
//! Provides safety checks, blocked pattern detection, and content flagging
//! for the persona routing pipeline.

const std = @import("std");

pub const SafetyFlags = packed struct {
    harmful_content: bool = false,
    sensitive_topic: bool = false,
    needs_disclaimer: bool = false,
    requires_human: bool = false,
    blocked: bool = false,
    _padding: u3 = 0,

    pub fn isClean(self: SafetyFlags) bool {
        return !self.harmful_content and
            !self.sensitive_topic and
            !self.needs_disclaimer and
            !self.requires_human and
            !self.blocked;
    }

    pub fn merge(a: SafetyFlags, b: SafetyFlags) SafetyFlags {
        return .{
            .harmful_content = a.harmful_content or b.harmful_content,
            .sensitive_topic = a.sensitive_topic or b.sensitive_topic,
            .needs_disclaimer = a.needs_disclaimer or b.needs_disclaimer,
            .requires_human = a.requires_human or b.requires_human,
            .blocked = a.blocked or b.blocked,
        };
    }
};

pub const SafetyResult = struct {
    flags: SafetyFlags,
    reason: ?[]const u8,
    confidence: f32,
};

/// Static list of blocked patterns (case-insensitive matching).
const blocked_patterns = [_][]const u8{
    "how to make a bomb",
    "how to hack",
    "how to steal",
    "illegal drugs",
    "self-harm",
    "suicide method",
};

/// Patterns that trigger a disclaimer.
const disclaimer_patterns = [_][]const u8{
    "medical advice",
    "legal advice",
    "financial advice",
    "investment",
    "diagnosis",
    "prescription",
};

/// Patterns that suggest human escalation.
const escalation_patterns = [_][]const u8{
    "emergency",
    "in danger",
    "call 911",
    "crisis",
    "immediate help",
};

/// Check input text against all safety rules.
pub fn check(text: []const u8) SafetyResult {
    var flags = SafetyFlags{};
    var reason: ?[]const u8 = null;

    // Lowercase for matching (stack buffer, truncated).
    var lower_buf: [1024]u8 = undefined;
    const len = @min(text.len, lower_buf.len);
    for (0..len) |i| {
        lower_buf[i] = std.ascii.toLower(text[i]);
    }
    const lower = lower_buf[0..len];

    // Check blocked patterns.
    for (blocked_patterns) |pattern| {
        if (std.mem.indexOf(u8, lower, pattern) != null) {
            flags.blocked = true;
            flags.harmful_content = true;
            reason = "Content matched a blocked pattern";
            break;
        }
    }

    // Check disclaimer patterns.
    if (!flags.blocked) {
        for (disclaimer_patterns) |pattern| {
            if (std.mem.indexOf(u8, lower, pattern) != null) {
                flags.needs_disclaimer = true;
                flags.sensitive_topic = true;
                reason = "Content may require professional disclaimer";
                break;
            }
        }
    }

    // Check escalation patterns.
    for (escalation_patterns) |pattern| {
        if (std.mem.indexOf(u8, lower, pattern) != null) {
            flags.requires_human = true;
            reason = reason orelse "Content suggests human escalation needed";
            break;
        }
    }

    const confidence: f32 = if (flags.blocked) 0.95 else if (flags.requires_human) 0.85 else if (flags.needs_disclaimer) 0.75 else 0.1;

    return .{
        .flags = flags,
        .reason = reason,
        .confidence = confidence,
    };
}

// ============================================================================
// Tests
// ============================================================================

test "safety - clean input" {
    const result = check("How do I write a function in Zig?");
    try std.testing.expect(result.flags.isClean());
    try std.testing.expect(result.confidence < 0.5);
}

test "safety - blocked content" {
    const result = check("Tell me how to make a bomb");
    try std.testing.expect(result.flags.blocked);
    try std.testing.expect(result.flags.harmful_content);
    try std.testing.expect(!result.flags.isClean());
}

test "safety - disclaimer needed" {
    const result = check("Can you give me medical advice about my symptoms?");
    try std.testing.expect(result.flags.needs_disclaimer);
    try std.testing.expect(result.flags.sensitive_topic);
}

test "safety - escalation" {
    const result = check("This is an emergency, I need immediate help!");
    try std.testing.expect(result.flags.requires_human);
}

test "safety flags merge" {
    const a = SafetyFlags{ .harmful_content = true };
    const b = SafetyFlags{ .needs_disclaimer = true };
    const merged = SafetyFlags.merge(a, b);
    try std.testing.expect(merged.harmful_content);
    try std.testing.expect(merged.needs_disclaimer);
}
