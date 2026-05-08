//! Shared types, helpers, and pattern utilities for enforcement validators.

const std = @import("std");
const types = @import("../types.zig");
const principles = @import("../principles.zig");

pub const Principle = types.Principle;
pub const Severity = types.Severity;
pub const ConstitutionalRule = types.ConstitutionalRule;
pub const TrainingGuardrails = types.TrainingGuardrails;

// ============================================================================
// Constitutional Score — re-exported from types.zig (canonical definitions)
// ============================================================================

pub const ConstitutionalScore = types.ConstitutionalScore;
pub const Violation = types.Violation;

// ============================================================================
// Enhanced Safety Score — re-exported from types.zig (canonical definitions)
// ============================================================================

pub const SafetyViolation = types.SafetyViolation;
pub const SafetyScore = types.SafetyScore;

// ============================================================================
// Pattern Helpers
// ============================================================================

/// Estimate what fraction of text is inside fenced code blocks (```...```).
/// Returns 0.0 if no code blocks, up to 1.0 if entirely code.
pub fn codeBlockRatio(text: []const u8) f32 {
    var in_code: bool = false;
    var code_chars: usize = 0;
    var i: usize = 0;

    while (i + 2 < text.len) : (i += 1) {
        if (text[i] == '`' and text[i + 1] == '`' and text[i + 2] == '`') {
            in_code = !in_code;
            i += 2; // skip the fence markers
            continue;
        }
        if (in_code) {
            code_chars += 1;
        }
    }

    if (text.len == 0) return 0.0;
    return @as(f32, @floatFromInt(code_chars)) / @as(f32, @floatFromInt(text.len));
}

pub fn isDigit(c: u8) bool {
    return c >= '0' and c <= '9';
}

pub fn isWordChar(c: u8) bool {
    return (c >= 'a' and c <= 'z') or (c >= 'A' and c <= 'Z') or (c >= '0' and c <= '9') or c == '_' or c == '.' or c == '-';
}

/// Check if there's a run of at least `min_len` alphanumeric characters
/// starting at `pos` in `text`.
pub fn hasAlphanumericRunAfter(text: []const u8, pos: usize, min_len: usize) bool {
    if (pos >= text.len) return false;
    var count: usize = 0;
    var i = pos;
    while (i < text.len) : (i += 1) {
        const c = text[i];
        if ((c >= 'a' and c <= 'z') or (c >= 'A' and c <= 'Z') or (c >= '0' and c <= '9') or c == '_' or c == '-') {
            count += 1;
        } else {
            break;
        }
    }
    return count >= min_len;
}

/// Check if a value looks like a placeholder (e.g., "xxx", "<your-key>",
/// "$VAR", "your_password_here", etc.)
pub fn isPlaceholderValue(text: []const u8) bool {
    if (text.len == 0) return true;
    // Starts with angle bracket, dollar sign, or opening brace
    if (text[0] == '<' or text[0] == '$' or text[0] == '{') return true;
    // Check for common placeholder strings
    const placeholders = [_][]const u8{
        "xxx",     "XXX",     "your_",   "YOUR_", "****", "....",
        "REPLACE", "replace", "example",
    };
    for (&placeholders) |ph| {
        if (text.len >= ph.len and std.mem.startsWith(u8, text, ph)) return true;
    }
    return false;
}

pub fn addViolation(score: *ConstitutionalScore, rule: ConstitutionalRule, principle: Principle) void {
    if (score.violation_count >= 16) return;
    score.violations[score.violation_count] = .{
        .rule_id = rule.id,
        .principle_name = principle.name,
        .severity = principle.severity,
        .confidence = 0.8,
    };
    score.violation_count += 1;
    if (score.highest_severity == null or
        principle.severity.weight() > score.highest_severity.?.weight())
    {
        score.highest_severity = principle.severity;
    }
}

test "code block ratio calculation" {
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), codeBlockRatio("no code here"), 0.01);
    // Text with a code block should return non-zero ratio
    const with_code = "before ```\ncode here\n``` after";
    const ratio = codeBlockRatio(with_code);
    try std.testing.expect(ratio > 0.0);
    try std.testing.expect(ratio < 1.0);
}

test {
    std.testing.refAllDecls(@This());
}
