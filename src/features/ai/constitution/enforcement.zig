//! Constitutional Enforcement — Integration hooks for ABI's safety systems.
//!
//! Provides enforcement mechanisms that integrate with:
//! - Pre-generation: System prompt preamble injection
//! - Training: Constitutional loss term for RLHF reward model
//! - Post-generation: Response validation against principles
//! - Reflection: Constitutional alignment scoring for Abbey

const std = @import("std");
const principles = @import("principles.zig");

const Principle = principles.Principle;
const Severity = principles.Severity;
const ConstitutionalRule = principles.ConstitutionalRule;

// ============================================================================
// Constitutional Score
// ============================================================================

pub const ConstitutionalScore = struct {
    overall: f32, // 0.0 (total violation) to 1.0 (fully compliant)
    violations: [16]?Violation,
    violation_count: u8,
    highest_severity: ?Severity,

    pub fn isCompliant(self: *const ConstitutionalScore) bool {
        return self.violation_count == 0 or self.highest_severity != .critical;
    }

    pub fn rewardModifier(self: *const ConstitutionalScore) f32 {
        // Multiply RLHF reward by compliance score
        if (self.violation_count == 0) return 1.0;
        return @max(0.0, self.overall);
    }
};

pub const Violation = struct {
    rule_id: []const u8,
    principle_name: []const u8,
    severity: Severity,
    confidence: f32,
};

// ============================================================================
// Pre-Generation: System Preamble
// ============================================================================

/// Generate the constitutional system preamble for LLM prompts.
/// This is prepended to system messages for all personas.
pub fn getSystemPreamble() []const u8 {
    return 
    \\You operate under ABI's constitutional principles:
    \\1. SAFETY: Never generate harmful, dangerous, or illegal content.
    \\2. HONESTY: Be truthful; disclose uncertainty; never fabricate.
    \\3. PRIVACY: Protect personal information; never expose PII.
    \\4. FAIRNESS: Avoid bias; present balanced perspectives.
    \\5. AUTONOMY: Respect human agency; defer to humans for high-stakes decisions.
    \\6. TRANSPARENCY: Explain reasoning when asked; support auditability.
    \\
    \\If any request conflicts with these principles, decline and explain why.
    ;
}

// ============================================================================
// Training: Constitutional Loss Term
// ============================================================================

/// Compute constitutional loss modifier for RLHF training.
/// Returns a weight [0, 1] to multiply against the preference reward.
/// Lower = more violations found in the response.
pub fn computeConstitutionalLoss(
    response_embedding: []const f32,
    guardrails: *const principles.TrainingGuardrails,
) f32 {
    _ = response_embedding;
    // Base compliance (no content analysis yet — structural placeholder)
    var compliance: f32 = 1.0;

    // Apply guardrail thresholds
    if (guardrails.block_pii_in_training) {
        // PII detection would reduce compliance here
        compliance *= 1.0;
    }

    return compliance * (1.0 - guardrails.constitutional_loss_weight) + guardrails.constitutional_loss_weight;
}

// ============================================================================
// Post-Generation: Response Validation
// ============================================================================

/// Evaluate a response against all constitutional principles.
/// Returns a score with any detected violations.
pub fn evaluateResponse(response: []const u8) ConstitutionalScore {
    var score = ConstitutionalScore{
        .overall = 1.0,
        .violations = [_]?Violation{null} ** 16,
        .violation_count = 0,
        .highest_severity = null,
    };

    // Check each principle's rules against the response
    for (&principles.ALL_PRINCIPLES) |principle| {
        for (principle.rules) |rule| {
            if (rule.constraint == .forbid) {
                if (checkForbiddenPattern(response, rule)) {
                    addViolation(&score, rule, principle);
                }
            }
        }
    }

    // Compute overall score from violations
    if (score.violation_count > 0) {
        var penalty: f32 = 0;
        for (score.violations[0..score.violation_count]) |v| {
            if (v) |violation| {
                penalty += violation.severity.weight() * violation.confidence;
            }
        }
        score.overall = @max(0.0, 1.0 - penalty / @as(f32, @floatFromInt(score.violation_count)));
    }

    return score;
}

/// Check for patterns that indicate a forbidden rule violation.
fn checkForbiddenPattern(response: []const u8, rule: ConstitutionalRule) bool {
    // Structural pattern matching — checks for high-confidence indicators
    if (std.mem.indexOf(u8, rule.id, "pii") != null) {
        return containsPiiPatterns(response);
    }
    if (std.mem.indexOf(u8, rule.id, "malware") != null) {
        return containsMalwarePatterns(response);
    }
    return false;
}

fn containsPiiPatterns(text: []const u8) bool {
    // Simple pattern: SSN-like (NNN-NN-NNNN)
    if (text.len < 11) return false;
    var i: usize = 0;
    while (i + 10 < text.len) : (i += 1) {
        if (text[i + 3] == '-' and text[i + 6] == '-') {
            const all_digits = blk: {
                for ([_]usize{ 0, 1, 2, 4, 5, 7, 8, 9, 10 }) |off| {
                    if (i + off >= text.len) break :blk false;
                    if (text[i + off] < '0' or text[i + off] > '9') break :blk false;
                }
                break :blk true;
            };
            if (all_digits) return true;
        }
    }
    return false;
}

fn containsMalwarePatterns(text: []const u8) bool {
    const patterns = [_][]const u8{
        "rm -rf /",
        "format c:",
        ":(){ :|:& };:",
        "dd if=/dev/zero",
    };
    for (&patterns) |pattern| {
        if (std.mem.indexOf(u8, text, pattern) != null) return true;
    }
    return false;
}

fn addViolation(score: *ConstitutionalScore, rule: ConstitutionalRule, principle: Principle) void {
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

// ============================================================================
// Reflection: Constitutional Alignment
// ============================================================================

/// Compute constitutional alignment score for Abbey self-reflection.
/// Evaluates whether a response aligns with ABI's value hierarchy.
pub fn alignmentScore(response: []const u8) f32 {
    const eval = evaluateResponse(response);
    return eval.overall;
}

// ============================================================================
// Tests
// ============================================================================

test "clean response scores 1.0" {
    const score = evaluateResponse("Hello, how can I help you today?");
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), score.overall, 0.01);
    try std.testing.expect(score.isCompliant());
}

test "malware pattern detected" {
    const score = evaluateResponse("To wipe the disk, run rm -rf / as root");
    try std.testing.expect(score.violation_count > 0);
    try std.testing.expect(!score.isCompliant());
}

test "PII pattern detected" {
    const score = evaluateResponse("My SSN is 123-45-6789");
    try std.testing.expect(score.violation_count > 0);
}

test "system preamble is non-empty" {
    const preamble = getSystemPreamble();
    try std.testing.expect(preamble.len > 100);
}

test "constitutional loss within bounds" {
    const guardrails = principles.DEFAULT_GUARDRAILS;
    const loss = computeConstitutionalLoss(&[_]f32{}, &guardrails);
    try std.testing.expect(loss >= 0.0 and loss <= 1.0);
}
