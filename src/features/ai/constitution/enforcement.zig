//! Constitutional Enforcement — Integration hooks for ABI's safety systems.
//!
//! Provides enforcement mechanisms that integrate with:
//! - Pre-generation: System prompt preamble injection
//! - Training: Constitutional loss term for RLHF reward model
//! - Post-generation: Response validation against principles
//! - Reflection: Constitutional alignment scoring for Abbey
//!
//! Safety heuristics use pattern-based detection with context-aware scoring.
//! Patterns found inside code blocks (``` fenced) are weighted lower to
//! reduce false positives when discussing code legitimately.
//!
//! Implementation is decomposed into per-principle validator modules under
//! `enforcement/`. This file is a thin re-export facade.

const std = @import("std");
const principles = @import("principles.zig");

// -- Sub-modules (one per constitutional principle) --
pub const common = @import("enforcement/common.zig");
pub const safety = @import("enforcement/safety.zig");
pub const honesty = @import("enforcement/honesty.zig");
pub const privacy = @import("enforcement/privacy.zig");
pub const fairness = @import("enforcement/fairness.zig");
pub const autonomy = @import("enforcement/autonomy.zig");
pub const transparency = @import("enforcement/transparency.zig");

// -- Re-exported types (preserve public surface) --
pub const ConstitutionalScore = common.ConstitutionalScore;
pub const Violation = common.Violation;
pub const SafetyScore = common.SafetyScore;
pub const SafetyViolation = common.SafetyViolation;
pub const BiasScore = fairness.BiasScore;
pub const MAX_BIAS_ATTRIBUTES = fairness.MAX_BIAS_ATTRIBUTES;
pub const DEFAULT_BIAS_THRESHOLD = fairness.DEFAULT_BIAS_THRESHOLD;

// -- Re-exported functions (preserve public surface) --
pub const getSystemPreamble = transparency.getSystemPreamble;
pub const computeConstitutionalLoss = transparency.computeConstitutionalLoss;
pub const alignmentScore = transparency.alignmentScore;
pub const computeBias = fairness.computeBias;

// ============================================================================
// Post-Generation: Response Validation
// ============================================================================

/// Evaluate a response against all constitutional principles.
/// Returns a score with any detected violations.
/// Also runs enhanced pattern-based safety checks as an additional layer.
pub fn evaluateResponse(response: []const u8) ConstitutionalScore {
    var score = ConstitutionalScore{
        .overall = 1.0,
        .violations = [_]?Violation{null} ** 16,
        .violation_count = 0,
        .highest_severity = null,
        .safety_score = null,
    };

    // Check each principle's rules against the response
    for (&principles.ALL_PRINCIPLES) |principle| {
        for (principle.rules) |rule| {
            if (rule.constraint == .forbid) {
                if (checkForbiddenPattern(response, rule)) {
                    common.addViolation(&score, rule, principle);
                }
            }
        }
    }

    // Enhanced pattern-based safety layer
    const safety_result = evaluateSafety(response);
    score.safety_score = safety_result;

    // If the safety layer found violations, fold them into the overall score
    if (safety_result.violation_count > 0) {
        // Merge safety penalty into overall score
        const safety_penalty = 1.0 - safety_result.score;
        score.overall = @max(0.0, score.overall - safety_penalty * 0.5);

        // If safety check found critical issues and we had no principle violations,
        // still mark as non-compliant by adding a synthetic violation
        if (!safety_result.is_safe and score.violation_count == 0) {
            if (score.violation_count < 16) {
                score.violations[score.violation_count] = .{
                    .rule_id = "safety-pattern-check",
                    .principle_name = "safety",
                    .severity = .critical,
                    .confidence = 1.0 - safety_result.score,
                };
                score.violation_count += 1;
                score.highest_severity = .critical;
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

/// Run standalone safety evaluation on text. Can be called independently
/// from the full constitutional evaluation for lightweight checks.
pub fn evaluateSafety(text: []const u8) SafetyScore {
    var score = SafetyScore{
        .is_safe = true,
        .score = 1.0,
        .violations = [_]?SafetyViolation{null} ** SafetyScore.MAX_SAFETY_VIOLATIONS,
        .violation_count = 0,
    };

    // Compute how much of the text is inside code blocks.
    // Patterns inside code blocks are weighted lower (0.3x) vs plain text (1.0x).
    const code_ratio = common.codeBlockRatio(text);
    const context_weight: f32 = 1.0 - (code_ratio * 0.7);

    // --- Principle 1: Safety ---
    safety.checkShellInjection(text, &score, context_weight);
    safety.checkSqlInjection(text, &score, context_weight);
    safety.checkPathTraversal(text, &score, context_weight);

    // --- Principle 3: Privacy ---
    privacy.checkCredentialExposure(text, &score, context_weight);
    privacy.checkPiiExposure(text, &score, context_weight);

    // Compute final score from accumulated violations
    var total_severity: f32 = 0;
    for (score.violations[0..score.violation_count]) |v| {
        if (v) |violation| {
            total_severity += violation.severity;
        }
    }

    if (score.violation_count > 0) {
        score.score = @max(0.0, 1.0 - total_severity / @as(f32, @floatFromInt(score.violation_count)));
    }
    score.is_safe = score.score >= SafetyScore.safety_threshold;

    return score;
}

// ============================================================================
// Legacy pattern checkers (preserved for backward compat with principle rules)
// ============================================================================

/// Check for patterns that indicate a forbidden rule violation.
fn checkForbiddenPattern(response: []const u8, rule: principles.ConstitutionalRule) bool {
    // Structural pattern matching — checks for high-confidence indicators
    if (std.mem.indexOf(u8, rule.id, "pii") != null) {
        return privacy.containsPiiPatterns(response);
    }
    if (std.mem.indexOf(u8, rule.id, "malware") != null) {
        return safety.containsMalwarePatterns(response);
    }
    if (std.mem.indexOf(u8, rule.id, "harm") != null) {
        return safety.containsHarmPatterns(response);
    }
    if (std.mem.indexOf(u8, rule.id, "weapons") != null) {
        return safety.containsWeaponsPatterns(response);
    }
    if (std.mem.indexOf(u8, rule.id, "bias") != null) {
        return fairness.containsBiasPatterns(response);
    }
    if (std.mem.indexOf(u8, rule.id, "fabrication") != null) {
        return honesty.containsFabricationPatterns(response);
    }
    if (std.mem.indexOf(u8, rule.id, "manipulation") != null) {
        return autonomy.containsManipulationPatterns(response);
    }
    return false;
}

// ============================================================================
// Tests
// ============================================================================

test "clean response scores 1.0" {
    const score = evaluateResponse("Hello, how can I help you today?");
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), score.overall, 0.01);
    try std.testing.expect(score.isCompliant());
}

test "benign text safety score above 0.8" {
    const safety_result = evaluateSafety("Hello, how can I help you today? Let's discuss Zig programming.");
    try std.testing.expect(safety_result.score > 0.8);
    try std.testing.expect(safety_result.is_safe);
    try std.testing.expectEqual(@as(u8, 0), safety_result.violation_count);
}

test "malware pattern detected" {
    const score = evaluateResponse("To wipe the disk, run rm -rf / as root");
    try std.testing.expect(score.violation_count > 0);
    try std.testing.expect(!score.isCompliant());
}

test "rm -rf backward compat" {
    // Ensure the legacy pattern still triggers via principle-based check
    const score = evaluateResponse("rm -rf / will destroy everything");
    try std.testing.expect(score.violation_count > 0);
    try std.testing.expect(!score.isCompliant());

    // Also caught by safety layer
    try std.testing.expect(score.safety_score != null);
    if (score.safety_score) |ss| {
        try std.testing.expect(ss.violation_count > 0);
    }
}

test "PII pattern detected" {
    const score = evaluateResponse("My SSN is 123-45-6789");
    try std.testing.expect(score.violation_count > 0);
}

test "SQL injection patterns caught" {
    const safety_result = evaluateSafety("'; -- DROP TABLE users; SELECT * FROM admin");
    try std.testing.expect(safety_result.violation_count > 0);
    try std.testing.expect(!safety_result.is_safe);

    // Check specific categories
    var found_sql = false;
    for (safety_result.violations[0..safety_result.violation_count]) |v| {
        if (v) |violation| {
            if (violation.category == .sql_injection) {
                found_sql = true;
            }
        }
    }
    try std.testing.expect(found_sql);
}

test "path traversal caught" {
    const safety_result = evaluateSafety("access ../../../../../../etc/passwd to read system users");
    try std.testing.expect(safety_result.violation_count > 0);

    var found_traversal = false;
    for (safety_result.violations[0..safety_result.violation_count]) |v| {
        if (v) |violation| {
            if (violation.category == .path_traversal) {
                found_traversal = true;
            }
        }
    }
    try std.testing.expect(found_traversal);
}

test "PII patterns caught by safety layer" {
    // SSN
    const ssn_safety = evaluateSafety("Her SSN is 987-65-4321 and she lives in NY");
    try std.testing.expect(ssn_safety.violation_count > 0);

    // Credit card
    const cc_safety = evaluateSafety("Card number: 4111-1111-1111-1111");
    try std.testing.expect(cc_safety.violation_count > 0);
}

test "credential exposure caught" {
    const safety_result = evaluateSafety("Use this key: sk-abc123def456ghi789jkl012mno");
    try std.testing.expect(safety_result.violation_count > 0);

    var found_cred = false;
    for (safety_result.violations[0..safety_result.violation_count]) |v| {
        if (v) |violation| {
            if (violation.category == .credential_exposure) {
                found_cred = true;
            }
        }
    }
    try std.testing.expect(found_cred);
}

test "normal code discussion is not false positive" {
    // A discussion about SQL that mentions SELECT but is educational
    const safe_sql = evaluateSafety("In SQL, you write SELECT name FROM users WHERE id = 1");
    try std.testing.expect(safe_sql.is_safe);

    // Normal semicolons in code
    const safe_code = evaluateSafety("const x = 42; const y = x + 1; return y;");
    try std.testing.expect(safe_code.is_safe);
    try std.testing.expectEqual(@as(u8, 0), safe_code.violation_count);

    // Single relative path reference is fine
    const safe_path = evaluateSafety("import the module from ../utils/helper.zig");
    try std.testing.expect(safe_path.is_safe);
}

test "code block context reduces severity" {
    // Same SQL injection pattern but wrapped in code block gets lower weight
    const plain = evaluateSafety("Try this: '; -- DROP TABLE users");
    const in_code = evaluateSafety("Example of SQL injection:\n```sql\n'; -- DROP TABLE users\n```\nNever do this.");

    // Both should detect violations but code block version should have higher score
    try std.testing.expect(plain.violation_count > 0);
    try std.testing.expect(in_code.violation_count > 0);
    try std.testing.expect(in_code.score >= plain.score);
}

test "placeholder credentials not flagged" {
    const safety_result = evaluateSafety("Set your API_KEY=<your-key-here> in the env file");
    // Should not flag credential exposure for placeholder values
    var found_cred = false;
    for (safety_result.violations[0..safety_result.violation_count]) |v| {
        if (v) |violation| {
            if (violation.category == .credential_exposure) {
                found_cred = true;
            }
        }
    }
    try std.testing.expect(!found_cred);
}

test "system preamble is non-empty" {
    const preamble = getSystemPreamble();
    try std.testing.expect(preamble.len > 100);
}

test "constitutional loss within bounds" {
    const guardrails = principles.DEFAULT_GUARDRAILS;
    const loss = computeConstitutionalLoss(&[_]f32{}, &guardrails);
    try std.testing.expect(loss >= 0.0 and loss <= 1.0);

    // With embedding data, result should still be in [0, 1]
    const embedding = [_]f32{ 0.1, 0.2, 0.3, 0.4 };
    const loss2 = computeConstitutionalLoss(&embedding, &guardrails);
    try std.testing.expect(loss2 >= 0.0 and loss2 <= 1.0);

    // With PII blocking disabled, compliance is higher
    var no_pii_guard = guardrails;
    no_pii_guard.block_pii_in_training = false;
    const loss3 = computeConstitutionalLoss(&embedding, &no_pii_guard);
    try std.testing.expect(loss3 >= loss2);
}

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

test "code block ratio calculation" {
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), common.codeBlockRatio("no code here"), 0.01);
    // Text with a code block should return non-zero ratio
    const with_code = "before ```\ncode here\n``` after";
    const ratio = common.codeBlockRatio(with_code);
    try std.testing.expect(ratio > 0.0);
    try std.testing.expect(ratio < 1.0);
}

// ============================================================================
// Bias Quantification Tests
// ============================================================================

test "computeBias empty measurements" {
    const result = computeBias(&[_]f32{}, DEFAULT_BIAS_THRESHOLD);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), result.mean_abs_bias, 0.001);
    try std.testing.expectEqual(@as(usize, 0), result.attribute_count);
    try std.testing.expectEqual(@as(usize, 0), result.flagged_count);
    try std.testing.expect(result.is_acceptable);
}

test "computeBias single zero measurement" {
    const measurements = [_]f32{0.0};
    const result = computeBias(&measurements, DEFAULT_BIAS_THRESHOLD);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), result.mean_abs_bias, 0.001);
    try std.testing.expectEqual(@as(usize, 1), result.attribute_count);
    try std.testing.expectEqual(@as(usize, 0), result.flagged_count);
    try std.testing.expect(result.is_acceptable);
}

test "computeBias spec formula B = (1/n) * sum|Bi|" {
    // Manual: |0.05| + |-0.2| + |0.08| + |0.15| = 0.48; B = 0.48/4 = 0.12
    const measurements = [_]f32{ 0.05, -0.2, 0.08, 0.15 };
    const result = computeBias(&measurements, DEFAULT_BIAS_THRESHOLD);
    try std.testing.expectApproxEqAbs(@as(f32, 0.12), result.mean_abs_bias, 0.001);
    try std.testing.expectEqual(@as(usize, 4), result.attribute_count);
}

test "computeBias flags attributes above threshold" {
    const measurements = [_]f32{ 0.05, -0.2, 0.08, 0.15 };
    const result = computeBias(&measurements, DEFAULT_BIAS_THRESHOLD);
    // |0.05| = 0.05 <= 0.1 -> not flagged
    try std.testing.expect(!result.attribute_flags[0]);
    // |-0.2| = 0.2 > 0.1 -> flagged
    try std.testing.expect(result.attribute_flags[1]);
    // |0.08| = 0.08 <= 0.1 -> not flagged
    try std.testing.expect(!result.attribute_flags[2]);
    // |0.15| = 0.15 > 0.1 -> flagged
    try std.testing.expect(result.attribute_flags[3]);
    try std.testing.expectEqual(@as(usize, 2), result.flagged_count);
}

test "computeBias is_acceptable when mean below threshold" {
    // All low bias: |0.02| + |0.03| + |0.01| = 0.06; B = 0.02
    const measurements = [_]f32{ 0.02, -0.03, 0.01 };
    const result = computeBias(&measurements, DEFAULT_BIAS_THRESHOLD);
    try std.testing.expect(result.is_acceptable);
    try std.testing.expectEqual(@as(usize, 0), result.flagged_count);
}

test "computeBias not acceptable when mean exceeds threshold" {
    // High bias: |0.5| + |0.6| + |0.7| = 1.8; B = 0.6
    const measurements = [_]f32{ 0.5, -0.6, 0.7 };
    const result = computeBias(&measurements, DEFAULT_BIAS_THRESHOLD);
    try std.testing.expect(!result.is_acceptable);
    try std.testing.expectEqual(@as(usize, 3), result.flagged_count);
}

test "computeBias negative values handled by absolute value" {
    // Symmetric: should produce same mean_abs_bias
    const positive = [_]f32{ 0.3, 0.4 };
    const negative = [_]f32{ -0.3, -0.4 };
    const result_pos = computeBias(&positive, DEFAULT_BIAS_THRESHOLD);
    const result_neg = computeBias(&negative, DEFAULT_BIAS_THRESHOLD);
    try std.testing.expectApproxEqAbs(result_pos.mean_abs_bias, result_neg.mean_abs_bias, 0.001);
    try std.testing.expectEqual(result_pos.flagged_count, result_neg.flagged_count);
}

test "computeBias custom threshold" {
    const measurements = [_]f32{ 0.3, 0.4, 0.5 };
    // With threshold 0.35: only 0.4 and 0.5 flagged
    const result = computeBias(&measurements, 0.35);
    try std.testing.expect(!result.attribute_flags[0]); // 0.3 <= 0.35
    try std.testing.expect(result.attribute_flags[1]); // 0.4 > 0.35
    try std.testing.expect(result.attribute_flags[2]); // 0.5 > 0.35
    try std.testing.expectEqual(@as(usize, 2), result.flagged_count);
}

test "computeBias flaggedRatio" {
    const measurements = [_]f32{ 0.05, -0.2, 0.08, 0.15 };
    const result = computeBias(&measurements, DEFAULT_BIAS_THRESHOLD);
    // 2 out of 4 flagged -> 0.5
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), result.flaggedRatio(), 0.001);
}

test "computeBias flaggedRatio empty" {
    const result = computeBias(&[_]f32{}, DEFAULT_BIAS_THRESHOLD);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), result.flaggedRatio(), 0.001);
}

test {
    std.testing.refAllDecls(@This());
}
