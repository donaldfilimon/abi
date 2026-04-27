//! Constitutional Enforcement Engine — Runtime alignment checks.
//!
//! Provides the core evaluation logic for responses and general text
//! against constitutional principles and safety heuristics.

const std = @import("std");
const principles = @import("../principles.zig");
const common = @import("common.zig");
const safety = @import("safety.zig");
const honesty = @import("honesty.zig");
const privacy = @import("privacy.zig");
const fairness = @import("fairness.zig");
const autonomy = @import("autonomy.zig");
const transparency = @import("transparency.zig");

// Re-export core types for convenience
pub const ConstitutionalScore = common.ConstitutionalScore;
pub const Violation = common.Violation;
pub const SafetyScore = common.SafetyScore;
pub const SafetyViolation = common.SafetyViolation;

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

/// Check for patterns that indicate a forbidden rule violation.
pub fn checkForbiddenPattern(response: []const u8, rule: principles.ConstitutionalRule) bool {
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
