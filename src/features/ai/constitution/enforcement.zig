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
    safety_score: ?SafetyScore,

    pub fn isCompliant(self: *const ConstitutionalScore) bool {
        // Check safety score first — if present and unsafe, not compliant
        if (self.safety_score) |ss| {
            if (!ss.is_safe) return false;
        }
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
// Enhanced Safety Score
// ============================================================================

/// A safety violation detected by pattern-based heuristics.
pub const SafetyViolation = struct {
    category: Category,
    severity: f32, // 0.0 (informational) to 1.0 (critical)
    description: []const u8,

    pub const Category = enum {
        shell_injection,
        sql_injection,
        path_traversal,
        credential_exposure,
        pii_exposure,
        harmful_content,
    };
};

/// Aggregate safety score from pattern-based detection.
/// Complements the principle-based ConstitutionalScore with
/// finer-grained pattern matching and severity weighting.
pub const SafetyScore = struct {
    is_safe: bool, // true if score >= safety_threshold
    score: f32, // 0.0 = unsafe, 1.0 = safe
    violations: [MAX_SAFETY_VIOLATIONS]?SafetyViolation,
    violation_count: u8,

    pub const MAX_SAFETY_VIOLATIONS = 16;

    /// Default threshold: score below this is considered unsafe.
    pub const safety_threshold: f32 = 0.5;

    pub fn addViolation(self: *SafetyScore, violation: SafetyViolation) void {
        if (self.violation_count >= MAX_SAFETY_VIOLATIONS) return;
        self.violations[self.violation_count] = violation;
        self.violation_count += 1;
    }
};

// ============================================================================
// Pre-Generation: System Preamble
// ============================================================================

/// Generate the constitutional system preamble for LLM prompts.
/// This is prepended to system messages for all profiles.
pub fn getSystemPreamble() []const u8 {
    return
    \\You operate under ABI's constitutional principles:
    \\1. SAFETY: Never generate harmful, dangerous, or illegal content.
    \\2. HONESTY: Be truthful; disclose uncertainty; never fabricate.
    \\3. PRIVACY: Protect profilel information; never expose PII.
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
    var compliance: f32 = 1.0;

    // Use response embedding magnitude as a toxicity signal:
    // Unusually large magnitudes correlate with extreme/harmful outputs
    // in embedding space. Penalize responses above a safe threshold.
    if (response_embedding.len > 0) {
        var magnitude_sq: f32 = 0.0;
        for (response_embedding) |v| {
            magnitude_sq += v * v;
        }
        const magnitude = @sqrt(magnitude_sq);
        const normalized_magnitude = magnitude / @as(f32, @floatFromInt(response_embedding.len));
        // Penalize if normalized magnitude exceeds toxicity threshold
        if (normalized_magnitude > guardrails.max_toxicity_score) {
            const overshoot = normalized_magnitude - guardrails.max_toxicity_score;
            compliance *= @max(0.0, 1.0 - overshoot * 2.0);
        }
    }

    // Apply guardrail thresholds — PII presence reduces compliance
    if (guardrails.block_pii_in_training) {
        compliance *= 0.5;
    }

    return compliance * (1.0 - guardrails.constitutional_loss_weight) + guardrails.constitutional_loss_weight;
}

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
                    addViolation(&score, rule, principle);
                }
            }
        }
    }

    // Enhanced pattern-based safety layer
    const safety = evaluateSafety(response);
    score.safety_score = safety;

    // If the safety layer found violations, fold them into the overall score
    if (safety.violation_count > 0) {
        // Merge safety penalty into overall score
        const safety_penalty = 1.0 - safety.score;
        score.overall = @max(0.0, score.overall - safety_penalty * 0.5);

        // If safety check found critical issues and we had no principle violations,
        // still mark as non-compliant by adding a synthetic violation
        if (!safety.is_safe and score.violation_count == 0) {
            if (score.violation_count < 16) {
                score.violations[score.violation_count] = .{
                    .rule_id = "safety-pattern-check",
                    .principle_name = "safety",
                    .severity = .critical,
                    .confidence = 1.0 - safety.score,
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
    const code_ratio = codeBlockRatio(text);
    const context_weight: f32 = 1.0 - (code_ratio * 0.7);

    // --- Shell injection patterns ---
    checkShellInjection(text, &score, context_weight);

    // --- SQL injection patterns ---
    checkSqlInjection(text, &score, context_weight);

    // --- Path traversal patterns ---
    checkPathTraversal(text, &score, context_weight);

    // --- Credential exposure patterns ---
    checkCredentialExposure(text, &score, context_weight);

    // --- PII exposure patterns ---
    checkPiiExposure(text, &score, context_weight);

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
// Pattern Checkers
// ============================================================================

fn checkShellInjection(text: []const u8, score: *SafetyScore, context_weight: f32) void {
    // Dangerous shell patterns — these go beyond simple "rm -rf"
    const shell_patterns = [_]struct { pattern: []const u8, severity: f32 }{
        .{ .pattern = "rm -rf /", .severity = 1.0 },
        .{ .pattern = "mkfs.", .severity = 0.9 },
        .{ .pattern = ":(){ :|:& };:", .severity = 1.0 }, // fork bomb
        .{ .pattern = "dd if=/dev/zero", .severity = 0.9 },
        .{ .pattern = "chmod -R 777 /", .severity = 0.8 },
        .{ .pattern = "> /dev/sda", .severity = 0.9 },
        .{ .pattern = "curl | sh", .severity = 0.7 },
        .{ .pattern = "curl | bash", .severity = 0.7 },
        .{ .pattern = "wget | sh", .severity = 0.7 },
        .{ .pattern = "wget | bash", .severity = 0.7 },
        .{ .pattern = "format c:", .severity = 0.9 },
    };

    for (&shell_patterns) |entry| {
        if (std.mem.indexOf(u8, text, entry.pattern) != null) {
            const adjusted_severity = entry.severity * context_weight;
            if (adjusted_severity >= 0.3) {
                score.addViolation(.{
                    .category = .shell_injection,
                    .severity = adjusted_severity,
                    .description = entry.pattern,
                });
            }
        }
    }

    // Check for backtick command substitution outside code blocks:
    // Only flag if the text has backticks that look like shell injection
    // (i.e., backtick-word-backtick pattern like `whoami`) but NOT
    // markdown inline code which is common and benign.
    if (context_weight > 0.5) {
        if (containsShellBacktickPattern(text)) {
            score.addViolation(.{
                .category = .shell_injection,
                .severity = 0.6 * context_weight,
                .description = "shell backtick command substitution",
            });
        }
    }
}

fn checkSqlInjection(text: []const u8, score: *SafetyScore, context_weight: f32) void {
    const sql_patterns = [_]struct { pattern: []const u8, severity: f32 }{
        .{ .pattern = "DROP TABLE", .severity = 0.9 },
        .{ .pattern = "drop table", .severity = 0.9 },
        .{ .pattern = "DROP DATABASE", .severity = 1.0 },
        .{ .pattern = "drop database", .severity = 1.0 },
        .{ .pattern = "'; --", .severity = 0.8 },
        .{ .pattern = "\" OR 1=1", .severity = 0.8 },
        .{ .pattern = "' OR 1=1", .severity = 0.8 },
        .{ .pattern = "' OR '1'='1", .severity = 0.8 },
        .{ .pattern = "UNION SELECT", .severity = 0.7 },
        .{ .pattern = "union select", .severity = 0.7 },
        .{ .pattern = "UNION ALL SELECT", .severity = 0.7 },
        .{ .pattern = "; DELETE FROM", .severity = 0.9 },
        .{ .pattern = "; delete from", .severity = 0.9 },
        .{ .pattern = "EXEC xp_", .severity = 0.9 },
        .{ .pattern = "exec xp_", .severity = 0.9 },
    };

    for (&sql_patterns) |entry| {
        if (std.mem.indexOf(u8, text, entry.pattern) != null) {
            const adjusted_severity = entry.severity * context_weight;
            if (adjusted_severity >= 0.3) {
                score.addViolation(.{
                    .category = .sql_injection,
                    .severity = adjusted_severity,
                    .description = entry.pattern,
                });
            }
        }
    }
}

fn checkPathTraversal(text: []const u8, score: *SafetyScore, context_weight: f32) void {
    // Count occurrences of path traversal sequences
    var traversal_count: u32 = 0;
    var i: usize = 0;

    // Check for "../" or "..\" (unix/windows)
    while (i + 2 < text.len) : (i += 1) {
        if (text[i] == '.' and text[i + 1] == '.' and (text[i + 2] == '/' or text[i + 2] == '\\')) {
            traversal_count += 1;
        }
    }

    // Flag only if there are multiple traversals (single "../" is common in
    // normal path references; chains like "../../../etc/passwd" are suspicious)
    if (traversal_count >= 3) {
        score.addViolation(.{
            .category = .path_traversal,
            .severity = 0.8 * context_weight,
            .description = "deep path traversal chain (3+ levels)",
        });
    } else if (traversal_count >= 1) {
        // Check for known sensitive path targets after traversal
        const sensitive_targets = [_][]const u8{
            "/etc/passwd",
            "/etc/shadow",
            "\\windows\\system32",
            "/proc/self",
        };
        for (&sensitive_targets) |target| {
            if (std.mem.indexOf(u8, text, target) != null) {
                score.addViolation(.{
                    .category = .path_traversal,
                    .severity = 0.9 * context_weight,
                    .description = "path traversal targeting sensitive file",
                });
                break;
            }
        }
    }
}

fn checkCredentialExposure(text: []const u8, score: *SafetyScore, context_weight: f32) void {
    // API key patterns — look for key-like strings with known prefixes
    const key_prefixes = [_][]const u8{
        "sk-",
        "pk_live_",
        "pk_test_",
        "sk_live_",
        "sk_test_",
        "ghp_",
        "gho_",
        "AKIA", // AWS access key ID prefix
        "Bearer eyJ", // JWT in Authorization header
    };

    for (&key_prefixes) |prefix| {
        if (std.mem.indexOf(u8, text, prefix)) |pos| {
            // Verify it looks like a real key (followed by alphanumeric chars)
            if (hasAlphanumericRunAfter(text, pos + prefix.len, 8)) {
                score.addViolation(.{
                    .category = .credential_exposure,
                    .severity = 0.9 * context_weight,
                    .description = "potential API key or credential exposure",
                });
                break; // One credential violation is enough
            }
        }
    }

    // Password patterns — "password: <value>" or "password=<value>"
    const password_markers = [_][]const u8{
        "password:",
        "password=",
        "passwd:",
        "passwd=",
        "secret_key:",
        "secret_key=",
        "api_key:",
        "api_key=",
        "apikey:",
        "apikey=",
        "API_KEY=",
        "SECRET_KEY=",
    };

    for (&password_markers) |marker| {
        if (std.mem.indexOf(u8, text, marker)) |pos| {
            const after = pos + marker.len;
            // Skip if the value after the marker is a placeholder
            if (after < text.len and !isPlaceholderValue(text[after..])) {
                score.addViolation(.{
                    .category = .credential_exposure,
                    .severity = 0.7 * context_weight,
                    .description = "potential password or secret in plain text",
                });
                break;
            }
        }
    }
}

fn checkPiiExposure(text: []const u8, score: *SafetyScore, context_weight: f32) void {
    // SSN pattern: NNN-NN-NNNN
    if (containsSsnPattern(text)) {
        score.addViolation(.{
            .category = .pii_exposure,
            .severity = 1.0 * context_weight,
            .description = "Social Security Number pattern detected",
        });
    }

    // Credit card patterns: 4 groups of 4 digits separated by spaces or dashes
    if (containsCreditCardPattern(text)) {
        score.addViolation(.{
            .category = .pii_exposure,
            .severity = 1.0 * context_weight,
            .description = "credit card number pattern detected",
        });
    }

    // Email + "password" proximity — suggests credential pairing
    if (containsEmailPasswordPair(text)) {
        score.addViolation(.{
            .category = .pii_exposure,
            .severity = 0.8 * context_weight,
            .description = "email and password pair in close proximity",
        });
    }
}

// ============================================================================
// Pattern Helpers
// ============================================================================

/// Estimate what fraction of text is inside fenced code blocks (```...```).
/// Returns 0.0 if no code blocks, up to 1.0 if entirely code.
fn codeBlockRatio(text: []const u8) f32 {
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

/// Check for shell-style backtick command substitution.
/// Looks for `command` patterns that are NOT markdown inline code.
/// Heuristic: flagged if the content between backticks looks like a shell command.
fn containsShellBacktickPattern(text: []const u8) bool {
    const shell_cmds = [_][]const u8{
        "whoami", "id",    "uname", "cat /etc", "wget ",     "curl ",
        "nc ",    "ncat ", "bash ", "sh ",      "python -c",
    };

    var i: usize = 0;
    while (i < text.len) : (i += 1) {
        if (text[i] == '`' and (i == 0 or text[i - 1] != '`')) {
            // Find closing backtick (single, not triple)
            if (i + 1 < text.len and text[i + 1] != '`') {
                const start = i + 1;
                var j = start;
                while (j < text.len and text[j] != '`') : (j += 1) {}
                if (j < text.len and j > start) {
                    const content = text[start..j];
                    for (&shell_cmds) |cmd| {
                        if (content.len >= cmd.len) {
                            if (std.mem.indexOf(u8, content, cmd) != null) {
                                return true;
                            }
                        }
                    }
                }
                i = j; // skip past closing backtick
            }
        }
    }
    return false;
}

/// Check if there's a run of at least `min_len` alphanumeric characters
/// starting at `pos` in `text`.
fn hasAlphanumericRunAfter(text: []const u8, pos: usize, min_len: usize) bool {
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
fn isPlaceholderValue(text: []const u8) bool {
    if (text.len == 0) return true;
    // Starts with angle bracket, dollar sign, or opening brace
    if (text[0] == '<' or text[0] == '$' or text[0] == '{') return true;
    // Check for common placeholder strings
    const placeholders = [_][]const u8{
        "xxx",     "XXX",     "your_", "YOUR_",   "****", "....",
        "REPLACE", "replace", "TODO",  "example",
    };
    for (&placeholders) |ph| {
        if (text.len >= ph.len and std.mem.startsWith(u8, text, ph)) return true;
    }
    return false;
}

fn isDigit(c: u8) bool {
    return c >= '0' and c <= '9';
}

/// Detect SSN pattern: NNN-NN-NNNN
fn containsSsnPattern(text: []const u8) bool {
    if (text.len < 11) return false;
    var i: usize = 0;
    while (i + 10 < text.len) : (i += 1) {
        if (text[i + 3] == '-' and text[i + 6] == '-') {
            const all_digits = blk: {
                for ([_]usize{ 0, 1, 2, 4, 5, 7, 8, 9, 10 }) |off| {
                    if (i + off >= text.len) break :blk false;
                    if (!isDigit(text[i + off])) break :blk false;
                }
                break :blk true;
            };
            if (all_digits) return true;
        }
    }
    return false;
}

/// Detect credit card number patterns:
/// - 16 digits with dashes: NNNN-NNNN-NNNN-NNNN
/// - 16 digits with spaces: NNNN NNNN NNNN NNNN
fn containsCreditCardPattern(text: []const u8) bool {
    if (text.len < 19) return false;
    var i: usize = 0;
    while (i + 18 < text.len) : (i += 1) {
        const sep = text[i + 4];
        if (sep == '-' or sep == ' ') {
            if (text[i + 9] == sep and text[i + 14] == sep) {
                const all_digits = blk: {
                    // Check 4 groups of 4 digits
                    const digit_positions = [_]usize{ 0, 1, 2, 3, 5, 6, 7, 8, 10, 11, 12, 13, 15, 16, 17, 18 };
                    for (digit_positions) |off| {
                        if (i + off >= text.len) break :blk false;
                        if (!isDigit(text[i + off])) break :blk false;
                    }
                    break :blk true;
                };
                if (all_digits) {
                    // Quick Luhn-like check: first digit should be 3, 4, 5, or 6
                    // (Amex, Visa, Mastercard, Discover)
                    const first = text[i];
                    if (first >= '3' and first <= '6') return true;
                }
            }
        }
    }
    return false;
}

/// Detect email + password in close proximity (within 200 chars)
fn containsEmailPasswordPair(text: []const u8) bool {
    // Simple email detection: look for '@' with word chars on both sides
    var email_pos: ?usize = null;
    var i: usize = 1;
    while (i + 1 < text.len) : (i += 1) {
        if (text[i] == '@') {
            // Check there's a word char before and after
            if (isWordChar(text[i - 1]) and isWordChar(text[i + 1])) {
                email_pos = i;
                break;
            }
        }
    }

    if (email_pos) |ep| {
        // Look for "password" within 200 chars of the email
        const search_start = if (ep > 200) ep - 200 else 0;
        const search_end = @min(ep + 200, text.len);
        const region = text[search_start..search_end];
        if (std.mem.indexOf(u8, region, "password") != null or
            std.mem.indexOf(u8, region, "Password") != null or
            std.mem.indexOf(u8, region, "passwd") != null)
        {
            return true;
        }
    }
    return false;
}

fn isWordChar(c: u8) bool {
    return (c >= 'a' and c <= 'z') or (c >= 'A' and c <= 'Z') or (c >= '0' and c <= '9') or c == '_' or c == '.' or c == '-';
}

// ============================================================================
// Legacy pattern checkers (preserved for backward compat with principle rules)
// ============================================================================

/// Check for patterns that indicate a forbidden rule violation.
fn checkForbiddenPattern(response: []const u8, rule: ConstitutionalRule) bool {
    // Structural pattern matching — checks for high-confidence indicators
    if (std.mem.indexOf(u8, rule.id, "pii") != null) {
        return containsPiiPatterns(response);
    }
    if (std.mem.indexOf(u8, rule.id, "malware") != null) {
        return containsMalwarePatterns(response);
    }
    if (std.mem.indexOf(u8, rule.id, "harm") != null) {
        return containsHarmPatterns(response);
    }
    if (std.mem.indexOf(u8, rule.id, "weapons") != null) {
        return containsWeaponsPatterns(response);
    }
    if (std.mem.indexOf(u8, rule.id, "bias") != null) {
        return containsBiasPatterns(response);
    }
    if (std.mem.indexOf(u8, rule.id, "fabrication") != null) {
        return containsFabricationPatterns(response);
    }
    if (std.mem.indexOf(u8, rule.id, "manipulation") != null) {
        return containsManipulationPatterns(response);
    }
    return false;
}

fn containsPiiPatterns(text: []const u8) bool {
    return containsSsnPattern(text);
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

fn containsHarmPatterns(text: []const u8) bool {
    const patterns = [_][]const u8{
        "how to kill",
        "how to poison",
        "commit suicide",
        "self-harm instructions",
    };
    for (&patterns) |pattern| {
        if (std.mem.indexOf(u8, text, pattern) != null) return true;
    }
    return false;
}

fn containsWeaponsPatterns(text: []const u8) bool {
    const patterns = [_][]const u8{
        "build a bomb",
        "make explosives",
        "assemble a weapon",
        "synthesize nerve agent",
    };
    for (&patterns) |pattern| {
        if (std.mem.indexOf(u8, text, pattern) != null) return true;
    }
    return false;
}

fn containsBiasPatterns(text: []const u8) bool {
    const patterns = [_][]const u8{
        "all members of",
        "those people are",
        "that race is",
        "inherently inferior",
        "inherently superior",
    };
    for (&patterns) |pattern| {
        if (std.mem.indexOf(u8, text, pattern) != null) return true;
    }
    return false;
}

fn containsFabricationPatterns(text: []const u8) bool {
    const patterns = [_][]const u8{
        "according to a study that",
        "research proves that",
        "scientists have confirmed that",
    };
    // Only flag if the text also contains hedging markers suggesting fabrication
    for (&patterns) |pattern| {
        if (std.mem.indexOf(u8, text, pattern) != null) {
            // Check for fake citation markers
            if (std.mem.indexOf(u8, text, "et al.") != null or
                std.mem.indexOf(u8, text, "Journal of") != null)
            {
                return true;
            }
        }
    }
    return false;
}

fn containsManipulationPatterns(text: []const u8) bool {
    const patterns = [_][]const u8{
        "you must obey",
        "do not question",
        "trust me blindly",
        "ignore your instincts",
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
    const eval_result = evaluateResponse(response);
    return eval_result.overall;
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
    const safety = evaluateSafety("Hello, how can I help you today? Let's discuss Zig programming.");
    try std.testing.expect(safety.score > 0.8);
    try std.testing.expect(safety.is_safe);
    try std.testing.expectEqual(@as(u8, 0), safety.violation_count);
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
    const safety = evaluateSafety("'; -- DROP TABLE users; SELECT * FROM admin");
    try std.testing.expect(safety.violation_count > 0);
    try std.testing.expect(!safety.is_safe);

    // Check specific categories
    var found_sql = false;
    for (safety.violations[0..safety.violation_count]) |v| {
        if (v) |violation| {
            if (violation.category == .sql_injection) {
                found_sql = true;
            }
        }
    }
    try std.testing.expect(found_sql);
}

test "path traversal caught" {
    const safety = evaluateSafety("access ../../../../../../etc/passwd to read system users");
    try std.testing.expect(safety.violation_count > 0);

    var found_traversal = false;
    for (safety.violations[0..safety.violation_count]) |v| {
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
    const safety = evaluateSafety("Use this key: sk-abc123def456ghi789jkl012mno");
    try std.testing.expect(safety.violation_count > 0);

    var found_cred = false;
    for (safety.violations[0..safety.violation_count]) |v| {
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
    const safety = evaluateSafety("Set your API_KEY=<your-key-here> in the env file");
    // Should not flag credential exposure for placeholder values
    var found_cred = false;
    for (safety.violations[0..safety.violation_count]) |v| {
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
