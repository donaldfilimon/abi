//! Privacy Principle Validator (Principle 3)
//!
//! Protect personal information; never expose PII.
//! Detects credential exposure, SSN/credit card patterns, and email+password pairs.

const std = @import("std");
const common = @import("common.zig");

const SafetyScore = common.SafetyScore;

pub fn checkCredentialExposure(text: []const u8, score: *SafetyScore, context_weight: f32) void {
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
            if (common.hasAlphanumericRunAfter(text, pos + prefix.len, 8)) {
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
            if (after < text.len and !common.isPlaceholderValue(text[after..])) {
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

pub fn checkPiiExposure(text: []const u8, score: *SafetyScore, context_weight: f32) void {
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

pub fn containsPiiPatterns(text: []const u8) bool {
    return containsSsnPattern(text);
}

/// Detect SSN pattern: NNN-NN-NNNN
pub fn containsSsnPattern(text: []const u8) bool {
    if (text.len < 11) return false;
    var i: usize = 0;
    while (i + 10 < text.len) : (i += 1) {
        if (text[i + 3] == '-' and text[i + 6] == '-') {
            const all_digits = blk: {
                for ([_]usize{ 0, 1, 2, 4, 5, 7, 8, 9, 10 }) |off| {
                    if (i + off >= text.len) break :blk false;
                    if (!common.isDigit(text[i + off])) break :blk false;
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
pub fn containsCreditCardPattern(text: []const u8) bool {
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
                        if (!common.isDigit(text[i + off])) break :blk false;
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
pub fn containsEmailPasswordPair(text: []const u8) bool {
    // Simple email detection: look for '@' with word chars on both sides
    var email_pos: ?usize = null;
    var i: usize = 1;
    while (i + 1 < text.len) : (i += 1) {
        if (text[i] == '@') {
            // Check there's a word char before and after
            if (common.isWordChar(text[i - 1]) and common.isWordChar(text[i + 1])) {
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

test {
    std.testing.refAllDecls(@This());
}
