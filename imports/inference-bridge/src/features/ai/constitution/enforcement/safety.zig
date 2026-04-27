//! Safety Principle Validator (Principle 1)
//!
//! Never generate harmful, dangerous, or illegal content.
//! Detects shell injection, malware, harm instructions, and weapons content.

const std = @import("std");
const common = @import("common.zig");

const SafetyScore = common.SafetyScore;

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

pub fn checkShellInjection(text: []const u8, score: *SafetyScore, context_weight: f32) void {
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

pub fn containsMalwarePatterns(text: []const u8) bool {
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

pub fn containsHarmPatterns(text: []const u8) bool {
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

pub fn containsWeaponsPatterns(text: []const u8) bool {
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

pub fn checkSqlInjection(text: []const u8, score: *SafetyScore, context_weight: f32) void {
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

pub fn checkPathTraversal(text: []const u8, score: *SafetyScore, context_weight: f32) void {
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

test {
    std.testing.refAllDecls(@This());
}
