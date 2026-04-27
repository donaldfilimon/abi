//! Password strength analysis functions.

const std = @import("std");
const csprng = @import("../csprng.zig");
const types = @import("types.zig");

const ClassFlags = struct {
    has_lower: bool,
    has_upper: bool,
    has_digit: bool,
    has_special: bool,
};

fn scoreLength(len: usize) u32 {
    var score: u32 = 0;
    if (len >= 8) score += 10;
    if (len >= 12) score += 10;
    if (len >= 16) score += 10;
    if (len >= 20) score += 10;
    return score;
}

fn analyzeClasses(password: []const u8) ClassFlags {
    var flags = ClassFlags{
        .has_lower = false,
        .has_upper = false,
        .has_digit = false,
        .has_special = false,
    };

    for (password) |c| {
        if (c >= 'a' and c <= 'z') flags.has_lower = true;
        if (c >= 'A' and c <= 'Z') flags.has_upper = true;
        if (c >= '0' and c <= '9') flags.has_digit = true;
        if ((c >= '!' and c <= '/') or (c >= ':' and c <= '@') or
            (c >= '[' and c <= '`') or (c >= '{' and c <= '~'))
        {
            flags.has_special = true;
        }
    }

    return flags;
}

fn appendFeedback(feedback: *csprng.FixedList([]const u8, 10), message: []const u8) void {
    feedback.append(message) catch |err| {
        std.log.debug("Password feedback limit reached: {t}", .{err});
    };
}

fn scoreClasses(feedback: *csprng.FixedList([]const u8, 10), flags: ClassFlags) u32 {
    var score: u32 = 0;

    if (flags.has_lower) score += 10 else appendFeedback(feedback, "Add lowercase letters");
    if (flags.has_upper) score += 10 else appendFeedback(feedback, "Add uppercase letters");
    if (flags.has_digit) score += 10 else appendFeedback(feedback, "Add numbers");
    if (flags.has_special) score += 15 else appendFeedback(feedback, "Add special characters");

    return score;
}

fn applyPenalty(feedback: *csprng.FixedList([]const u8, 10), score: *u32, condition: bool, penalty: u32, message: []const u8) void {
    if (!condition) return;
    score.* -|= penalty;
    appendFeedback(feedback, message);
}

fn strengthFromScore(score: u32) types.PasswordStrength {
    return if (score >= 60)
        .very_strong
    else if (score >= 45)
        .strong
    else if (score >= 30)
        .fair
    else if (score >= 15)
        .weak
    else
        .very_weak;
}

fn crackTimeFromScore(score: u32) []const u8 {
    return if (score >= 60)
        "centuries"
    else if (score >= 45)
        "years"
    else if (score >= 30)
        "months"
    else if (score >= 15)
        "days"
    else
        "instant";
}

/// Analyze password strength
pub fn analyzeStrength(password: []const u8) types.StrengthAnalysis {
    var score: u32 = scoreLength(password.len);
    var feedback: csprng.FixedList([]const u8, 10) = .{};

    const class_flags = analyzeClasses(password);
    score += scoreClasses(&feedback, class_flags);

    const has_common = containsCommonPattern(password);
    applyPenalty(&feedback, &score, has_common, 20, "Avoid common patterns");

    applyPenalty(&feedback, &score, hasSequentialChars(password), 10, "Avoid sequential characters");
    applyPenalty(&feedback, &score, hasRepeatedChars(password), 10, "Avoid repeated characters");

    const strength = strengthFromScore(score);
    const crack_time = crackTimeFromScore(score);

    return .{
        .strength = strength,
        .score = score,
        .feedback = feedback.slice(),
        .has_lowercase = class_flags.has_lower,
        .has_uppercase = class_flags.has_upper,
        .has_digits = class_flags.has_digit,
        .has_special = class_flags.has_special,
        .has_common_pattern = has_common,
        .estimated_crack_time = crack_time,
    };
}

fn containsCommonPattern(password: []const u8) bool {
    const common_patterns = &[_][]const u8{
        "password", "123456", "qwerty",   "abc123",   "letmein",
        "welcome",  "monkey", "dragon",   "master",   "login",
        "admin",    "root",   "pass",     "test",     "guest",
        "hello",    "shadow", "sunshine", "princess", "football",
    };

    var lower_buf: [128]u8 = undefined;
    const lower = std.ascii.lowerString(lower_buf[0..@min(password.len, 128)], password);

    for (common_patterns) |pattern| {
        if (std.mem.indexOf(u8, lower, pattern) != null) {
            return true;
        }
    }

    return false;
}

fn hasSequentialChars(password: []const u8) bool {
    if (password.len < 3) return false;

    for (0..password.len - 2) |i| {
        const a = password[i];
        const b = password[i + 1];
        const c = password[i + 2];

        if (b == a + 1 and c == b + 1) return true;
        if (b == a -% 1 and c == b -% 1 and a > 0 and b > 0) return true;
    }

    return false;
}

fn hasRepeatedChars(password: []const u8) bool {
    if (password.len < 3) return false;

    for (0..password.len - 2) |i| {
        if (password[i] == password[i + 1] and password[i + 1] == password[i + 2]) {
            return true;
        }
    }

    return false;
}

test "password strength analysis" {
    const weak = analyzeStrength("password");
    try std.testing.expect(weak.strength == .very_weak or weak.strength == .weak);
    try std.testing.expect(weak.has_common_pattern);

    const strong = analyzeStrength("MyStr0ng!P@ssw0rd#2024");
    try std.testing.expect(strong.strength == .strong or strong.strength == .very_strong);
    try std.testing.expect(strong.has_lowercase);
    try std.testing.expect(strong.has_uppercase);
    try std.testing.expect(strong.has_digits);
    try std.testing.expect(strong.has_special);
}

test "sequential chars detection" {
    try std.testing.expect(hasSequentialChars("abc"));
    try std.testing.expect(hasSequentialChars("123"));
    try std.testing.expect(hasSequentialChars("xyz"));
    try std.testing.expect(!hasSequentialChars("aZb"));
    try std.testing.expect(!hasSequentialChars("a1b"));
}

test "repeated chars detection" {
    try std.testing.expect(hasRepeatedChars("aaa"));
    try std.testing.expect(hasRepeatedChars("111"));
    try std.testing.expect(!hasRepeatedChars("aba"));
    try std.testing.expect(!hasRepeatedChars("121"));
}

test {
    std.testing.refAllDecls(@This());
}
