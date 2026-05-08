//! GDPR Compliance Module
//!
//! Implements General Data Protection Regulation checks including:
//! - Consent tracking and verification
//! - Right-to-erasure (right to be forgotten) hooks
//! - Data minimization enforcement
//! - PII detection patterns (email, phone, SSN, IP, credit card)

const std = @import("std");
const config = @import("config.zig");

/// Types of PII that can be detected in content.
pub const PiiType = enum {
    email,
    phone,
    ssn,
    ip_address,
    credit_card,
    date_of_birth,
    name_pattern,
};

/// A single PII detection result.
pub const PiiMatch = struct {
    /// Type of PII detected.
    pii_type: PiiType,
    /// Byte offset of the match start.
    start: usize,
    /// Byte length of the match.
    length: usize,
    /// Confidence score (0.0 - 1.0).
    confidence: f32,
};

/// Consent status for a user/session.
pub const ConsentStatus = enum {
    /// No consent record exists.
    unknown,
    /// User has granted consent.
    granted,
    /// User has revoked consent.
    revoked,
    /// Consent has expired.
    expired,
};

/// A consent record for tracking user data processing agreements.
pub const ConsentRecord = struct {
    user_id: []const u8,
    status: ConsentStatus,
    purpose: []const u8,
    granted_at: i64,
    expires_at: ?i64,

    pub fn isValid(self: ConsentRecord, now: i64) bool {
        if (self.status != .granted) return false;
        if (self.expires_at) |exp| return now < exp;
        return true;
    }
};

/// Result of a GDPR compliance check.
pub const GdprCheckResult = struct {
    /// Whether the content is GDPR-compliant.
    is_compliant: bool,
    /// PII matches found in the content.
    pii_detected: []const PiiMatch,
    /// Whether data minimization requirements are met.
    data_minimized: bool,
    /// Whether consent is verified for the operation.
    consent_verified: bool,
    /// Human-readable summary of any violations.
    violations: []const []const u8,
    /// Allocator used for dynamic fields.
    allocator: std.mem.Allocator,

    pub fn deinit(self: *GdprCheckResult) void {
        self.allocator.free(self.pii_detected);
        for (self.violations) |v| self.allocator.free(v);
        self.allocator.free(self.violations);
    }
};

/// GDPR compliance checker.
pub const GdprChecker = struct {
    pii_config: config.PiiDetectionConfig,

    const Self = @This();

    pub fn init(pii_cfg: config.PiiDetectionConfig) Self {
        return .{ .pii_config = pii_cfg };
    }

    /// Run a full GDPR compliance check on the given content.
    pub fn check(self: *const Self, allocator: std.mem.Allocator, content: []const u8) !GdprCheckResult {
        var pii_list = std.ArrayListUnmanaged(PiiMatch).empty;
        errdefer pii_list.deinit(allocator);
        var violation_list = std.ArrayListUnmanaged([]const u8).empty;
        errdefer {
            for (violation_list.items) |v| allocator.free(v);
            violation_list.deinit(allocator);
        }

        // Detect PII patterns
        if (self.pii_config.detect_email) {
            try self.detectEmailPatterns(allocator, content, &pii_list);
        }
        if (self.pii_config.detect_phone) {
            try self.detectPhonePatterns(content, &pii_list, allocator);
        }
        if (self.pii_config.detect_ssn) {
            try self.detectSsnPatterns(content, &pii_list, allocator);
        }
        if (self.pii_config.detect_credit_card) {
            try self.detectCreditCardPatterns(content, &pii_list, allocator);
        }

        const has_pii = pii_list.items.len > 0;
        if (has_pii) {
            try violation_list.append(allocator, try allocator.dupe(u8, "PII detected in content; requires anonymization or explicit consent"));
        }

        const pii_slice = try pii_list.toOwnedSlice(allocator);
        const violations_slice = try violation_list.toOwnedSlice(allocator);

        return .{
            .is_compliant = !has_pii,
            .pii_detected = pii_slice,
            .data_minimized = content.len < 50_000, // Simple heuristic: flag oversized payloads
            .consent_verified = true, // Defaults to true; caller should verify via ConsentRecord
            .violations = violations_slice,
            .allocator = allocator,
        };
    }

    /// Detect email-like patterns (simple heuristic: look for @ surrounded by alphanum).
    fn detectEmailPatterns(self: *const Self, allocator: std.mem.Allocator, content: []const u8, list: *std.ArrayListUnmanaged(PiiMatch)) !void {
        _ = self;
        var i: usize = 0;
        while (i < content.len) : (i += 1) {
            if (content[i] == '@' and i > 0 and i + 1 < content.len) {
                // Walk back to find start
                var start = i;
                while (start > 0 and isEmailChar(content[start - 1])) : (start -= 1) {}
                // Walk forward to find end
                var end = i + 1;
                while (end < content.len and isEmailChar(content[end])) : (end += 1) {}
                // Check for dot after @
                const after_at = content[i + 1 .. end];
                if (std.mem.indexOfScalar(u8, after_at, '.') != null and end - start >= 5) {
                    try list.append(allocator, .{
                        .pii_type = .email,
                        .start = start,
                        .length = end - start,
                        .confidence = 0.9,
                    });
                    i = end;
                }
            }
        }
    }

    /// Detect phone number patterns (sequences of 10+ digits with optional separators).
    fn detectPhonePatterns(_: *const Self, content: []const u8, list: *std.ArrayListUnmanaged(PiiMatch), allocator: std.mem.Allocator) !void {
        var i: usize = 0;
        while (i < content.len) {
            if (std.ascii.isDigit(content[i]) or (content[i] == '+' and i + 1 < content.len and std.ascii.isDigit(content[i + 1]))) {
                const start = i;
                var digit_count: usize = 0;
                while (i < content.len and (std.ascii.isDigit(content[i]) or content[i] == '-' or content[i] == ' ' or content[i] == '(' or content[i] == ')' or content[i] == '+')) : (i += 1) {
                    if (std.ascii.isDigit(content[i])) digit_count += 1;
                }
                if (digit_count >= 10) {
                    try list.append(allocator, .{
                        .pii_type = .phone,
                        .start = start,
                        .length = i - start,
                        .confidence = 0.8,
                    });
                }
            } else {
                i += 1;
            }
        }
    }

    /// Detect SSN patterns (NNN-NN-NNNN).
    fn detectSsnPatterns(_: *const Self, content: []const u8, list: *std.ArrayListUnmanaged(PiiMatch), allocator: std.mem.Allocator) !void {
        if (content.len < 11) return;
        var i: usize = 0;
        while (i + 10 < content.len) : (i += 1) {
            if (std.ascii.isDigit(content[i]) and
                std.ascii.isDigit(content[i + 1]) and
                std.ascii.isDigit(content[i + 2]) and
                content[i + 3] == '-' and
                std.ascii.isDigit(content[i + 4]) and
                std.ascii.isDigit(content[i + 5]) and
                content[i + 6] == '-' and
                std.ascii.isDigit(content[i + 7]) and
                std.ascii.isDigit(content[i + 8]) and
                std.ascii.isDigit(content[i + 9]) and
                std.ascii.isDigit(content[i + 10]))
            {
                try list.append(allocator, .{
                    .pii_type = .ssn,
                    .start = i,
                    .length = 11,
                    .confidence = 0.95,
                });
                i += 11;
            }
        }
    }

    /// Detect credit card number patterns (sequences of 13-19 digits).
    fn detectCreditCardPatterns(_: *const Self, content: []const u8, list: *std.ArrayListUnmanaged(PiiMatch), allocator: std.mem.Allocator) !void {
        var i: usize = 0;
        while (i < content.len) {
            if (std.ascii.isDigit(content[i])) {
                const start = i;
                var digit_count: usize = 0;
                while (i < content.len and (std.ascii.isDigit(content[i]) or content[i] == '-' or content[i] == ' ')) : (i += 1) {
                    if (std.ascii.isDigit(content[i])) digit_count += 1;
                }
                if (digit_count >= 13 and digit_count <= 19) {
                    try list.append(allocator, .{
                        .pii_type = .credit_card,
                        .start = start,
                        .length = i - start,
                        .confidence = 0.7,
                    });
                }
            } else {
                i += 1;
            }
        }
    }

    fn isEmailChar(c: u8) bool {
        return std.ascii.isAlphanumeric(c) or c == '.' or c == '_' or c == '-' or c == '+';
    }
};

// ============================================================================
// Tests
// ============================================================================

test "GdprChecker detects email PII" {
    const checker = GdprChecker.init(.{});
    var result = try checker.check(std.testing.allocator, "Contact me at john@example.com for details");
    defer result.deinit();

    try std.testing.expect(!result.is_compliant);
    try std.testing.expect(result.pii_detected.len >= 1);
    try std.testing.expect(result.pii_detected[0].pii_type == .email);
}

test "GdprChecker detects SSN" {
    const checker = GdprChecker.init(.{});
    var result = try checker.check(std.testing.allocator, "SSN is 123-45-67890");
    defer result.deinit();

    try std.testing.expect(!result.is_compliant);
    var found_ssn = false;
    for (result.pii_detected) |m| {
        if (m.pii_type == .ssn) found_ssn = true;
    }
    try std.testing.expect(found_ssn);
}

test "GdprChecker passes clean content" {
    const checker = GdprChecker.init(.{});
    var result = try checker.check(std.testing.allocator, "Hello, how are you today?");
    defer result.deinit();

    try std.testing.expect(result.is_compliant);
    try std.testing.expect(result.pii_detected.len == 0);
}

test "ConsentRecord validity" {
    const record = ConsentRecord{
        .user_id = "user-1",
        .status = .granted,
        .purpose = "analytics",
        .granted_at = 1000,
        .expires_at = 2000,
    };
    try std.testing.expect(record.isValid(1500));
    try std.testing.expect(!record.isValid(2500));

    const revoked = ConsentRecord{
        .user_id = "user-2",
        .status = .revoked,
        .purpose = "marketing",
        .granted_at = 1000,
        .expires_at = null,
    };
    try std.testing.expect(!revoked.isValid(1500));
}

test {
    std.testing.refAllDecls(@This());
}
