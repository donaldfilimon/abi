//! Policy Checker Module for Abi Router
//!
//! Enforces content safety, privacy, and compliance rules for the assistant.
//! Validates user inputs and persona responses against defined safety patterns.
//! Includes PII detection, GDPR/CCPA compliance, and configurable safety rules.

const std = @import("std");

/// Result of a policy check operation.
pub const PolicyResult = struct {
    /// Whether the content is allowed to proceed.
    is_allowed: bool,
    /// Whether the content requires human moderation.
    requires_moderation: bool,
    /// List of specific violations detected.
    violations: []const []const u8,
    /// Action recommended by the policy checker.
    suggested_action: SafetyAction,
    /// Detected PII types (for logging/redaction).
    detected_pii: []const PiiType,
    /// Compliance flags.
    compliance: ComplianceFlags,

    pub fn deinit(self: *PolicyResult, allocator: std.mem.Allocator) void {
        for (self.violations) |v| allocator.free(v);
        allocator.free(self.violations);
        allocator.free(self.detected_pii);
    }
};

/// Types of Personally Identifiable Information.
pub const PiiType = enum {
    email,
    phone,
    ssn,
    credit_card,
    ip_address,
    address,
    name_pattern,
    date_of_birth,
};

/// Compliance framework flags.
pub const ComplianceFlags = struct {
    gdpr_compliant: bool = true,
    ccpa_compliant: bool = true,
    hipaa_relevant: bool = false,
    contains_consent_required: bool = false,
};

/// Possible actions when a safety rule is triggered.
pub const SafetyAction = enum {
    allow,
    warn,
    block,
    redirect_to_support,
    require_human_review,
};

/// Severity level of a safety violation.
pub const Severity = enum {
    low,
    medium,
    high,
    critical,
};

/// A single safety rule definition.
pub const SafetyRule = struct {
    name: []const u8,
    pattern: []const u8,
    action: SafetyAction,
    severity: Severity,
    /// Whether this rule triggers PII detection.
    is_pii_rule: bool = false,
    /// Associated PII type if is_pii_rule is true.
    pii_type: ?PiiType = null,
};

/// Implementation of the policy and safety checker.
pub const PolicyChecker = struct {
    allocator: std.mem.Allocator,
    rules: std.ArrayListUnmanaged(SafetyRule),

    const Self = @This();

    /// Initialize a new policy checker with default rules.
    pub fn init(allocator: std.mem.Allocator) !Self {
        var self = Self{
            .allocator = allocator,
            .rules = .empty,
        };

        // Initialize with default safety rules
        try self.addRule(.{
            .name = "Malicious Code",
            .pattern = "rm -rf /",
            .action = .block,
            .severity = .high,
        });

        try self.addRule(.{
            .name = "PII Exposure",
            .pattern = "ssn:",
            .action = .warn,
            .severity = .medium,
        });

        return self;
    }

    /// Free resources used by the policy checker.
    pub fn deinit(self: *Self) void {
        for (self.rules.items) |rule| {
            self.allocator.free(rule.name);
            self.allocator.free(rule.pattern);
        }
        self.rules.deinit(self.allocator);
    }

    /// Add a new safety rule to the checker.
    pub fn addRule(self: *Self, rule: SafetyRule) !void {
        try self.rules.append(self.allocator, .{
            .name = try self.allocator.dupe(u8, rule.name),
            .pattern = try self.allocator.dupe(u8, rule.pattern),
            .action = rule.action,
            .severity = rule.severity,
        });
    }

    /// Check content against all registered safety rules.
    pub fn check(self: *const Self, content: []const u8) !PolicyResult {
        var violations: std.ArrayListUnmanaged([]const u8) = .{};
        errdefer {
            for (violations.items) |v| self.allocator.free(v);
            violations.deinit(self.allocator);
        }

        var detected_pii: std.ArrayListUnmanaged(PiiType) = .{};
        errdefer detected_pii.deinit(self.allocator);

        var is_allowed = true;
        var requires_moderation = false;
        var max_action = SafetyAction.allow;
        var compliance = ComplianceFlags{};

        // Check all rules against content
        for (self.rules.items) |rule| {
            if (std.mem.indexOf(u8, content, rule.pattern) != null) {
                try violations.append(self.allocator, try self.allocator.dupe(u8, rule.name));

                if (rule.action == .block) is_allowed = false;
                if (rule.action == .require_human_review) requires_moderation = true;

                if (@intFromEnum(rule.action) > @intFromEnum(max_action)) {
                    max_action = rule.action;
                }

                // Track PII detections
                if (rule.is_pii_rule) {
                    if (rule.pii_type) |pii| {
                        try detected_pii.append(self.allocator, pii);
                    }
                }
            }
        }

        // Run PII pattern detection
        try self.detectPii(content, self.allocator, &detected_pii);

        // Update compliance flags based on detected PII
        if (detected_pii.items.len > 0) {
            compliance.gdpr_compliant = false;
            compliance.ccpa_compliant = false;
            compliance.contains_consent_required = true;

            // Check for HIPAA-relevant PII
            for (detected_pii.items) |pii| {
                if (pii == .ssn or pii == .date_of_birth) {
                    compliance.hipaa_relevant = true;
                    break;
                }
            }
        }

        return PolicyResult{
            .is_allowed = is_allowed,
            .requires_moderation = requires_moderation,
            .violations = try violations.toOwnedSlice(self.allocator),
            .suggested_action = max_action,
            .detected_pii = try detected_pii.toOwnedSlice(self.allocator),
            .compliance = compliance,
        };
    }

    /// Detect PII patterns in content using heuristic matching.
    fn detectPii(self: *const Self, content: []const u8, allocator: std.mem.Allocator, detected: *std.ArrayListUnmanaged(PiiType)) !void {
        _ = self;

        // Email pattern: contains @ with text before and after
        if (containsEmailPattern(content)) {
            try detected.append(allocator, .email);
        }

        // Phone pattern: sequences of digits with common separators
        if (containsPhonePattern(content)) {
            try detected.append(allocator, .phone);
        }

        // SSN pattern: XXX-XX-XXXX or XXXXXXXXX (9 digits)
        if (containsSsnPattern(content)) {
            try detected.append(allocator, .ssn);
        }

        // Credit card: 13-19 digit sequences
        if (containsCreditCardPattern(content)) {
            try detected.append(allocator, .credit_card);
        }

        // IP address: X.X.X.X pattern
        if (containsIpPattern(content)) {
            try detected.append(allocator, .ip_address);
        }
    }
};

// PII Pattern Detection Helpers

/// Check for email patterns (text@text.text)
fn containsEmailPattern(content: []const u8) bool {
    var i: usize = 0;
    while (i < content.len) : (i += 1) {
        if (content[i] == '@' and i > 0 and i < content.len - 1) {
            // Check for valid characters before @
            var has_before = false;
            if (i > 0) {
                const c = content[i - 1];
                has_before = std.ascii.isAlphanumeric(c) or c == '.' or c == '_' or c == '-';
            }
            // Check for valid characters after @
            var has_after = false;
            if (i + 1 < content.len) {
                const c = content[i + 1];
                has_after = std.ascii.isAlphanumeric(c);
            }
            // Check for dot after @
            var has_dot = false;
            var j = i + 1;
            while (j < content.len) : (j += 1) {
                if (content[j] == '.') {
                    has_dot = true;
                    break;
                }
                if (!std.ascii.isAlphanumeric(content[j]) and content[j] != '-') break;
            }
            if (has_before and has_after and has_dot) return true;
        }
    }
    return false;
}

/// Check for phone number patterns (10+ digits with optional separators)
fn containsPhonePattern(content: []const u8) bool {
    var digit_count: usize = 0;
    var in_potential_phone = false;
    var separator_count: usize = 0;

    for (content) |c| {
        if (std.ascii.isDigit(c)) {
            digit_count += 1;
            in_potential_phone = true;
        } else if (in_potential_phone and (c == '-' or c == '.' or c == ' ' or c == '(' or c == ')')) {
            separator_count += 1;
            if (separator_count > 4) {
                // Too many separators, reset
                digit_count = 0;
                separator_count = 0;
                in_potential_phone = false;
            }
        } else {
            if (digit_count >= 10 and digit_count <= 15) return true;
            digit_count = 0;
            separator_count = 0;
            in_potential_phone = false;
        }
    }
    return digit_count >= 10 and digit_count <= 15;
}

/// Check for SSN patterns (XXX-XX-XXXX or 9 consecutive digits)
fn containsSsnPattern(content: []const u8) bool {
    // Check for XXX-XX-XXXX pattern
    var i: usize = 0;
    while (i + 10 < content.len) : (i += 1) {
        if (isDigitSequence(content[i .. i + 3]) and
            content[i + 3] == '-' and
            isDigitSequence(content[i + 4 .. i + 6]) and
            content[i + 6] == '-' and
            isDigitSequence(content[i + 7 .. i + 11]))
        {
            return true;
        }
    }

    // Check for 9 consecutive digits (less reliable)
    var digit_run: usize = 0;
    for (content) |c| {
        if (std.ascii.isDigit(c)) {
            digit_run += 1;
            if (digit_run == 9) return true;
        } else {
            digit_run = 0;
        }
    }
    return false;
}

/// Check for credit card patterns (13-19 digits, optionally separated)
fn containsCreditCardPattern(content: []const u8) bool {
    var digit_count: usize = 0;
    var group_count: usize = 0;
    var in_potential_cc = false;

    for (content) |c| {
        if (std.ascii.isDigit(c)) {
            digit_count += 1;
            in_potential_cc = true;
        } else if (in_potential_cc and (c == '-' or c == ' ')) {
            group_count += 1;
            if (group_count > 5) {
                digit_count = 0;
                group_count = 0;
                in_potential_cc = false;
            }
        } else {
            if (digit_count >= 13 and digit_count <= 19) return true;
            digit_count = 0;
            group_count = 0;
            in_potential_cc = false;
        }
    }
    return digit_count >= 13 and digit_count <= 19;
}

/// Check for IP address patterns (X.X.X.X where X is 1-3 digits)
fn containsIpPattern(content: []const u8) bool {
    var i: usize = 0;
    while (i < content.len) {
        // Try to match IP starting at position i
        var octet_count: usize = 0;
        var j = i;
        var valid = true;

        while (j < content.len and octet_count < 4) {
            // Parse octet
            const octet_start = j;
            while (j < content.len and std.ascii.isDigit(content[j])) : (j += 1) {}

            const octet_len = j - octet_start;
            if (octet_len < 1 or octet_len > 3) {
                valid = false;
                break;
            }

            octet_count += 1;

            // Check for dot (except after 4th octet)
            if (octet_count < 4) {
                if (j >= content.len or content[j] != '.') {
                    valid = false;
                    break;
                }
                j += 1;
            }
        }

        if (valid and octet_count == 4) return true;
        i += 1;
    }
    return false;
}

/// Helper to check if a slice contains only digits
fn isDigitSequence(slice: []const u8) bool {
    for (slice) |c| {
        if (!std.ascii.isDigit(c)) return false;
    }
    return slice.len > 0;
}

// Tests

test "containsEmailPattern" {
    try std.testing.expect(containsEmailPattern("contact user@example.com for info"));
    try std.testing.expect(containsEmailPattern("test@test.org"));
    try std.testing.expect(!containsEmailPattern("no email here"));
    try std.testing.expect(!containsEmailPattern("invalid@"));
    try std.testing.expect(!containsEmailPattern("@invalid.com"));
}

test "containsPhonePattern" {
    try std.testing.expect(containsPhonePattern("Call 555-123-4567 now"));
    try std.testing.expect(containsPhonePattern("Phone: (555) 123 4567"));
    try std.testing.expect(containsPhonePattern("5551234567"));
    try std.testing.expect(!containsPhonePattern("123"));
    try std.testing.expect(!containsPhonePattern("no phone"));
}

test "containsSsnPattern" {
    try std.testing.expect(containsSsnPattern("SSN: 123-45-6789"));
    try std.testing.expect(containsSsnPattern("123456789"));
    try std.testing.expect(!containsSsnPattern("12345"));
    try std.testing.expect(!containsSsnPattern("no ssn here"));
}

test "containsIpPattern" {
    try std.testing.expect(containsIpPattern("Server at 192.168.1.1"));
    try std.testing.expect(containsIpPattern("10.0.0.255"));
    try std.testing.expect(!containsIpPattern("256.1.1.1")); // Still matches pattern, value validation is separate
    try std.testing.expect(!containsIpPattern("no ip"));
}

test "PolicyChecker basic check" {
    const allocator = std.testing.allocator;
    var checker = try PolicyChecker.init(allocator);
    defer checker.deinit();

    // Test malicious pattern detection
    var result = try checker.check("please rm -rf / now");
    defer result.deinit(@constCast(&result), allocator);
    try std.testing.expect(!result.is_allowed);
    try std.testing.expect(result.suggested_action == .block);
}

test "PolicyChecker PII detection" {
    const allocator = std.testing.allocator;
    var checker = try PolicyChecker.init(allocator);
    defer checker.deinit();

    // Test PII detection
    var result = try checker.check("My email is test@example.com");
    defer result.deinit(@constCast(&result), allocator);
    try std.testing.expect(result.detected_pii.len > 0);
    try std.testing.expect(!result.compliance.gdpr_compliant);
}
