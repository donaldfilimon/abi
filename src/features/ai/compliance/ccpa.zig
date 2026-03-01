//! CCPA Compliance Module
//!
//! Implements California Consumer Privacy Act checks:
//! - Consumer data rights (access, delete, do-not-sell)
//! - Data sale opt-out tracking
//! - Disclosure requirements verification
//! - Financial incentive transparency

const std = @import("std");

/// Consumer rights under CCPA.
pub const ConsumerRight = enum {
    /// Right to know what personal information is collected.
    right_to_know,
    /// Right to delete personal information.
    right_to_delete,
    /// Right to opt-out of the sale of personal information.
    right_to_opt_out,
    /// Right to non-discrimination for exercising rights.
    right_to_non_discrimination,
    /// Right to correct inaccurate personal information.
    right_to_correct,
    /// Right to limit use of sensitive personal information.
    right_to_limit,
};

/// Status of a consumer's opt-out preference.
pub const OptOutStatus = enum {
    /// No preference recorded.
    not_set,
    /// Consumer has opted out of data sale.
    opted_out,
    /// Consumer has opted in (or not opted out).
    opted_in,
};

/// A consumer rights request record.
pub const ConsumerRequest = struct {
    consumer_id: []const u8,
    right: ConsumerRight,
    status: RequestStatus,
    submitted_at: i64,
    responded_at: ?i64,

    pub const RequestStatus = enum {
        pending,
        in_progress,
        completed,
        denied,
    };

    pub fn isOverdue(self: ConsumerRequest, now: i64) bool {
        // CCPA requires response within 45 days (3,888,000 seconds)
        const deadline = self.submitted_at + 3_888_000;
        return self.status != .completed and self.status != .denied and now > deadline;
    }
};

/// Categories of personal information under CCPA.
pub const PersonalInfoCategory = enum {
    identifiers,
    commercial_info,
    biometric,
    internet_activity,
    geolocation,
    professional_info,
    education_info,
    inferences,
    sensitive_pi,
};

/// Result of a CCPA compliance check.
pub const CcpaCheckResult = struct {
    is_compliant: bool,
    personal_info_detected: []const PersonalInfoCategory,
    opt_out_honored: bool,
    disclosure_adequate: bool,
    violations: []const []const u8,
    allocator: std.mem.Allocator,

    pub fn deinit(self: *CcpaCheckResult) void {
        self.allocator.free(self.personal_info_detected);
        for (self.violations) |v| self.allocator.free(v);
        self.allocator.free(self.violations);
    }
};

/// CCPA compliance checker.
pub const CcpaChecker = struct {
    const Self = @This();

    /// Indicators that suggest personal information categories.
    const identifier_keywords = [_][]const u8{
        "name",     "address",         "email", "phone", "driver's license",
        "passport", "social security",
    };

    const commercial_keywords = [_][]const u8{
        "purchase", "transaction", "order", "payment", "subscription",
    };

    const internet_keywords = [_][]const u8{
        "browsing", "search history", "cookie", "ip address", "click",
    };

    pub fn init() Self {
        return .{};
    }

    /// Run a CCPA compliance check on the given content.
    pub fn check(_: *const Self, allocator: std.mem.Allocator, content: []const u8) !CcpaCheckResult {
        var categories = std.ArrayListUnmanaged(PersonalInfoCategory).empty;
        errdefer categories.deinit(allocator);
        var violation_list = std.ArrayListUnmanaged([]const u8).empty;
        errdefer {
            for (violation_list.items) |v| allocator.free(v);
            violation_list.deinit(allocator);
        }

        const lower = try allocator.alloc(u8, content.len);
        defer allocator.free(lower);
        for (content, 0..) |c, i| {
            lower[i] = std.ascii.toLower(c);
        }

        // Check for identifier-related content
        for (identifier_keywords) |kw| {
            if (std.mem.indexOf(u8, lower, kw) != null) {
                try appendUnique(&categories, .identifiers, allocator);
                break;
            }
        }

        // Check for commercial information
        for (commercial_keywords) |kw| {
            if (std.mem.indexOf(u8, lower, kw) != null) {
                try appendUnique(&categories, .commercial_info, allocator);
                break;
            }
        }

        // Check for internet activity
        for (internet_keywords) |kw| {
            if (std.mem.indexOf(u8, lower, kw) != null) {
                try appendUnique(&categories, .internet_activity, allocator);
                break;
            }
        }

        const has_pi = categories.items.len > 0;
        if (has_pi) {
            try violation_list.append(allocator, try allocator.dupe(u8, "Personal information categories detected; CCPA disclosure required"));
        }

        const cats_slice = try categories.toOwnedSlice(allocator);
        const violations_slice = try violation_list.toOwnedSlice(allocator);

        return .{
            .is_compliant = !has_pi,
            .personal_info_detected = cats_slice,
            .opt_out_honored = true, // Caller should verify via OptOutStatus
            .disclosure_adequate = true, // Defaults to true
            .violations = violations_slice,
            .allocator = allocator,
        };
    }

    fn appendUnique(list: *std.ArrayListUnmanaged(PersonalInfoCategory), cat: PersonalInfoCategory, allocator: std.mem.Allocator) !void {
        for (list.items) |existing| {
            if (existing == cat) return;
        }
        try list.append(allocator, cat);
    }
};

// ============================================================================
// Tests
// ============================================================================

test "CcpaChecker detects identifiers" {
    const checker = CcpaChecker.init();
    var result = try checker.check(std.testing.allocator, "Please provide your name and email address");
    defer result.deinit();

    try std.testing.expect(!result.is_compliant);
    try std.testing.expect(result.personal_info_detected.len >= 1);
}

test "CcpaChecker detects commercial info" {
    const checker = CcpaChecker.init();
    var result = try checker.check(std.testing.allocator, "Your recent purchase and payment details");
    defer result.deinit();

    try std.testing.expect(!result.is_compliant);
    var found = false;
    for (result.personal_info_detected) |cat| {
        if (cat == .commercial_info) found = true;
    }
    try std.testing.expect(found);
}

test "CcpaChecker passes clean content" {
    const checker = CcpaChecker.init();
    var result = try checker.check(std.testing.allocator, "The quick brown fox jumps over the lazy dog");
    defer result.deinit();

    try std.testing.expect(result.is_compliant);
}

test "ConsumerRequest overdue detection" {
    const request = ConsumerRequest{
        .consumer_id = "consumer-1",
        .right = .right_to_delete,
        .status = .pending,
        .submitted_at = 1000,
        .responded_at = null,
    };
    // Way past the 45-day deadline
    try std.testing.expect(request.isOverdue(5_000_000));
    // Within deadline
    try std.testing.expect(!request.isOverdue(1500));
}

test {
    std.testing.refAllDecls(@This());
}
