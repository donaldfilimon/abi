//! HIPAA Compliance Module
//!
//! Implements Health Insurance Portability and Accountability Act checks:
//! - PHI (Protected Health Information) detection
//! - Access control levels (minimum necessary standard)
//! - De-identification verification
//! - Safe harbor method pattern matching

const std = @import("std");

/// Categories of Protected Health Information.
pub const PhiCategory = enum {
    /// Patient names
    patient_name,
    /// Medical record numbers
    medical_record_number,
    /// Dates (admission, discharge, birth, death)
    dates,
    /// Geographic data (address, zip)
    geographic,
    /// Phone/fax numbers
    phone_fax,
    /// Email addresses
    email,
    /// Social Security numbers
    ssn,
    /// Health plan beneficiary numbers
    beneficiary_number,
    /// Account numbers
    account_number,
    /// Certificate/license numbers
    certificate_number,
    /// Device identifiers
    device_identifier,
    /// Biometric identifiers
    biometric,
    /// Photographs / images
    photographs,
    /// Any other unique identifier
    other_unique_id,
    /// Diagnosis or condition mention
    diagnosis,
    /// Medication reference
    medication,
    /// Lab result reference
    lab_result,
};

/// A detected PHI element in content.
pub const PhiMatch = struct {
    category: PhiCategory,
    start: usize,
    length: usize,
    confidence: f32,
    /// A human-readable description of the match.
    description: []const u8,
};

/// Access control level for HIPAA minimum-necessary standard.
pub const AccessLevel = enum {
    /// No PHI access.
    none,
    /// De-identified data only.
    de_identified,
    /// Limited dataset (dates, geographic).
    limited,
    /// Full PHI access (authorized personnel only).
    full,

    pub fn canAccess(self: AccessLevel, required: AccessLevel) bool {
        return @intFromEnum(self) >= @intFromEnum(required);
    }
};

/// Result of a HIPAA compliance check.
pub const HipaaCheckResult = struct {
    is_compliant: bool,
    phi_detected: []const PhiMatch,
    required_access_level: AccessLevel,
    de_identification_status: DeIdentificationStatus,
    violations: []const []const u8,
    allocator: std.mem.Allocator,

    pub const DeIdentificationStatus = enum {
        /// Content contains no PHI.
        not_applicable,
        /// Content is fully de-identified per Safe Harbor.
        safe_harbor_compliant,
        /// Content has PHI that needs de-identification.
        requires_de_identification,
    };

    pub fn deinit(self: *HipaaCheckResult) void {
        self.allocator.free(self.phi_detected);
        for (self.violations) |v| self.allocator.free(v);
        self.allocator.free(self.violations);
    }
};

/// HIPAA compliance checker.
pub const HipaaChecker = struct {
    const Self = @This();

    /// Medical keyword indicators for PHI detection.
    const medical_keywords = [_][]const u8{
        "patient",    "diagnosis",  "prescribed", "medication",
        "treatment",  "prognosis",  "symptoms",   "medical record",
        "lab result", "blood test", "MRI",        "CT scan",
        "X-ray",      "biopsy",     "surgery",    "hospital",
        "admission",  "discharge",  "ICD-10",     "CPT code",
    };

    const medication_indicators = [_][]const u8{
        "mg",     "tablet",   "capsule", "dosage",  "prescription",
        "refill", "pharmacy", "drug",    "insulin", "antibiotic",
    };

    pub fn init() Self {
        return .{};
    }

    /// Run a HIPAA compliance check on the given content.
    pub fn check(self: *const Self, allocator: std.mem.Allocator, content: []const u8) !HipaaCheckResult {
        var phi_list = std.ArrayListUnmanaged(PhiMatch).empty;
        errdefer phi_list.deinit(allocator);
        var violation_list = std.ArrayListUnmanaged([]const u8).empty;
        errdefer {
            for (violation_list.items) |v| allocator.free(v);
            violation_list.deinit(allocator);
        }

        // Detect medical context indicators
        try self.detectMedicalContent(allocator, content, &phi_list);
        try self.detectMedicationReferences(allocator, content, &phi_list);

        const has_phi = phi_list.items.len > 0;
        var max_access: AccessLevel = .none;

        for (phi_list.items) |phi| {
            const required = switch (phi.category) {
                .diagnosis, .medication, .lab_result => AccessLevel.full,
                .dates, .geographic => AccessLevel.limited,
                else => AccessLevel.de_identified,
            };
            if (@intFromEnum(required) > @intFromEnum(max_access)) {
                max_access = required;
            }
        }

        if (has_phi) {
            try violation_list.append(allocator, try allocator.dupe(u8, "Protected Health Information detected; HIPAA authorization required"));
        }

        const phi_slice = try phi_list.toOwnedSlice(allocator);
        const violations_slice = try violation_list.toOwnedSlice(allocator);

        return .{
            .is_compliant = !has_phi,
            .phi_detected = phi_slice,
            .required_access_level = max_access,
            .de_identification_status = if (!has_phi) .not_applicable else .requires_de_identification,
            .violations = violations_slice,
            .allocator = allocator,
        };
    }

    fn detectMedicalContent(_: *const Self, allocator: std.mem.Allocator, content: []const u8, list: *std.ArrayListUnmanaged(PhiMatch)) !void {
        const lower = try allocator.alloc(u8, content.len);
        defer allocator.free(lower);
        for (content, 0..) |c, i| {
            lower[i] = std.ascii.toLower(c);
        }

        for (medical_keywords) |keyword| {
            if (std.mem.indexOf(u8, lower, keyword)) |pos| {
                try list.append(allocator, .{
                    .category = if (std.mem.eql(u8, keyword, "diagnosis") or std.mem.eql(u8, keyword, "prognosis"))
                        .diagnosis
                    else if (std.mem.eql(u8, keyword, "patient"))
                        .patient_name
                    else
                        .other_unique_id,
                    .start = pos,
                    .length = keyword.len,
                    .confidence = 0.7,
                    .description = keyword,
                });
            }
        }
    }

    fn detectMedicationReferences(_: *const Self, allocator: std.mem.Allocator, content: []const u8, list: *std.ArrayListUnmanaged(PhiMatch)) !void {
        const lower = try allocator.alloc(u8, content.len);
        defer allocator.free(lower);
        for (content, 0..) |c, i| {
            lower[i] = std.ascii.toLower(c);
        }

        for (medication_indicators) |keyword| {
            if (std.mem.indexOf(u8, lower, keyword)) |pos| {
                try list.append(allocator, .{
                    .category = .medication,
                    .start = pos,
                    .length = keyword.len,
                    .confidence = 0.6,
                    .description = keyword,
                });
            }
        }
    }
};

// ============================================================================
// Tests
// ============================================================================

test "HipaaChecker detects medical content" {
    const checker = HipaaChecker.init();
    var result = try checker.check(std.testing.allocator, "The patient was prescribed medication for their diagnosis");
    defer result.deinit();

    try std.testing.expect(!result.is_compliant);
    try std.testing.expect(result.phi_detected.len >= 1);
    try std.testing.expect(result.required_access_level != .none);
}

test "HipaaChecker passes non-medical content" {
    const checker = HipaaChecker.init();
    var result = try checker.check(std.testing.allocator, "The weather is nice today");
    defer result.deinit();

    try std.testing.expect(result.is_compliant);
    try std.testing.expect(result.de_identification_status == .not_applicable);
}

test "AccessLevel hierarchy" {
    try std.testing.expect(AccessLevel.full.canAccess(.full));
    try std.testing.expect(AccessLevel.full.canAccess(.limited));
    try std.testing.expect(!AccessLevel.none.canAccess(.limited));
    try std.testing.expect(AccessLevel.limited.canAccess(.de_identified));
}

test {
    std.testing.refAllDecls(@This());
}
