//! Compliance Module — Unified regulatory compliance for ABI.
//!
//! Orchestrates GDPR, HIPAA, and CCPA compliance checks with audit trail
//! logging. Integrates with the persona system to validate responses and
//! interactions against applicable regulations.
//!
//! Integration points:
//! - Pre-response: `checkContent()` → validate before sending to user
//! - Post-generation: `validateResponse()` → audit trail + compliance gate
//! - Audit: `getAuditTrail()` → query compliance event history

const std = @import("std");
pub const config = @import("config.zig");
pub const gdpr = @import("gdpr.zig");
pub const hipaa = @import("hipaa.zig");
pub const ccpa = @import("ccpa.zig");
pub const audit = @import("audit.zig");

// Re-export core types
pub const ComplianceConfig = config.ComplianceConfig;
pub const GdprChecker = gdpr.GdprChecker;
pub const GdprCheckResult = gdpr.GdprCheckResult;
pub const PiiType = gdpr.PiiType;
pub const PiiMatch = gdpr.PiiMatch;
pub const ConsentStatus = gdpr.ConsentStatus;
pub const ConsentRecord = gdpr.ConsentRecord;
pub const HipaaChecker = hipaa.HipaaChecker;
pub const HipaaCheckResult = hipaa.HipaaCheckResult;
pub const PhiCategory = hipaa.PhiCategory;
pub const PhiMatch = hipaa.PhiMatch;
pub const AccessLevel = hipaa.AccessLevel;
pub const CcpaChecker = ccpa.CcpaChecker;
pub const CcpaCheckResult = ccpa.CcpaCheckResult;
pub const ConsumerRight = ccpa.ConsumerRight;
pub const PersonalInfoCategory = ccpa.PersonalInfoCategory;
pub const AuditTrail = audit.AuditTrail;
pub const AuditEvent = audit.AuditEvent;
pub const AuditEventType = audit.AuditEventType;
pub const AuditSeverity = audit.AuditSeverity;

/// Unified compliance check result.
pub const ComplianceCheckResult = struct {
    /// Overall compliance status.
    is_compliant: bool,
    /// GDPR result (null if GDPR disabled).
    gdpr_result: ?gdpr.GdprCheckResult,
    /// HIPAA result (null if HIPAA disabled).
    hipaa_result: ?hipaa.HipaaCheckResult,
    /// CCPA result (null if CCPA disabled).
    ccpa_result: ?ccpa.CcpaCheckResult,
    /// Number of total violations across all regulations.
    total_violations: usize,
    /// Audit event ID for this check.
    audit_event_id: ?u64,

    pub fn deinit(self: *ComplianceCheckResult) void {
        if (self.gdpr_result) |*r| r.deinit();
        if (self.hipaa_result) |*r| r.deinit();
        if (self.ccpa_result) |*r| r.deinit();
    }
};

/// The Compliance Engine — orchestrates all regulatory checks.
pub const ComplianceEngine = struct {
    allocator: std.mem.Allocator,
    cfg: ComplianceConfig,
    gdpr_checker: GdprChecker,
    hipaa_checker: HipaaChecker,
    ccpa_checker: CcpaChecker,
    audit_trail: ?*AuditTrail,

    const Self = @This();

    pub fn init(allocator: std.mem.Allocator, compliance_cfg: ComplianceConfig) !*Self {
        const self = try allocator.create(Self);
        errdefer allocator.destroy(self);

        var trail: ?*AuditTrail = null;
        if (compliance_cfg.enable_audit) {
            trail = try AuditTrail.init(allocator, compliance_cfg.audit.max_events);
        }

        self.* = .{
            .allocator = allocator,
            .cfg = compliance_cfg,
            .gdpr_checker = GdprChecker.init(compliance_cfg.pii),
            .hipaa_checker = HipaaChecker.init(),
            .ccpa_checker = CcpaChecker.init(),
            .audit_trail = trail,
        };
        return self;
    }

    pub fn deinit(self: *Self) void {
        if (self.audit_trail) |trail| trail.deinit(self.allocator);
        self.allocator.destroy(self);
    }

    /// Run all enabled compliance checks on content.
    pub fn checkContent(self: *Self, content: []const u8, subject_id: []const u8) !ComplianceCheckResult {
        var gdpr_result: ?gdpr.GdprCheckResult = null;
        var hipaa_result: ?hipaa.HipaaCheckResult = null;
        var ccpa_result: ?ccpa.CcpaCheckResult = null;
        var total_violations: usize = 0;
        var overall_compliant = true;

        errdefer {
            if (gdpr_result) |*r| r.deinit();
            if (hipaa_result) |*r| r.deinit();
            if (ccpa_result) |*r| r.deinit();
        }

        if (self.cfg.enable_gdpr) {
            var result = try self.gdpr_checker.check(self.allocator, content);
            if (!result.is_compliant) {
                overall_compliant = false;
                total_violations += result.violations.len;
                if (self.audit_trail) |trail| {
                    _ = trail.record(.pii_detected, .warning, subject_id, "GDPR: PII detected in content", .gdpr);
                }
            }
            gdpr_result = result;
        }

        if (self.cfg.enable_hipaa) {
            var result = try self.hipaa_checker.check(self.allocator, content);
            if (!result.is_compliant) {
                overall_compliant = false;
                total_violations += result.violations.len;
                if (self.audit_trail) |trail| {
                    _ = trail.record(.phi_detected, .warning, subject_id, "HIPAA: PHI detected in content", .hipaa);
                }
            }
            hipaa_result = result;
        }

        if (self.cfg.enable_ccpa) {
            var result = try self.ccpa_checker.check(self.allocator, content);
            if (!result.is_compliant) {
                overall_compliant = false;
                total_violations += result.violations.len;
                if (self.audit_trail) |trail| {
                    _ = trail.record(.pii_detected, .warning, subject_id, "CCPA: Personal info detected", .ccpa);
                }
            }
            ccpa_result = result;
        }

        // Record the overall check
        var audit_id: ?u64 = null;
        if (self.audit_trail) |trail| {
            const severity: AuditSeverity = if (overall_compliant) .info else .warning;
            audit_id = trail.record(.policy_check, severity, subject_id, "Compliance check completed", .general);
        }

        // In strict mode, non-compliance is escalated
        if (!overall_compliant and self.cfg.strictness == .strict) {
            if (self.audit_trail) |trail| {
                _ = trail.record(.violation_detected, .critical, subject_id, "Strict mode: compliance violation escalated", .general);
            }
        }

        return .{
            .is_compliant = overall_compliant,
            .gdpr_result = gdpr_result,
            .hipaa_result = hipaa_result,
            .ccpa_result = ccpa_result,
            .total_violations = total_violations,
            .audit_event_id = audit_id,
        };
    }

    /// Get the audit trail (null if auditing is disabled).
    pub fn getAuditTrail(self: *Self) ?*AuditTrail {
        return self.audit_trail;
    }

    /// Check if a specific regulation is enabled.
    pub fn isRegulationEnabled(self: *const Self, regulation: AuditEvent.Regulation) bool {
        return switch (regulation) {
            .gdpr => self.cfg.enable_gdpr,
            .hipaa => self.cfg.enable_hipaa,
            .ccpa => self.cfg.enable_ccpa,
            .general, .none => true,
        };
    }
};

// ============================================================================
// Tests
// ============================================================================

test "ComplianceEngine init and deinit" {
    const allocator = std.testing.allocator;
    const engine = try ComplianceEngine.init(allocator, .{});
    defer engine.deinit();

    try std.testing.expect(engine.audit_trail != null);
}

test "ComplianceEngine checks clean content" {
    const allocator = std.testing.allocator;
    const engine = try ComplianceEngine.init(allocator, .{});
    defer engine.deinit();

    var result = try engine.checkContent("Hello world", "test-user");
    defer result.deinit();

    try std.testing.expect(result.is_compliant);
    try std.testing.expect(result.total_violations == 0);
    try std.testing.expect(result.audit_event_id != null);
}

test "ComplianceEngine detects PII with GDPR" {
    const allocator = std.testing.allocator;
    const engine = try ComplianceEngine.init(allocator, .{ .enable_gdpr = true });
    defer engine.deinit();

    var result = try engine.checkContent("Email me at test@example.com", "user-1");
    defer result.deinit();

    try std.testing.expect(!result.is_compliant);
    try std.testing.expect(result.gdpr_result != null);
    try std.testing.expect(result.total_violations > 0);
}

test "ComplianceEngine multi-regulation check" {
    const allocator = std.testing.allocator;
    const engine = try ComplianceEngine.init(allocator, .{
        .enable_gdpr = true,
        .enable_hipaa = true,
        .enable_ccpa = true,
    });
    defer engine.deinit();

    var result = try engine.checkContent("The patient gave their email john@hospital.com for diagnosis", "session-1");
    defer result.deinit();

    try std.testing.expect(!result.is_compliant);
    try std.testing.expect(result.gdpr_result != null);
    try std.testing.expect(result.hipaa_result != null);
    try std.testing.expect(result.ccpa_result != null);
}

test "ComplianceEngine regulation enabled check" {
    const engine_ptr = try ComplianceEngine.init(std.testing.allocator, .{
        .enable_gdpr = true,
        .enable_hipaa = false,
    });
    defer engine_ptr.deinit();

    try std.testing.expect(engine_ptr.isRegulationEnabled(.gdpr));
    try std.testing.expect(!engine_ptr.isRegulationEnabled(.hipaa));
    try std.testing.expect(engine_ptr.isRegulationEnabled(.general));
}

test {
    std.testing.refAllDecls(@This());
}
