//! Configuration types for the Compliance Module.
//!
//! Defines settings for GDPR, HIPAA, CCPA, and audit trail behavior.

const std = @import("std");

/// Top-level compliance configuration.
pub const ComplianceConfig = struct {
    /// Enable GDPR checks.
    enable_gdpr: bool = true,
    /// Enable HIPAA checks.
    enable_hipaa: bool = false,
    /// Enable CCPA checks.
    enable_ccpa: bool = false,
    /// Enable audit trail logging.
    enable_audit: bool = true,
    /// Overall strictness level for compliance enforcement.
    strictness: StrictnessLevel = .standard,
    /// PII detection settings.
    pii: PiiDetectionConfig = .{},
    /// Audit trail settings.
    audit: AuditConfig = .{},

    pub const StrictnessLevel = enum {
        /// Log warnings but allow processing.
        permissive,
        /// Block non-compliant content; default.
        standard,
        /// Block and escalate; strictest mode.
        strict,
    };

    pub fn defaults() ComplianceConfig {
        return .{};
    }
};

/// Settings for PII (Personally Identifiable Information) detection.
pub const PiiDetectionConfig = struct {
    /// Detect email addresses.
    detect_email: bool = true,
    /// Detect phone numbers.
    detect_phone: bool = true,
    /// Detect Social Security Numbers.
    detect_ssn: bool = true,
    /// Detect IP addresses.
    detect_ip: bool = true,
    /// Detect credit card numbers.
    detect_credit_card: bool = true,
    /// Detect dates of birth.
    detect_dob: bool = true,
    /// Minimum pattern length to consider as PII.
    min_pattern_length: u32 = 5,
};

/// Settings for the audit trail subsystem.
pub const AuditConfig = struct {
    /// Maximum number of audit events to retain in the ring buffer.
    max_events: u32 = 10_000,
    /// Whether to include full content in audit logs (may contain PII).
    log_content: bool = false,
    /// Retention period in seconds (0 = unlimited).
    retention_seconds: u64 = 0,
};

test {
    std.testing.refAllDecls(@This());
}

test "ComplianceConfig defaults" {
    const cfg = ComplianceConfig.defaults();
    try std.testing.expect(cfg.enable_gdpr);
    try std.testing.expect(!cfg.enable_hipaa);
    try std.testing.expect(cfg.enable_audit);
    try std.testing.expect(cfg.strictness == .standard);
}
