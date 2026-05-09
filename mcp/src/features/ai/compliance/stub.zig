//! Compliance Module stub — disabled at compile time.

const std = @import("std");

// ── Base types ─────────────────────────────────────────────────────────────

const _StrictnessLevel = enum { permissive, standard, strict };
const _PiiDetectionConfig = struct {
    detect_email: bool = true,
    detect_phone: bool = true,
    detect_ssn: bool = true,
    detect_ip: bool = true,
    detect_credit_card: bool = true,
    detect_dob: bool = true,
    min_pattern_length: u32 = 5,
};
const _AuditConfig = struct { max_events: u32 = 10_000, log_content: bool = false, retention_seconds: u64 = 0 };
const _ComplianceConfig = struct {
    enable_gdpr: bool = true,
    enable_hipaa: bool = false,
    enable_ccpa: bool = false,
    enable_audit: bool = true,
    strictness: _StrictnessLevel = .standard,
    pii: _PiiDetectionConfig = .{},
    audit: _AuditConfig = .{},
    pub const StrictnessLevel = _StrictnessLevel;
    pub fn defaults() _ComplianceConfig {
        return .{};
    }
};

const _PiiType = enum { email, phone, ssn, ip_address, credit_card, date_of_birth, name_pattern };
const _PiiMatch = struct { pii_type: _PiiType, start: usize, length: usize, confidence: f32 };
const _ConsentStatus = enum { unknown, granted, revoked, expired };
const _ConsentRecord = struct {
    user_id: []const u8,
    status: _ConsentStatus,
    purpose: []const u8,
    granted_at: i64,
    expires_at: ?i64,
    pub fn isValid(self: @This(), now: i64) bool {
        if (self.status != .granted) return false;
        if (self.expires_at) |exp| return now < exp;
        return true;
    }
};
const _GdprCheckResult = struct {
    is_compliant: bool,
    pii_detected: []const _PiiMatch,
    data_minimized: bool,
    consent_verified: bool,
    violations: []const []const u8,
    allocator: std.mem.Allocator,
    pub fn deinit(_: *@This()) void {}
};
const _GdprChecker = struct {
    pii_config: _PiiDetectionConfig,
    pub fn init(pii_cfg: _PiiDetectionConfig) @This() {
        return .{ .pii_config = pii_cfg };
    }
    pub fn check(_: *const @This(), _: std.mem.Allocator, _: []const u8) !_GdprCheckResult {
        return error.FeatureDisabled;
    }
};

const _PhiCategory = enum {
    patient_name,
    medical_record_number,
    dates,
    geographic,
    phone_fax,
    email,
    ssn,
    beneficiary_number,
    account_number,
    certificate_number,
    device_identifier,
    biometric,
    photographs,
    other_unique_id,
    diagnosis,
    medication,
    lab_result,
};
const _PhiMatch = struct { category: _PhiCategory, start: usize, length: usize, confidence: f32, description: []const u8 };
const _AccessLevel = enum {
    none,
    de_identified,
    limited,
    full,
    pub fn canAccess(self: @This(), required: @This()) bool {
        return @intFromEnum(self) >= @intFromEnum(required);
    }
};
const _DeIdentificationStatus = enum { not_applicable, safe_harbor_compliant, requires_de_identification };
const _HipaaCheckResult = struct {
    is_compliant: bool,
    phi_detected: []const _PhiMatch,
    required_access_level: _AccessLevel,
    de_identification_status: _DeIdentificationStatus,
    violations: []const []const u8,
    allocator: std.mem.Allocator,
    pub fn deinit(_: *@This()) void {}
};
const _HipaaChecker = struct {
    pub fn init() @This() {
        return .{};
    }
    pub fn check(_: *const @This(), _: std.mem.Allocator, _: []const u8) !_HipaaCheckResult {
        return error.FeatureDisabled;
    }
};

const _ConsumerRight = enum { right_to_know, right_to_delete, right_to_opt_out, right_to_non_discrimination, right_to_correct, right_to_limit };
const _OptOutStatus = enum { not_set, opted_out, opted_in };
const _RequestStatus = enum { pending, in_progress, completed, denied };
const _ConsumerRequest = struct {
    consumer_id: []const u8,
    right: _ConsumerRight,
    status: _RequestStatus,
    submitted_at: i64,
    responded_at: ?i64,
    pub const RequestStatus = _RequestStatus;
    pub fn isOverdue(self: @This(), now: i64) bool {
        return self.status != .completed and self.status != .denied and now > self.submitted_at + 3_888_000;
    }
};
const _ProfilelInfoCategory = enum { identifiers, commercial_info, biometric, internet_activity, geolocation, professional_info, education_info, inferences, sensitive_pi };
const _CcpaCheckResult = struct {
    is_compliant: bool,
    profilel_info_detected: []const _ProfilelInfoCategory,
    opt_out_honored: bool,
    disclosure_adequate: bool,
    violations: []const []const u8,
    allocator: std.mem.Allocator,
    pub fn deinit(_: *@This()) void {}
};
const _CcpaChecker = struct {
    pub fn init() @This() {
        return .{};
    }
    pub fn check(_: *const @This(), _: std.mem.Allocator, _: []const u8) !_CcpaCheckResult {
        return error.FeatureDisabled;
    }
};

const _AuditEventType = enum {
    consent_granted,
    consent_revoked,
    data_accessed,
    erasure_requested,
    erasure_completed,
    violation_detected,
    pii_detected,
    phi_detected,
    policy_check,
    routing_decision,
    data_export,
    access_denied,
};
const _AuditSeverity = enum { info, warning, err, critical };
const _AuditEvent = struct {
    id: u64,
    event_type: _AuditEventType,
    severity: _AuditSeverity,
    timestamp: i64,
    subject_id: [64]u8,
    subject_id_len: u8,
    description: [256]u8,
    description_len: u16,
    regulation: Regulation,
    pub const Regulation = enum { none, gdpr, hipaa, ccpa, general };
    pub fn getSubjectId(self: *const @This()) []const u8 {
        return self.subject_id[0..self.subject_id_len];
    }
    pub fn getDescription(self: *const @This()) []const u8 {
        return self.description[0..self.description_len];
    }
};
const _AuditTrail = struct {
    events: []_AuditEvent,
    head: usize,
    count: usize,
    next_id: u64,
    mutex: u8,
    pub fn init(_: std.mem.Allocator, _: u32) !*_AuditTrail {
        return error.FeatureDisabled;
    }
    pub fn deinit(_: *_AuditTrail, _: std.mem.Allocator) void {}
    pub fn record(_: *_AuditTrail, _: _AuditEventType, _: _AuditSeverity, _: []const u8, _: []const u8, _: _AuditEvent.Regulation) u64 {
        return 0;
    }
    pub fn eventCount(_: *_AuditTrail) usize {
        return 0;
    }
    pub fn getRecent(_: *_AuditTrail, _: std.mem.Allocator, _: usize) ![]const _AuditEvent {
        return error.FeatureDisabled;
    }
    pub fn queryByType(_: *_AuditTrail, _: std.mem.Allocator, _: _AuditEventType) ![]const _AuditEvent {
        return error.FeatureDisabled;
    }
    pub fn countBySeverity(_: *_AuditTrail, _: _AuditSeverity) usize {
        return 0;
    }
};

const _ComplianceCheckResult = struct {
    is_compliant: bool,
    gdpr_result: ?_GdprCheckResult,
    hipaa_result: ?_HipaaCheckResult,
    ccpa_result: ?_CcpaCheckResult,
    total_violations: usize,
    audit_event_id: ?u64,
    pub fn deinit(_: *@This()) void {}
};

// ── Sub-module stubs (re-export base types) ────────────────────────────────

pub const config = struct {
    pub const ComplianceConfig = _ComplianceConfig;
    pub const PiiDetectionConfig = _PiiDetectionConfig;
    pub const AuditConfig = _AuditConfig;
};

pub const gdpr = struct {
    pub const PiiType = _PiiType;
    pub const PiiMatch = _PiiMatch;
    pub const ConsentStatus = _ConsentStatus;
    pub const ConsentRecord = _ConsentRecord;
    pub const GdprChecker = _GdprChecker;
    pub const GdprCheckResult = _GdprCheckResult;
};

pub const hipaa = struct {
    pub const PhiCategory = _PhiCategory;
    pub const PhiMatch = _PhiMatch;
    pub const AccessLevel = _AccessLevel;
    pub const HipaaChecker = _HipaaChecker;
    pub const HipaaCheckResult = _HipaaCheckResult;
};

pub const ccpa = struct {
    pub const ConsumerRight = _ConsumerRight;
    pub const OptOutStatus = _OptOutStatus;
    pub const ConsumerRequest = _ConsumerRequest;
    pub const ProfilelInfoCategory = _ProfilelInfoCategory;
    pub const CcpaChecker = _CcpaChecker;
    pub const CcpaCheckResult = _CcpaCheckResult;
};

pub const audit = struct {
    pub const AuditEventType = _AuditEventType;
    pub const AuditSeverity = _AuditSeverity;
    pub const AuditEvent = _AuditEvent;
    pub const AuditTrail = _AuditTrail;
};

// ── Top-level re-exports (match mod.zig) ───────────────────────────────────

pub const ComplianceConfig = _ComplianceConfig;
pub const GdprChecker = _GdprChecker;
pub const GdprCheckResult = _GdprCheckResult;
pub const PiiType = _PiiType;
pub const PiiMatch = _PiiMatch;
pub const ConsentStatus = _ConsentStatus;
pub const ConsentRecord = _ConsentRecord;
pub const HipaaChecker = _HipaaChecker;
pub const HipaaCheckResult = _HipaaCheckResult;
pub const PhiCategory = _PhiCategory;
pub const PhiMatch = _PhiMatch;
pub const AccessLevel = _AccessLevel;
pub const CcpaChecker = _CcpaChecker;
pub const CcpaCheckResult = _CcpaCheckResult;
pub const ConsumerRight = _ConsumerRight;
pub const ProfilelInfoCategory = _ProfilelInfoCategory;
pub const AuditTrail = _AuditTrail;
pub const AuditEvent = _AuditEvent;
pub const AuditEventType = _AuditEventType;
pub const AuditSeverity = _AuditSeverity;
pub const ComplianceCheckResult = _ComplianceCheckResult;

// ── ComplianceEngine stub ──────────────────────────────────────────────────

pub const ComplianceEngine = struct {
    allocator: std.mem.Allocator,
    cfg: _ComplianceConfig,
    gdpr_checker: _GdprChecker,
    hipaa_checker: _HipaaChecker,
    ccpa_checker: _CcpaChecker,
    audit_trail: ?*_AuditTrail,

    const Self = @This();

    pub fn init(_: std.mem.Allocator, _: _ComplianceConfig) !*Self {
        return error.FeatureDisabled;
    }
    pub fn deinit(_: *Self) void {}
    pub fn checkContent(_: *Self, _: []const u8, _: []const u8) !_ComplianceCheckResult {
        return error.FeatureDisabled;
    }
    pub fn getAuditTrail(_: *Self) ?*_AuditTrail {
        return null;
    }
    pub fn isRegulationEnabled(_: *const Self, _: _AuditEvent.Regulation) bool {
        return false;
    }
};

test {
    std.testing.refAllDecls(@This());
}
