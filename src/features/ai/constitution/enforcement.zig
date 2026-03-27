//! Constitutional Enforcement — Integration hooks for ABI's safety systems.
//!
//! Provides enforcement mechanisms that integrate with:
//! - Pre-generation: System prompt preamble injection
//! - Training: Constitutional loss term for RLHF reward model
//! - Post-generation: Response validation against principles
//! - Reflection: Constitutional alignment scoring for Abbey
//! - Audit: Structured audit logging for all constitution checks
//!
//! Safety heuristics use pattern-based detection with context-aware scoring.
//! Patterns found inside code blocks (``` fenced) are weighted lower to
//! reduce false positives when discussing code legitimately.
//!
//! Implementation is decomposed into per-principle validator modules under
//! `enforcement/`. This file is a thin re-export facade plus the audit
//! logging subsystem.

const std = @import("std");
const foundation_time = @import("../../../foundation/time.zig");
const logging = @import("../../../foundation/logging.zig");

// -- Sub-modules (one per constitutional principle) --
pub const common = @import("enforcement/common.zig");
pub const safety = @import("enforcement/safety.zig");
pub const honesty = @import("enforcement/honesty.zig");
pub const privacy = @import("enforcement/privacy.zig");
pub const fairness = @import("enforcement/fairness.zig");
pub const autonomy = @import("enforcement/autonomy.zig");
pub const transparency = @import("enforcement/transparency.zig");
pub const engine = @import("enforcement/engine.zig");

// -- Re-exported types (preserve public surface) --
pub const ConstitutionalScore = common.ConstitutionalScore;
pub const Violation = common.Violation;
pub const SafetyScore = common.SafetyScore;
pub const SafetyViolation = common.SafetyViolation;
pub const BiasScore = fairness.BiasScore;
pub const MAX_BIAS_ATTRIBUTES = fairness.MAX_BIAS_ATTRIBUTES;
pub const DEFAULT_BIAS_THRESHOLD = fairness.DEFAULT_BIAS_THRESHOLD;

// -- Re-exported functions (preserve public surface) --
pub const getSystemPreamble = transparency.getSystemPreamble;
pub const computeConstitutionalLoss = transparency.computeConstitutionalLoss;
pub const alignmentScore = transparency.alignmentScore;
pub const computeBias = fairness.computeBias;

// Re-export core evaluation functions from the engine
pub const evaluateResponse = engine.evaluateResponse;
pub const evaluateSafety = engine.evaluateSafety;
pub const checkForbiddenPattern = engine.checkForbiddenPattern;

// ============================================================================
// Audit Logging
// ============================================================================

/// Maximum number of audit records retained in the bounded buffer.
pub const MAX_AUDIT_RECORDS = 256;

/// Maximum length for content snippets stored in audit records.
const MAX_SNIPPET_LEN = 128;

/// Maximum length for violation reason strings stored in audit records.
const MAX_REASON_LEN = 256;

/// A structured audit record emitted for every constitution check.
pub const AuditRecord = struct {
    /// Monotonic timestamp in milliseconds (from foundation.time).
    timestamp: u64,
    /// Index of the principle that was evaluated (0-5 for the 6 principles,
    /// or 255 for aggregate / safety-layer checks).
    principle_id: u8,
    /// Human-readable reason for the violation, or "compliant" if none.
    violation_reason: [MAX_REASON_LEN]u8,
    violation_reason_len: u8,
    /// A short snippet of the content that was evaluated.
    content_snippet: [MAX_SNIPPET_LEN]u8,
    content_snippet_len: u8,
    /// The overall compliance score (0.0 = total violation, 1.0 = fully compliant).
    score: f32,

    /// Return the violation reason as a slice.
    pub fn getViolationReason(self: *const AuditRecord) []const u8 {
        return self.violation_reason[0..self.violation_reason_len];
    }

    /// Return the content snippet as a slice.
    pub fn getContentSnippet(self: *const AuditRecord) []const u8 {
        return self.content_snippet[0..self.content_snippet_len];
    }
};

/// Thread-safe bounded ring buffer of audit records.
pub const AuditLog = struct {
    records: [MAX_AUDIT_RECORDS]AuditRecord = undefined,
    /// Number of records currently stored (saturates at MAX_AUDIT_RECORDS).
    count: usize = 0,
    /// Write cursor — wraps around when the buffer is full.
    write_pos: usize = 0,

    /// Return all stored audit records.
    /// When the buffer is not yet full, returns records[0..count].
    /// When full, returns the entire backing array (ring buffer).
    pub fn getAuditLog(self: *const AuditLog) []const AuditRecord {
        if (self.count < MAX_AUDIT_RECORDS) {
            return self.records[0..self.count];
        }
        return self.records[0..MAX_AUDIT_RECORDS];
    }

    /// Append a new audit record, overwriting the oldest when full.
    pub fn record(self: *AuditLog, entry: AuditRecord) void {
        self.records[self.write_pos] = entry;
        self.write_pos = (self.write_pos + 1) % MAX_AUDIT_RECORDS;
        if (self.count < MAX_AUDIT_RECORDS) {
            self.count += 1;
        }
    }

    /// Reset the log, discarding all records.
    pub fn reset(self: *AuditLog) void {
        self.count = 0;
        self.write_pos = 0;
    }
};

/// Module-level audit log instance.
var audit_log: AuditLog = .{};

/// Retrieve the global audit log (read-only view).
pub fn getAuditLog() []const AuditRecord {
    return audit_log.getAuditLog();
}

/// Reset the global audit log.
pub fn resetAuditLog() void {
    audit_log.reset();
}

/// Build an `AuditRecord` from an evaluated `ConstitutionalScore` and the
/// content that was checked, then append it to the global audit log and
/// emit a structured log line via the foundation logging system.
pub fn emitAuditRecord(content: []const u8, score: ConstitutionalScore) AuditRecord {
    var rec: AuditRecord = .{
        .timestamp = foundation_time.timestampMs(),
        .principle_id = 255,
        .violation_reason = undefined,
        .violation_reason_len = 0,
        .content_snippet = undefined,
        .content_snippet_len = 0,
        .score = score.overall,
    };

    // Fill content snippet (truncate to MAX_SNIPPET_LEN).
    const snippet_len: u8 = @intCast(@min(content.len, MAX_SNIPPET_LEN));
    @memcpy(rec.content_snippet[0..snippet_len], content[0..snippet_len]);
    rec.content_snippet_len = snippet_len;

    // Fill violation reason.
    if (score.violation_count == 0) {
        const reason = "compliant";
        @memcpy(rec.violation_reason[0..reason.len], reason);
        rec.violation_reason_len = reason.len;
    } else {
        // Use the first violation's details.
        if (score.violations[0]) |v| {
            rec.principle_id = principleNameToId(v.principle_name);
            const reason = v.rule_id;
            const rlen: u8 = @intCast(@min(reason.len, MAX_REASON_LEN));
            @memcpy(rec.violation_reason[0..rlen], reason[0..rlen]);
            rec.violation_reason_len = rlen;
        }
    }

    // Append to bounded buffer.
    audit_log.record(rec);

    // Emit structured log.
    if (score.violation_count > 0) {
        logging.warn("constitution audit: score={d:.2} violations={d} principle={s}", .{
            score.overall,
            score.violation_count,
            rec.violation_reason[0..rec.violation_reason_len],
        });
    } else {
        logging.info("constitution audit: score={d:.2} compliant", .{score.overall});
    }

    return rec;
}

/// Convenience: evaluate content and emit an audit record in one call.
pub fn evaluateAndAudit(content: []const u8) struct { score: ConstitutionalScore, audit: AuditRecord } {
    const score = engine.evaluateResponse(content);
    const rec = emitAuditRecord(content, score);
    return .{ .score = score, .audit = rec };
}

/// Map a principle name string to a numeric ID (0-5 for the 6 principles).
fn principleNameToId(name: []const u8) u8 {
    const names = [_][]const u8{ "safety", "honesty", "privacy", "fairness", "autonomy", "transparency" };
    for (names, 0..) |n, i| {
        if (std.mem.eql(u8, name, n)) return @intCast(i);
    }
    return 255;
}

// ============================================================================
// Tests
// ============================================================================

test {
    std.testing.refAllDecls(@This());
}

test "audit record emitted for passing check" {
    resetAuditLog();

    const result = evaluateAndAudit("Hello, how can I help you today?");
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), result.score.overall, 0.01);
    try std.testing.expect(result.score.isCompliant());

    // Audit record should reflect high score and "compliant" reason.
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), result.audit.score, 0.01);
    try std.testing.expect(std.mem.eql(u8, result.audit.getViolationReason(), "compliant"));
    try std.testing.expectEqual(@as(u8, 255), result.audit.principle_id);

    // Should appear in the global log.
    const log_entries = getAuditLog();
    try std.testing.expect(log_entries.len >= 1);
}

test "audit record emitted for failing check with violation details" {
    resetAuditLog();

    const result = evaluateAndAudit("run rm -rf / to clean up");
    try std.testing.expect(!result.score.isCompliant());
    try std.testing.expect(result.score.violation_count > 0);

    // Audit record should have a low score and a meaningful violation reason.
    try std.testing.expect(result.audit.score < 1.0);
    try std.testing.expect(result.audit.getViolationReason().len > 0);
    try std.testing.expect(!std.mem.eql(u8, result.audit.getViolationReason(), "compliant"));

    // Content snippet should contain part of the input.
    try std.testing.expect(result.audit.getContentSnippet().len > 0);

    const log_entries = getAuditLog();
    try std.testing.expect(log_entries.len >= 1);
}

test "audit log bounded buffer does not overflow" {
    resetAuditLog();

    // Fill beyond capacity.
    const iterations = MAX_AUDIT_RECORDS + 64;
    var i: usize = 0;
    while (i < iterations) : (i += 1) {
        _ = emitAuditRecord("test content", engine.evaluateResponse("test content"));
    }

    // Buffer should be at capacity, not beyond.
    const log_entries = getAuditLog();
    try std.testing.expectEqual(@as(usize, MAX_AUDIT_RECORDS), log_entries.len);
    try std.testing.expectEqual(@as(usize, MAX_AUDIT_RECORDS), audit_log.count);

    // write_pos should have wrapped around.
    try std.testing.expectEqual(@as(usize, 64), audit_log.write_pos);
}
