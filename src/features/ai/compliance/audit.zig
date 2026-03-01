//! Audit Trail Module
//!
//! Provides event logging for compliance and accountability:
//! - Ring-buffer based event storage
//! - Typed audit events (consent, access, erasure, violation)
//! - Timestamp-based querying
//! - Export capabilities

const std = @import("std");
const config = @import("config.zig");

/// Types of audit events.
pub const AuditEventType = enum {
    /// User consent granted or updated.
    consent_granted,
    /// User consent revoked.
    consent_revoked,
    /// Personal data accessed.
    data_accessed,
    /// Data erasure requested.
    erasure_requested,
    /// Data erasure completed.
    erasure_completed,
    /// Compliance violation detected.
    violation_detected,
    /// PII detected in content.
    pii_detected,
    /// PHI detected in content.
    phi_detected,
    /// Policy check performed.
    policy_check,
    /// Persona routing decision made.
    routing_decision,
    /// Data export requested.
    data_export,
    /// Access denied due to insufficient permissions.
    access_denied,
};

/// Severity level for audit events.
pub const AuditSeverity = enum {
    /// Informational event.
    info,
    /// Warning: potential compliance issue.
    warning,
    /// Error: compliance violation detected.
    err,
    /// Critical: requires immediate attention.
    critical,
};

/// A single audit event.
pub const AuditEvent = struct {
    /// Monotonic event ID.
    id: u64,
    /// Event type.
    event_type: AuditEventType,
    /// Severity level.
    severity: AuditSeverity,
    /// Timestamp (Unix seconds).
    timestamp: i64,
    /// Associated user or session ID.
    subject_id: [64]u8,
    subject_id_len: u8,
    /// Short description of the event.
    description: [256]u8,
    description_len: u16,
    /// Regulation this event relates to.
    regulation: Regulation,

    pub const Regulation = enum {
        none,
        gdpr,
        hipaa,
        ccpa,
        general,
    };

    /// Get the subject ID as a slice.
    pub fn getSubjectId(self: *const AuditEvent) []const u8 {
        return self.subject_id[0..self.subject_id_len];
    }

    /// Get the description as a slice.
    pub fn getDescription(self: *const AuditEvent) []const u8 {
        return self.description[0..self.description_len];
    }
};

/// Thread-safe audit trail with ring-buffer storage.
pub const AuditTrail = struct {
    events: []AuditEvent,
    head: usize,
    count: usize,
    next_id: u64,
    mutex: std.Thread.Mutex,

    const Self = @This();

    pub fn init(allocator: std.mem.Allocator, max_events: u32) !*Self {
        const self = try allocator.create(Self);
        errdefer allocator.destroy(self);

        const cap = if (max_events == 0) 1000 else max_events;
        const events = try allocator.alloc(AuditEvent, cap);

        self.* = .{
            .events = events,
            .head = 0,
            .count = 0,
            .next_id = 1,
            .mutex = .{},
        };
        return self;
    }

    pub fn deinit(self: *Self, allocator: std.mem.Allocator) void {
        allocator.free(self.events);
        allocator.destroy(self);
    }

    /// Record a new audit event.
    pub fn record(
        self: *Self,
        event_type: AuditEventType,
        severity: AuditSeverity,
        subject_id: []const u8,
        description: []const u8,
        regulation: AuditEvent.Regulation,
    ) u64 {
        self.mutex.lock();
        defer self.mutex.unlock();

        const id = self.next_id;
        self.next_id += 1;

        var event = AuditEvent{
            .id = id,
            .event_type = event_type,
            .severity = severity,
            .timestamp = std.time.timestamp(),
            .subject_id = undefined,
            .subject_id_len = 0,
            .description = undefined,
            .description_len = 0,
            .regulation = regulation,
        };

        // Copy subject_id (truncate if too long)
        const sid_len: u8 = @intCast(@min(subject_id.len, 64));
        @memcpy(event.subject_id[0..sid_len], subject_id[0..sid_len]);
        event.subject_id_len = sid_len;

        // Copy description (truncate if too long)
        const desc_len: u16 = @intCast(@min(description.len, 256));
        @memcpy(event.description[0..desc_len], description[0..desc_len]);
        event.description_len = desc_len;

        // Ring buffer insertion
        self.events[self.head] = event;
        self.head = (self.head + 1) % self.events.len;
        if (self.count < self.events.len) self.count += 1;

        return id;
    }

    /// Get the total number of recorded events (may be less than total if ring buffer wrapped).
    pub fn eventCount(self: *Self) usize {
        self.mutex.lock();
        defer self.mutex.unlock();
        return self.count;
    }

    /// Get the most recent N events (newest first).
    pub fn getRecent(self: *Self, allocator: std.mem.Allocator, max_count: usize) ![]const AuditEvent {
        self.mutex.lock();
        defer self.mutex.unlock();

        const n = @min(max_count, self.count);
        var result = try allocator.alloc(AuditEvent, n);
        errdefer allocator.free(result);

        var idx: usize = 0;
        while (idx < n) : (idx += 1) {
            const pos = if (self.head >= idx + 1)
                self.head - idx - 1
            else
                self.events.len - (idx + 1 - self.head);
            result[idx] = self.events[pos];
        }

        return result;
    }

    /// Query events by type.
    pub fn queryByType(self: *Self, allocator: std.mem.Allocator, event_type: AuditEventType) ![]const AuditEvent {
        self.mutex.lock();
        defer self.mutex.unlock();

        var matches = std.ArrayListUnmanaged(AuditEvent).empty;
        errdefer matches.deinit(allocator);

        const start = if (self.count < self.events.len) 0 else self.head;
        var i: usize = 0;
        while (i < self.count) : (i += 1) {
            const pos = (start + i) % self.events.len;
            if (self.events[pos].event_type == event_type) {
                try matches.append(allocator, self.events[pos]);
            }
        }

        return matches.toOwnedSlice(allocator);
    }

    /// Count events by severity level.
    pub fn countBySeverity(self: *Self, severity: AuditSeverity) usize {
        self.mutex.lock();
        defer self.mutex.unlock();

        var count: usize = 0;
        const start = if (self.count < self.events.len) 0 else self.head;
        var i: usize = 0;
        while (i < self.count) : (i += 1) {
            const pos = (start + i) % self.events.len;
            if (self.events[pos].severity == severity) count += 1;
        }
        return count;
    }
};

// ============================================================================
// Tests
// ============================================================================

test "AuditTrail record and retrieve" {
    const allocator = std.testing.allocator;
    var trail = try AuditTrail.init(allocator, 100);
    defer trail.deinit(allocator);

    const id1 = trail.record(.consent_granted, .info, "user-1", "User granted consent", .gdpr);
    const id2 = trail.record(.pii_detected, .warning, "session-42", "Email found in content", .gdpr);

    try std.testing.expect(id1 == 1);
    try std.testing.expect(id2 == 2);
    try std.testing.expect(trail.eventCount() == 2);

    const recent = try trail.getRecent(allocator, 10);
    defer allocator.free(recent);
    try std.testing.expect(recent.len == 2);
    // Most recent first
    try std.testing.expect(recent[0].id == 2);
    try std.testing.expect(recent[1].id == 1);
}

test "AuditTrail ring buffer wraps" {
    const allocator = std.testing.allocator;
    var trail = try AuditTrail.init(allocator, 3);
    defer trail.deinit(allocator);

    _ = trail.record(.policy_check, .info, "a", "check 1", .general);
    _ = trail.record(.policy_check, .info, "b", "check 2", .general);
    _ = trail.record(.policy_check, .info, "c", "check 3", .general);
    _ = trail.record(.violation_detected, .err, "d", "check 4", .general);

    // Should have wrapped; count stays at 3
    try std.testing.expect(trail.eventCount() == 3);

    const recent = try trail.getRecent(allocator, 10);
    defer allocator.free(recent);
    try std.testing.expect(recent.len == 3);
    // Most recent should have id 4
    try std.testing.expect(recent[0].id == 4);
}

test "AuditTrail query by type" {
    const allocator = std.testing.allocator;
    var trail = try AuditTrail.init(allocator, 100);
    defer trail.deinit(allocator);

    _ = trail.record(.consent_granted, .info, "u1", "consent", .gdpr);
    _ = trail.record(.pii_detected, .warning, "u2", "pii", .gdpr);
    _ = trail.record(.consent_granted, .info, "u3", "consent", .gdpr);

    const consents = try trail.queryByType(allocator, .consent_granted);
    defer allocator.free(consents);
    try std.testing.expect(consents.len == 2);
}

test "AuditTrail count by severity" {
    const allocator = std.testing.allocator;
    var trail = try AuditTrail.init(allocator, 100);
    defer trail.deinit(allocator);

    _ = trail.record(.policy_check, .info, "a", "ok", .general);
    _ = trail.record(.violation_detected, .err, "b", "bad", .general);
    _ = trail.record(.violation_detected, .err, "c", "bad", .general);

    try std.testing.expect(trail.countBySeverity(.info) == 1);
    try std.testing.expect(trail.countBySeverity(.err) == 2);
    try std.testing.expect(trail.countBySeverity(.critical) == 0);
}

test {
    std.testing.refAllDecls(@This());
}
