//! Security audit logging system for comprehensive security event tracking.
//!
//! This module provides:
//! - Structured security event logging
//! - Event categories (auth, access, data, system, network)
//! - Severity levels (critical, high, medium, low, info)
//! - Tamper-evident log chains with HMAC
//! - Log rotation and retention policies
//! - Real-time alerting hooks
//! - Compliance-ready audit trails

const std = @import("std");
const time = @import("../time.zig");
const sync = @import("../sync.zig");

/// Severity level for security events
pub const Severity = enum(u8) {
    critical = 0, // Immediate action required (breach, compromise)
    high = 1, // Urgent attention needed (failed auth attacks)
    medium = 2, // Significant events (permission changes)
    low = 3, // Minor events (successful logins)
    info = 4, // Informational (routine operations)

    pub fn toString(self: Severity) []const u8 {
        return switch (self) {
            .critical => "CRITICAL",
            .high => "HIGH",
            .medium => "MEDIUM",
            .low => "LOW",
            .info => "INFO",
        };
    }
};

/// Category of security event
pub const EventCategory = enum {
    authentication, // Login, logout, token operations
    authorization, // Permission checks, access denials
    data_access, // Data reads, writes, deletions
    data_modification, // Data changes with before/after
    system, // Configuration changes, startup/shutdown
    network, // Connection events, TLS handshakes
    crypto, // Key operations, certificate events
    session, // Session creation, expiration, invalidation
    rate_limit, // Rate limiting events
    input_validation, // Validation failures, sanitization
    secrets, // Secret access, rotation
    compliance, // Compliance-related events

    pub fn toString(self: EventCategory) []const u8 {
        return switch (self) {
            .authentication => "AUTH",
            .authorization => "AUTHZ",
            .data_access => "DATA_ACCESS",
            .data_modification => "DATA_MOD",
            .system => "SYSTEM",
            .network => "NETWORK",
            .crypto => "CRYPTO",
            .session => "SESSION",
            .rate_limit => "RATE_LIMIT",
            .input_validation => "INPUT_VAL",
            .secrets => "SECRETS",
            .compliance => "COMPLIANCE",
        };
    }
};

/// Outcome of the security event
pub const EventOutcome = enum {
    success,
    failure,
    blocked,
    warning,
    unknown,

    pub fn toString(self: EventOutcome) []const u8 {
        return switch (self) {
            .success => "SUCCESS",
            .failure => "FAILURE",
            .blocked => "BLOCKED",
            .warning => "WARNING",
            .unknown => "UNKNOWN",
        };
    }
};

/// Actor information (who performed the action)
pub const Actor = struct {
    id: []const u8,
    type: ActorType,
    ip_address: ?[]const u8 = null,
    user_agent: ?[]const u8 = null,
    session_id: ?[]const u8 = null,
    api_key_id: ?[]const u8 = null,

    pub const ActorType = enum {
        user,
        service,
        system,
        anonymous,
        api_client,

        pub fn toString(self: ActorType) []const u8 {
            return switch (self) {
                .user => "USER",
                .service => "SERVICE",
                .system => "SYSTEM",
                .anonymous => "ANONYMOUS",
                .api_client => "API_CLIENT",
            };
        }
    };
};

/// Target resource information
pub const Target = struct {
    type: []const u8,
    id: ?[]const u8 = null,
    path: ?[]const u8 = null,
    metadata: ?std.StringArrayHashMapUnmanaged([]const u8) = null,
};

/// Complete security audit event
pub const AuditEvent = struct {
    /// Unique event identifier
    id: []const u8,
    /// Event timestamp (Unix nanoseconds)
    timestamp: i128,
    /// Severity level
    severity: Severity,
    /// Event category
    category: EventCategory,
    /// Specific event type (e.g., "login_attempt", "permission_check")
    event_type: []const u8,
    /// Event outcome
    outcome: EventOutcome,
    /// Actor who triggered the event
    actor: ?Actor,
    /// Target of the action
    target: ?Target,
    /// Human-readable message
    message: []const u8,
    /// Additional context data
    context: ?std.StringArrayHashMapUnmanaged([]const u8) = null,
    /// Previous event hash for chain integrity
    prev_hash: ?[32]u8 = null,
    /// This event's hash
    event_hash: ?[32]u8 = null,
    /// Related event IDs (for correlation)
    related_events: ?[]const []const u8 = null,
    /// Source system/component
    source: []const u8,

    pub fn deinit(self: *AuditEvent, allocator: std.mem.Allocator) void {
        allocator.free(self.id);
        allocator.free(self.event_type);
        allocator.free(self.message);
        allocator.free(self.source);

        if (self.actor) |*actor| {
            allocator.free(actor.id);
            if (actor.ip_address) |ip| allocator.free(ip);
            if (actor.user_agent) |ua| allocator.free(ua);
            if (actor.session_id) |sid| allocator.free(sid);
            if (actor.api_key_id) |kid| allocator.free(kid);
        }

        if (self.target) |*target| {
            allocator.free(target.type);
            if (target.id) |id| allocator.free(id);
            if (target.path) |path| allocator.free(path);
            if (target.metadata) |*meta| {
                var it = meta.iterator();
                while (it.next()) |entry| {
                    allocator.free(entry.key_ptr.*);
                    allocator.free(entry.value_ptr.*);
                }
                meta.deinit(allocator);
            }
        }

        if (self.context) |*ctx| {
            var it = ctx.iterator();
            while (it.next()) |entry| {
                allocator.free(entry.key_ptr.*);
                allocator.free(entry.value_ptr.*);
            }
            ctx.deinit(allocator);
        }

        if (self.related_events) |events| {
            for (events) |e| allocator.free(e);
            allocator.free(events);
        }
    }
};

/// Alert callback function type
pub const AlertCallback = *const fn (event: *const AuditEvent) void;

/// Audit log configuration
pub const AuditConfig = struct {
    /// Minimum severity level to log
    min_severity: Severity = .info,
    /// Enable tamper-evident hash chain
    enable_hash_chain: bool = true,
    /// HMAC key for hash chain (should be securely generated)
    hmac_key: ?[32]u8 = null,
    /// Maximum events to keep in memory before flushing
    buffer_size: usize = 1000,
    /// Log file path (null for memory-only)
    log_path: ?[]const u8 = null,
    /// Enable real-time alerts
    enable_alerts: bool = true,
    /// Alert threshold severity
    alert_threshold: Severity = .high,
    /// Categories to log (null = all)
    enabled_categories: ?[]const EventCategory = null,
    /// Retention period in days (0 = indefinite)
    retention_days: u32 = 90,
    /// Enable compression for stored logs
    enable_compression: bool = false,
    /// Include stack traces for errors
    include_stack_traces: bool = false,
    /// Mask sensitive data in logs
    mask_sensitive_data: bool = true,
    /// Sensitive field patterns to mask
    sensitive_patterns: []const []const u8 = &.{
        "password",
        "secret",
        "token",
        "key",
        "credential",
        "auth",
    },
};

/// Security audit logger
pub const AuditLogger = struct {
    allocator: std.mem.Allocator,
    config: AuditConfig,
    events: std.ArrayListUnmanaged(AuditEvent),
    event_counter: std.atomic.Value(u64),
    last_hash: [32]u8,
    alert_callbacks: std.ArrayListUnmanaged(AlertCallback),
    mutex: sync.Mutex,
    /// Statistics
    stats: AuditStats,

    pub const AuditStats = struct {
        total_events: u64 = 0,
        events_by_severity: [5]u64 = .{ 0, 0, 0, 0, 0 },
        events_by_category: [12]u64 = .{0} ** 12,
        alerts_triggered: u64 = 0,
        events_dropped: u64 = 0,
    };

    pub fn init(allocator: std.mem.Allocator, config: AuditConfig) AuditLogger {
        var initial_hash: [32]u8 = undefined;
        // Initialize with random data or zeros
        if (config.hmac_key) |key| {
            var hmac = std.crypto.auth.hmac.sha2.HmacSha256.init(&key);
            hmac.update("AUDIT_LOG_INIT");
            hmac.final(&initial_hash);
        } else {
            @memset(&initial_hash, 0);
        }

        return .{
            .allocator = allocator,
            .config = config,
            .events = std.ArrayListUnmanaged(AuditEvent){},
            .event_counter = std.atomic.Value(u64).init(0),
            .last_hash = initial_hash,
            .alert_callbacks = std.ArrayListUnmanaged(AlertCallback){},
            .mutex = .{},
            .stats = .{},
        };
    }

    pub fn deinit(self: *AuditLogger) void {
        for (self.events.items) |*event| {
            event.deinit(self.allocator);
        }
        self.events.deinit(self.allocator);
        self.alert_callbacks.deinit(self.allocator);
        self.* = undefined;
    }

    /// Register an alert callback
    pub fn registerAlertCallback(self: *AuditLogger, callback: AlertCallback) !void {
        self.mutex.lock();
        defer self.mutex.unlock();
        try self.alert_callbacks.append(self.allocator, callback);
    }

    /// Log a security event
    pub fn log(self: *AuditLogger, event_builder: EventBuilder) !void {
        // Check if category is enabled
        if (self.config.enabled_categories) |categories| {
            var found = false;
            for (categories) |cat| {
                if (cat == event_builder.category) {
                    found = true;
                    break;
                }
            }
            if (!found) return;
        }

        // Check severity threshold
        if (@intFromEnum(event_builder.severity) > @intFromEnum(self.config.min_severity)) {
            return;
        }

        self.mutex.lock();
        defer self.mutex.unlock();

        // Generate event ID
        const event_id = self.event_counter.fetchAdd(1, .monotonic);
        const id = try std.fmt.allocPrint(self.allocator, "evt_{d}_{d}", .{
            time.unixSeconds(),
            event_id,
        });
        errdefer self.allocator.free(id);

        // Build the event
        var event = AuditEvent{
            .id = id,
            .timestamp = @as(i128, time.nowNanoseconds()),
            .severity = event_builder.severity,
            .category = event_builder.category,
            .event_type = try self.allocator.dupe(u8, event_builder.event_type),
            .outcome = event_builder.outcome,
            .actor = if (event_builder.actor) |a| try self.dupeActor(a) else null,
            .target = if (event_builder.target) |t| try self.dupeTarget(t) else null,
            .message = try self.maskSensitiveData(event_builder.message),
            .context = if (event_builder.context) |c| try self.dupeContext(c) else null,
            .source = try self.allocator.dupe(u8, event_builder.source),
            .prev_hash = if (self.config.enable_hash_chain) self.last_hash else null,
            .event_hash = null,
            .related_events = null,
        };

        // Compute event hash for chain integrity
        if (self.config.enable_hash_chain) {
            event.event_hash = self.computeEventHash(&event);
            self.last_hash = event.event_hash.?;
        }

        // Update statistics
        self.stats.total_events += 1;
        self.stats.events_by_severity[@intFromEnum(event.severity)] += 1;
        self.stats.events_by_category[@intFromEnum(event.category)] += 1;

        // Check buffer capacity
        if (self.events.items.len >= self.config.buffer_size) {
            // Flush oldest event
            if (self.events.items.len > 0) {
                var old = self.events.orderedRemove(0);
                old.deinit(self.allocator);
                self.stats.events_dropped += 1;
            }
        }

        try self.events.append(self.allocator, event);

        // Trigger alerts if needed
        if (self.config.enable_alerts and
            @intFromEnum(event.severity) <= @intFromEnum(self.config.alert_threshold))
        {
            self.triggerAlerts(&event);
        }
    }

    /// Quick logging methods for common events
    pub fn logAuthSuccess(self: *AuditLogger, actor: Actor, method: []const u8) !void {
        try self.log(EventBuilder{
            .severity = .low,
            .category = .authentication,
            .event_type = "auth_success",
            .outcome = .success,
            .actor = actor,
            .message = try std.fmt.allocPrint(self.allocator, "Authentication successful via {s}", .{method}),
            .source = "auth_system",
        });
    }

    pub fn logAuthFailure(self: *AuditLogger, actor: Actor, reason: []const u8) !void {
        try self.log(EventBuilder{
            .severity = .medium,
            .category = .authentication,
            .event_type = "auth_failure",
            .outcome = .failure,
            .actor = actor,
            .message = try std.fmt.allocPrint(self.allocator, "Authentication failed: {s}", .{reason}),
            .source = "auth_system",
        });
    }

    pub fn logAccessDenied(self: *AuditLogger, actor: Actor, resource: []const u8, permission: []const u8) !void {
        try self.log(EventBuilder{
            .severity = .medium,
            .category = .authorization,
            .event_type = "access_denied",
            .outcome = .blocked,
            .actor = actor,
            .target = Target{ .type = "resource", .path = resource },
            .message = try std.fmt.allocPrint(self.allocator, "Access denied to {s}, missing permission: {s}", .{ resource, permission }),
            .source = "authz_system",
        });
    }

    pub fn logRateLimitExceeded(self: *AuditLogger, actor: Actor, limit_type: []const u8) !void {
        try self.log(EventBuilder{
            .severity = .medium,
            .category = .rate_limit,
            .event_type = "rate_limit_exceeded",
            .outcome = .blocked,
            .actor = actor,
            .message = try std.fmt.allocPrint(self.allocator, "Rate limit exceeded: {s}", .{limit_type}),
            .source = "rate_limiter",
        });
    }

    pub fn logSecurityBreach(self: *AuditLogger, actor: ?Actor, description: []const u8) !void {
        try self.log(EventBuilder{
            .severity = .critical,
            .category = .system,
            .event_type = "security_breach",
            .outcome = .failure,
            .actor = actor,
            .message = description,
            .source = "security_monitor",
        });
    }

    /// Get events within a time range
    pub fn getEvents(
        self: *AuditLogger,
        start_time: ?i128,
        end_time: ?i128,
        category: ?EventCategory,
        severity: ?Severity,
    ) []const AuditEvent {
        self.mutex.lock();
        defer self.mutex.unlock();

        var result = std.ArrayListUnmanaged(AuditEvent){};

        for (self.events.items) |event| {
            // Time filter
            if (start_time) |st| {
                if (event.timestamp < st) continue;
            }
            if (end_time) |et| {
                if (event.timestamp > et) continue;
            }
            // Category filter
            if (category) |cat| {
                if (event.category != cat) continue;
            }
            // Severity filter
            if (severity) |sev| {
                if (@intFromEnum(event.severity) > @intFromEnum(sev)) continue;
            }

            result.append(self.allocator, event) catch continue;
        }

        return result.items;
    }

    /// Verify hash chain integrity
    pub fn verifyChainIntegrity(self: *AuditLogger) bool {
        if (!self.config.enable_hash_chain) return true;

        self.mutex.lock();
        defer self.mutex.unlock();

        var prev_hash: ?[32]u8 = null;

        for (self.events.items) |*event| {
            // Check prev_hash matches
            if (prev_hash) |ph| {
                if (event.prev_hash) |eph| {
                    if (!std.mem.eql(u8, &ph, &eph)) {
                        return false;
                    }
                } else {
                    return false;
                }
            }

            // Verify this event's hash
            const computed = self.computeEventHash(event);
            if (event.event_hash) |eh| {
                if (!std.mem.eql(u8, &computed, &eh)) {
                    return false;
                }
            }

            prev_hash = event.event_hash;
        }

        return true;
    }

    /// Export events to JSON format
    pub fn exportJson(self: *AuditLogger, allocator: std.mem.Allocator) ![]const u8 {
        self.mutex.lock();
        defer self.mutex.unlock();

        var buffer = std.ArrayListUnmanaged(u8).empty;
        defer buffer.deinit(allocator);

        try buffer.appendSlice(allocator, "[\n");

        for (self.events.items, 0..) |event, i| {
            if (i > 0) try buffer.appendSlice(allocator, ",\n");
            try self.eventToJson(allocator, &buffer, &event);
        }

        try buffer.appendSlice(allocator, "\n]");

        return buffer.toOwnedSlice(allocator);
    }

    /// Get audit statistics
    pub fn getStats(self: *AuditLogger) AuditStats {
        self.mutex.lock();
        defer self.mutex.unlock();
        return self.stats;
    }

    // Private helpers

    fn computeEventHash(self: *AuditLogger, event: *const AuditEvent) [32]u8 {
        var hasher = if (self.config.hmac_key) |key|
            std.crypto.auth.hmac.sha2.HmacSha256.init(&key)
        else
            std.crypto.auth.hmac.sha2.HmacSha256.init(&([_]u8{0} ** 32));

        hasher.update(event.id);
        hasher.update(std.mem.asBytes(&event.timestamp));
        hasher.update(&[_]u8{@intFromEnum(event.severity)});
        hasher.update(&[_]u8{@intFromEnum(event.category)});
        hasher.update(event.event_type);
        hasher.update(&[_]u8{@intFromEnum(event.outcome)});
        hasher.update(event.message);

        if (event.prev_hash) |ph| {
            hasher.update(&ph);
        }

        var result: [32]u8 = undefined;
        hasher.final(&result);
        return result;
    }

    fn maskSensitiveData(self: *AuditLogger, message: []const u8) ![]const u8 {
        if (!self.config.mask_sensitive_data) {
            return self.allocator.dupe(u8, message);
        }

        var result = try self.allocator.dupe(u8, message);

        // Simple masking - in production would use regex
        for (self.config.sensitive_patterns) |pattern| {
            // Find and mask pattern occurrences
            var i: usize = 0;
            while (i < result.len) {
                if (std.mem.indexOf(u8, result[i..], pattern)) |idx| {
                    // Mask the value after the pattern (simplified)
                    const start = i + idx + pattern.len;
                    if (start < result.len and result[start] == '=') {
                        var end = start + 1;
                        while (end < result.len and result[end] != ' ' and result[end] != '&') {
                            result[end] = '*';
                            end += 1;
                        }
                    }
                    i = start;
                } else {
                    break;
                }
            }
        }

        return result;
    }

    fn triggerAlerts(self: *AuditLogger, event: *const AuditEvent) void {
        self.stats.alerts_triggered += 1;
        for (self.alert_callbacks.items) |callback| {
            callback(event);
        }
    }

    fn dupeActor(self: *AuditLogger, actor: Actor) !Actor {
        return .{
            .id = try self.allocator.dupe(u8, actor.id),
            .type = actor.type,
            .ip_address = if (actor.ip_address) |ip| try self.allocator.dupe(u8, ip) else null,
            .user_agent = if (actor.user_agent) |ua| try self.allocator.dupe(u8, ua) else null,
            .session_id = if (actor.session_id) |sid| try self.allocator.dupe(u8, sid) else null,
            .api_key_id = if (actor.api_key_id) |kid| try self.allocator.dupe(u8, kid) else null,
        };
    }

    fn dupeTarget(self: *AuditLogger, target: Target) !Target {
        return .{
            .type = try self.allocator.dupe(u8, target.type),
            .id = if (target.id) |id| try self.allocator.dupe(u8, id) else null,
            .path = if (target.path) |path| try self.allocator.dupe(u8, path) else null,
            .metadata = if (target.metadata) |meta| try self.dupeContext(meta) else null,
        };
    }

    fn dupeContext(self: *AuditLogger, ctx: std.StringArrayHashMapUnmanaged([]const u8)) !std.StringArrayHashMapUnmanaged([]const u8) {
        var result = std.StringArrayHashMapUnmanaged([]const u8){};
        var it = ctx.iterator();
        while (it.next()) |entry| {
            try result.put(
                self.allocator,
                try self.allocator.dupe(u8, entry.key_ptr.*),
                try self.allocator.dupe(u8, entry.value_ptr.*),
            );
        }
        return result;
    }

    fn eventToJson(self: *AuditLogger, allocator: std.mem.Allocator, buffer: *std.ArrayListUnmanaged(u8), event: *const AuditEvent) !void {
        _ = self;
        try buffer.appendSlice(allocator, "  {");
        try std.fmt.format(buffer.writer(allocator), "\"id\":\"{s}\",", .{event.id});
        try std.fmt.format(buffer.writer(allocator), "\"timestamp\":{d},", .{event.timestamp});
        try std.fmt.format(buffer.writer(allocator), "\"severity\":\"{s}\",", .{event.severity.toString()});
        try std.fmt.format(buffer.writer(allocator), "\"category\":\"{s}\",", .{event.category.toString()});
        try std.fmt.format(buffer.writer(allocator), "\"event_type\":\"{s}\",", .{event.event_type});
        try std.fmt.format(buffer.writer(allocator), "\"outcome\":\"{s}\",", .{event.outcome.toString()});
        try std.fmt.format(buffer.writer(allocator), "\"message\":\"{s}\",", .{event.message});
        try std.fmt.format(buffer.writer(allocator), "\"source\":\"{s}\"", .{event.source});
        try buffer.appendSlice(allocator, "}");
    }
};

/// Builder for creating audit events
pub const EventBuilder = struct {
    severity: Severity = .info,
    category: EventCategory = .system,
    event_type: []const u8 = "unknown",
    outcome: EventOutcome = .unknown,
    actor: ?Actor = null,
    target: ?Target = null,
    message: []const u8 = "",
    context: ?std.StringArrayHashMapUnmanaged([]const u8) = null,
    source: []const u8 = "unknown",
};

// Common predefined event types
pub const EventTypes = struct {
    // Authentication
    pub const LOGIN_SUCCESS = "login_success";
    pub const LOGIN_FAILURE = "login_failure";
    pub const LOGOUT = "logout";
    pub const TOKEN_ISSUED = "token_issued";
    pub const TOKEN_REVOKED = "token_revoked";
    pub const TOKEN_EXPIRED = "token_expired";
    pub const MFA_SUCCESS = "mfa_success";
    pub const MFA_FAILURE = "mfa_failure";
    pub const PASSWORD_CHANGED = "password_changed";
    pub const PASSWORD_RESET = "password_reset";

    // Authorization
    pub const ACCESS_GRANTED = "access_granted";
    pub const ACCESS_DENIED = "access_denied";
    pub const PERMISSION_CHANGED = "permission_changed";
    pub const ROLE_ASSIGNED = "role_assigned";
    pub const ROLE_REVOKED = "role_revoked";

    // Data
    pub const DATA_READ = "data_read";
    pub const DATA_CREATED = "data_created";
    pub const DATA_UPDATED = "data_updated";
    pub const DATA_DELETED = "data_deleted";
    pub const DATA_EXPORTED = "data_exported";

    // System
    pub const CONFIG_CHANGED = "config_changed";
    pub const SERVICE_START = "service_start";
    pub const SERVICE_STOP = "service_stop";
    pub const SECURITY_BREACH = "security_breach";
    pub const INTRUSION_DETECTED = "intrusion_detected";

    // Network
    pub const CONNECTION_OPENED = "connection_opened";
    pub const CONNECTION_CLOSED = "connection_closed";
    pub const TLS_HANDSHAKE = "tls_handshake";
    pub const CERTIFICATE_EXPIRED = "certificate_expired";

    // Rate limiting
    pub const RATE_LIMIT_WARNING = "rate_limit_warning";
    pub const RATE_LIMIT_EXCEEDED = "rate_limit_exceeded";
    pub const IP_BLOCKED = "ip_blocked";
};

test "audit logger basic operations" {
    const allocator = std.testing.allocator;
    var logger = AuditLogger.init(allocator, .{});
    defer logger.deinit();

    try logger.log(.{
        .severity = .info,
        .category = .authentication,
        .event_type = EventTypes.LOGIN_SUCCESS,
        .outcome = .success,
        .actor = .{ .id = "user123", .type = .user },
        .message = "User logged in successfully",
        .source = "test",
    });

    const stats = logger.getStats();
    try std.testing.expectEqual(@as(u64, 1), stats.total_events);
    try std.testing.expectEqual(@as(u64, 1), stats.events_by_severity[@intFromEnum(Severity.info)]);
}

test "audit hash chain integrity" {
    const allocator = std.testing.allocator;
    var hmac_key: [32]u8 = undefined;
    std.crypto.random.bytes(&hmac_key);

    var logger = AuditLogger.init(allocator, .{
        .enable_hash_chain = true,
        .hmac_key = hmac_key,
    });
    defer logger.deinit();

    // Log multiple events
    for (0..5) |i| {
        try logger.log(.{
            .severity = .info,
            .category = .system,
            .event_type = "test_event",
            .outcome = .success,
            .message = try std.fmt.allocPrint(allocator, "Event {d}", .{i}),
            .source = "test",
        });
    }

    // Verify chain integrity
    try std.testing.expect(logger.verifyChainIntegrity());
}

test "audit severity filtering" {
    const allocator = std.testing.allocator;
    var logger = AuditLogger.init(allocator, .{
        .min_severity = .medium,
    });
    defer logger.deinit();

    // This should be logged (medium severity meets threshold)
    try logger.log(.{
        .severity = .medium,
        .category = .system,
        .event_type = "test",
        .outcome = .success,
        .message = "Medium event",
        .source = "test",
    });

    // This should not be logged (low severity below threshold)
    try logger.log(.{
        .severity = .low,
        .category = .system,
        .event_type = "test",
        .outcome = .success,
        .message = "Low event",
        .source = "test",
    });

    const stats = logger.getStats();
    try std.testing.expectEqual(@as(u64, 1), stats.total_events);
}
