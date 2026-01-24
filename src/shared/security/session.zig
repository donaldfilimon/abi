//! Session management system for secure user sessions.
//!
//! This module provides:
//! - Secure session ID generation
//! - Session storage and retrieval
//! - Session expiration and renewal
//! - Concurrent session management
//! - Session fixation protection
//! - Device/IP binding (optional)
//! - Activity tracking

const std = @import("std");
const time = @import("../time.zig");
const crypto = std.crypto;

/// Session configuration
pub const SessionConfig = struct {
    /// Session ID length in bytes
    id_length: usize = 32,
    /// Session lifetime in seconds
    lifetime: i64 = 3600, // 1 hour
    /// Idle timeout in seconds (0 = no idle timeout)
    idle_timeout: i64 = 1800, // 30 minutes
    /// Enable session renewal on activity
    renew_on_activity: bool = true,
    /// Renewal threshold (renew if less than this many seconds left)
    renewal_threshold: i64 = 600, // 10 minutes
    /// Maximum concurrent sessions per user
    max_sessions_per_user: usize = 5,
    /// Bind session to IP address
    bind_to_ip: bool = false,
    /// Bind session to user agent
    bind_to_user_agent: bool = false,
    /// Enable secure cookie flags
    secure_cookie: bool = true,
    /// Cookie name
    cookie_name: []const u8 = "session_id",
    /// Cookie path
    cookie_path: []const u8 = "/",
    /// Cookie domain
    cookie_domain: ?[]const u8 = null,
    /// SameSite attribute
    same_site: SameSite = .strict,
    /// HTTP only flag
    http_only: bool = true,
    /// Enable session rotation on privilege escalation
    rotate_on_privilege_change: bool = true,
    /// HMAC key for session ID signing
    signing_key: ?[32]u8 = null,
};

/// SameSite cookie attribute
pub const SameSite = enum {
    strict,
    lax,
    none,

    pub fn toString(self: SameSite) []const u8 {
        return switch (self) {
            .strict => "Strict",
            .lax => "Lax",
            .none => "None",
        };
    }
};

/// Session data
pub const Session = struct {
    /// Unique session ID
    id: []const u8,
    /// User ID (if authenticated)
    user_id: ?[]const u8,
    /// Session creation time
    created_at: i64,
    /// Last activity time
    last_activity: i64,
    /// Session expiration time
    expires_at: i64,
    /// Client IP address
    ip_address: ?[]const u8,
    /// Client user agent
    user_agent: ?[]const u8,
    /// Custom session data
    data: std.StringArrayHashMapUnmanaged([]const u8),
    /// Authentication level
    auth_level: AuthLevel,
    /// Is session valid
    is_valid: bool,
    /// Previous session ID (for rotation tracking)
    previous_id: ?[]const u8,

    pub const AuthLevel = enum(u8) {
        none = 0,
        basic = 1,
        standard = 2,
        elevated = 3,
        admin = 4,
    };

    pub fn deinit(self: *Session, allocator: std.mem.Allocator) void {
        allocator.free(self.id);
        if (self.user_id) |uid| allocator.free(uid);
        if (self.ip_address) |ip| allocator.free(ip);
        if (self.user_agent) |ua| allocator.free(ua);
        if (self.previous_id) |pid| allocator.free(pid);

        var it = self.data.iterator();
        while (it.next()) |entry| {
            allocator.free(entry.key_ptr.*);
            allocator.free(entry.value_ptr.*);
        }
        self.data.deinit(allocator);
    }

    /// Check if session is expired
    pub fn isExpired(self: Session) bool {
        return time.unixSeconds() > self.expires_at;
    }

    /// Check if session is idle (timed out due to inactivity)
    pub fn isIdle(self: Session, idle_timeout: i64) bool {
        if (idle_timeout == 0) return false;
        return time.unixSeconds() - self.last_activity > idle_timeout;
    }

    /// Get a session data value
    pub fn get(self: *Session, key: []const u8) ?[]const u8 {
        return self.data.get(key);
    }

    /// Set a session data value
    pub fn set(self: *Session, allocator: std.mem.Allocator, key: []const u8, value: []const u8) !void {
        const key_copy = try allocator.dupe(u8, key);
        errdefer allocator.free(key_copy);

        const value_copy = try allocator.dupe(u8, value);
        errdefer allocator.free(value_copy);

        // Remove old value if exists
        if (self.data.fetchRemove(key)) |kv| {
            allocator.free(kv.key);
            allocator.free(kv.value);
        }

        try self.data.put(allocator, key_copy, value_copy);
    }

    /// Remove a session data value
    pub fn remove(self: *Session, allocator: std.mem.Allocator, key: []const u8) bool {
        if (self.data.fetchRemove(key)) |kv| {
            allocator.free(kv.key);
            allocator.free(kv.value);
            return true;
        }
        return false;
    }
};

/// Session manager
pub const SessionManager = struct {
    allocator: std.mem.Allocator,
    config: SessionConfig,
    sessions: std.StringArrayHashMapUnmanaged(*Session),
    user_sessions: std.StringArrayHashMapUnmanaged(std.ArrayListUnmanaged([]const u8)),
    mutex: std.Thread.Mutex,
    stats: SessionStats,

    pub const SessionStats = struct {
        total_sessions_created: u64 = 0,
        active_sessions: u64 = 0,
        expired_sessions: u64 = 0,
        invalidated_sessions: u64 = 0,
        rotated_sessions: u64 = 0,
    };

    pub fn init(allocator: std.mem.Allocator, config: SessionConfig) SessionManager {
        return .{
            .allocator = allocator,
            .config = config,
            .sessions = std.StringArrayHashMapUnmanaged(*Session){},
            .user_sessions = std.StringArrayHashMapUnmanaged(std.ArrayListUnmanaged([]const u8)){},
            .mutex = .{},
            .stats = .{},
        };
    }

    pub fn deinit(self: *SessionManager) void {
        var it = self.sessions.iterator();
        while (it.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
            entry.value_ptr.*.deinit(self.allocator);
            self.allocator.destroy(entry.value_ptr.*);
        }
        self.sessions.deinit(self.allocator);

        var user_it = self.user_sessions.iterator();
        while (user_it.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
            for (entry.value_ptr.items) |sid| {
                self.allocator.free(sid);
            }
            entry.value_ptr.deinit(self.allocator);
        }
        self.user_sessions.deinit(self.allocator);
    }

    /// Create a new session
    pub fn create(self: *SessionManager, options: CreateOptions) !*Session {
        self.mutex.lock();
        defer self.mutex.unlock();

        const now = time.unixSeconds();

        // Check max sessions for user
        if (options.user_id) |uid| {
            if (self.user_sessions.get(uid)) |user_sids| {
                if (user_sids.items.len >= self.config.max_sessions_per_user) {
                    // Remove oldest session
                    if (user_sids.items.len > 0) {
                        const oldest_sid = user_sids.items[0];
                        self.destroySessionInternal(oldest_sid);
                    }
                }
            }
        }

        // Generate session ID
        const session_id = try self.generateSessionId();
        errdefer self.allocator.free(session_id);

        // Create session
        const session = try self.allocator.create(Session);
        errdefer self.allocator.destroy(session);

        session.* = .{
            .id = session_id,
            .user_id = if (options.user_id) |uid| try self.allocator.dupe(u8, uid) else null,
            .created_at = now,
            .last_activity = now,
            .expires_at = now + self.config.lifetime,
            .ip_address = if (options.ip_address) |ip| try self.allocator.dupe(u8, ip) else null,
            .user_agent = if (options.user_agent) |ua| try self.allocator.dupe(u8, ua) else null,
            .data = std.StringArrayHashMapUnmanaged([]const u8){},
            .auth_level = options.auth_level,
            .is_valid = true,
            .previous_id = null,
        };

        // Store session
        const id_copy = try self.allocator.dupe(u8, session_id);
        try self.sessions.put(self.allocator, id_copy, session);

        // Track user sessions
        if (options.user_id) |uid| {
            var user_sids = self.user_sessions.get(uid) orelse blk: {
                const uid_copy = try self.allocator.dupe(u8, uid);
                try self.user_sessions.put(self.allocator, uid_copy, std.ArrayListUnmanaged([]const u8){});
                break :blk self.user_sessions.get(uid).?;
            };
            try user_sids.append(self.allocator, try self.allocator.dupe(u8, session_id));
        }

        self.stats.total_sessions_created += 1;
        self.stats.active_sessions += 1;

        return session;
    }

    /// Get a session by ID
    pub fn get(self: *SessionManager, session_id: []const u8, options: GetOptions) !?*Session {
        self.mutex.lock();
        defer self.mutex.unlock();

        const session = self.sessions.get(session_id) orelse return null;

        // Validate session
        if (!session.is_valid) return null;

        if (session.isExpired()) {
            self.stats.expired_sessions += 1;
            session.is_valid = false;
            return null;
        }

        if (session.isIdle(self.config.idle_timeout)) {
            self.stats.expired_sessions += 1;
            session.is_valid = false;
            return null;
        }

        // Validate bindings
        if (self.config.bind_to_ip) {
            if (session.ip_address) |expected_ip| {
                if (options.ip_address) |actual_ip| {
                    if (!std.mem.eql(u8, expected_ip, actual_ip)) {
                        return null;
                    }
                } else {
                    return null;
                }
            }
        }

        if (self.config.bind_to_user_agent) {
            if (session.user_agent) |expected_ua| {
                if (options.user_agent) |actual_ua| {
                    if (!std.mem.eql(u8, expected_ua, actual_ua)) {
                        return null;
                    }
                } else {
                    return null;
                }
            }
        }

        // Update activity and potentially renew
        const now = time.unixSeconds();
        session.last_activity = now;

        if (self.config.renew_on_activity) {
            const remaining = session.expires_at - now;
            if (remaining < self.config.renewal_threshold) {
                session.expires_at = now + self.config.lifetime;
            }
        }

        return session;
    }

    /// Destroy a session
    pub fn destroy(self: *SessionManager, session_id: []const u8) bool {
        self.mutex.lock();
        defer self.mutex.unlock();

        return self.destroySessionInternal(session_id);
    }

    /// Destroy all sessions for a user
    pub fn destroyAllForUser(self: *SessionManager, user_id: []const u8) u32 {
        self.mutex.lock();
        defer self.mutex.unlock();

        var count: u32 = 0;

        if (self.user_sessions.get(user_id)) |user_sids| {
            for (user_sids.items) |sid| {
                if (self.destroySessionInternal(sid)) {
                    count += 1;
                }
            }
        }

        return count;
    }

    /// Rotate session ID (for security, e.g., after login)
    pub fn rotate(self: *SessionManager, old_session_id: []const u8) !?[]const u8 {
        self.mutex.lock();
        defer self.mutex.unlock();

        const session = self.sessions.get(old_session_id) orelse return null;

        // Generate new ID
        const new_session_id = try self.generateSessionId();

        // Update session
        const old_id = session.id;
        session.id = new_session_id;
        session.previous_id = old_id;

        // Update storage
        _ = self.sessions.fetchRemove(old_session_id);
        const new_id_copy = try self.allocator.dupe(u8, new_session_id);
        try self.sessions.put(self.allocator, new_id_copy, session);

        // Update user sessions tracking
        if (session.user_id) |uid| {
            if (self.user_sessions.getPtr(uid)) |user_sids| {
                for (user_sids.items, 0..) |sid, i| {
                    if (std.mem.eql(u8, sid, old_session_id)) {
                        self.allocator.free(sid);
                        user_sids.items[i] = try self.allocator.dupe(u8, new_session_id);
                        break;
                    }
                }
            }
        }

        self.stats.rotated_sessions += 1;

        return new_session_id;
    }

    /// Elevate session authentication level
    pub fn elevate(self: *SessionManager, session_id: []const u8, new_level: Session.AuthLevel) !?[]const u8 {
        const session = self.get(session_id, .{}) catch return null;
        if (session == null) return null;

        session.?.auth_level = new_level;

        // Rotate session if configured
        if (self.config.rotate_on_privilege_change) {
            return self.rotate(session_id);
        }

        return session_id;
    }

    /// Get session cookie header value
    pub fn getCookieHeader(self: *SessionManager, session: *const Session) ![]const u8 {
        var buffer = std.ArrayList(u8).init(self.allocator);
        errdefer buffer.deinit();

        try std.fmt.format(buffer.writer(), "{s}={s}", .{
            self.config.cookie_name,
            session.id,
        });

        if (self.config.cookie_domain) |domain| {
            try std.fmt.format(buffer.writer(), "; Domain={s}", .{domain});
        }

        try std.fmt.format(buffer.writer(), "; Path={s}", .{self.config.cookie_path});

        if (self.config.http_only) {
            try buffer.appendSlice("; HttpOnly");
        }

        if (self.config.secure_cookie) {
            try buffer.appendSlice("; Secure");
        }

        try std.fmt.format(buffer.writer(), "; SameSite={s}", .{self.config.same_site.toString()});

        const max_age = session.expires_at - time.unixSeconds();
        if (max_age > 0) {
            try std.fmt.format(buffer.writer(), "; Max-Age={d}", .{max_age});
        }

        return buffer.toOwnedSlice();
    }

    /// Get statistics
    pub fn getStats(self: *SessionManager) SessionStats {
        self.mutex.lock();
        defer self.mutex.unlock();
        return self.stats;
    }

    /// Clean up expired sessions
    pub fn cleanup(self: *SessionManager) u32 {
        self.mutex.lock();
        defer self.mutex.unlock();

        var to_remove = std.ArrayList([]const u8).init(self.allocator);
        defer to_remove.deinit();

        var it = self.sessions.iterator();
        while (it.next()) |entry| {
            const session = entry.value_ptr.*;
            if (!session.is_valid or session.isExpired() or session.isIdle(self.config.idle_timeout)) {
                to_remove.append(entry.key_ptr.*) catch continue;
            }
        }

        var count: u32 = 0;
        for (to_remove.items) |sid| {
            if (self.destroySessionInternal(sid)) {
                count += 1;
            }
        }

        return count;
    }

    // Private methods

    fn generateSessionId(self: *SessionManager) ![]const u8 {
        var random_bytes: [64]u8 = undefined;
        crypto.random.bytes(random_bytes[0..self.config.id_length]);

        // Sign if key is configured
        if (self.config.signing_key) |key| {
            var hmac = crypto.auth.hmac.sha2.HmacSha256.init(&key);
            hmac.update(random_bytes[0..self.config.id_length]);
            var mac: [32]u8 = undefined;
            hmac.final(&mac);

            // Append first 8 bytes of MAC as signature
            @memcpy(random_bytes[self.config.id_length .. self.config.id_length + 8], mac[0..8]);

            const total_len = self.config.id_length + 8;
            return try self.base64UrlEncode(random_bytes[0..total_len]);
        }

        return try self.base64UrlEncode(random_bytes[0..self.config.id_length]);
    }

    fn base64UrlEncode(self: *SessionManager, data: []const u8) ![]const u8 {
        const encoder = std.base64.url_safe_no_pad;
        const size = encoder.Encoder.calcSize(data.len);
        const buf = try self.allocator.alloc(u8, size);
        _ = encoder.Encoder.encode(buf, data);
        return buf;
    }

    fn destroySessionInternal(self: *SessionManager, session_id: []const u8) bool {
        if (self.sessions.fetchRemove(session_id)) |kv| {
            const session = kv.value;

            // Remove from user tracking
            if (session.user_id) |uid| {
                if (self.user_sessions.getPtr(uid)) |user_sids| {
                    var i: usize = 0;
                    while (i < user_sids.items.len) {
                        if (std.mem.eql(u8, user_sids.items[i], session_id)) {
                            self.allocator.free(user_sids.items[i]);
                            _ = user_sids.orderedRemove(i);
                        } else {
                            i += 1;
                        }
                    }
                }
            }

            self.allocator.free(kv.key);
            session.deinit(self.allocator);
            self.allocator.destroy(session);

            self.stats.invalidated_sessions += 1;
            self.stats.active_sessions -|= 1;

            return true;
        }
        return false;
    }
};

pub const CreateOptions = struct {
    user_id: ?[]const u8 = null,
    ip_address: ?[]const u8 = null,
    user_agent: ?[]const u8 = null,
    auth_level: Session.AuthLevel = .none,
};

pub const GetOptions = struct {
    ip_address: ?[]const u8 = null,
    user_agent: ?[]const u8 = null,
};

/// Session errors
pub const SessionError = error{
    SessionNotFound,
    SessionExpired,
    SessionInvalid,
    MaxSessionsExceeded,
    OutOfMemory,
};

// Tests

test "session creation and retrieval" {
    const allocator = std.testing.allocator;
    var manager = SessionManager.init(allocator, .{});
    defer manager.deinit();

    // Create session
    const session = try manager.create(.{
        .user_id = "user123",
        .ip_address = "192.168.1.1",
    });

    try std.testing.expect(session.id.len > 0);
    try std.testing.expectEqualStrings("user123", session.user_id.?);

    // Retrieve session
    const retrieved = try manager.get(session.id, .{});
    try std.testing.expect(retrieved != null);
    try std.testing.expectEqualStrings(session.id, retrieved.?.id);
}

test "session data storage" {
    const allocator = std.testing.allocator;
    var manager = SessionManager.init(allocator, .{});
    defer manager.deinit();

    const session = try manager.create(.{});

    // Set data
    try session.set(allocator, "key1", "value1");
    try session.set(allocator, "key2", "value2");

    // Get data
    try std.testing.expectEqualStrings("value1", session.get("key1").?);
    try std.testing.expectEqualStrings("value2", session.get("key2").?);

    // Remove data
    try std.testing.expect(session.remove(allocator, "key1"));
    try std.testing.expect(session.get("key1") == null);
}

test "session rotation" {
    const allocator = std.testing.allocator;
    var manager = SessionManager.init(allocator, .{});
    defer manager.deinit();

    const session = try manager.create(.{ .user_id = "user123" });
    const old_id = try allocator.dupe(u8, session.id);
    defer allocator.free(old_id);

    // Rotate
    const new_id = try manager.rotate(old_id);
    try std.testing.expect(new_id != null);
    try std.testing.expect(!std.mem.eql(u8, old_id, new_id.?));

    // Old ID should not work
    const old_session = try manager.get(old_id, .{});
    try std.testing.expect(old_session == null);

    // New ID should work
    const new_session = try manager.get(new_id.?, .{});
    try std.testing.expect(new_session != null);
}

test "max sessions per user" {
    const allocator = std.testing.allocator;
    var manager = SessionManager.init(allocator, .{
        .max_sessions_per_user = 2,
    });
    defer manager.deinit();

    // Create max sessions
    const s1 = try manager.create(.{ .user_id = "user123" });
    const s2 = try manager.create(.{ .user_id = "user123" });

    const id1 = try allocator.dupe(u8, s1.id);
    defer allocator.free(id1);

    _ = s2;

    // Third session should remove first
    _ = try manager.create(.{ .user_id = "user123" });

    // First session should be gone
    const retrieved = try manager.get(id1, .{});
    try std.testing.expect(retrieved == null);
}
