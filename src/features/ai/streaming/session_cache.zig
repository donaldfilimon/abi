//! Session token cache for streaming reconnection.
//!
//! Stores the last N tokens per streaming session to support client
//! reconnection via Last-Event-ID. Uses LRU eviction when capacity
//! is reached and TTL-based expiration for stale sessions.
//!
//! ## Usage
//!
//! ```zig
//! var cache = SessionCache.init(allocator, .{});
//! defer cache.deinit();
//!
//! // Store tokens as they're streamed
//! try cache.storeToken("session-123", 1, "Hello");
//! try cache.storeToken("session-123", 2, " world");
//!
//! // Client reconnects with Last-Event-ID: 1
//! const tokens = try cache.getTokensSince("session-123", 1);
//! // tokens = [" world"]
//! ```

const std = @import("std");
const backends = @import("backends/mod.zig");
const platform_time = @import("../../../services/shared/time.zig");
const sync = @import("../../../services/shared/sync.zig");

/// Session cache configuration.
pub const SessionCacheConfig = struct {
    /// Maximum number of sessions to cache.
    max_sessions: usize = 100,

    /// Maximum tokens to cache per session.
    max_tokens_per_session: usize = 100,

    /// Time-to-live for cached sessions (milliseconds).
    ttl_ms: u64 = 300_000, // 5 minutes

    /// Cleanup interval for expired sessions (milliseconds).
    cleanup_interval_ms: u64 = 60_000, // 1 minute

    /// Optional time provider for deterministic tests.
    time_provider: ?platform_time.TimeProvider = null,
};

/// A cached token with its event ID and timestamp.
pub const CachedToken = struct {
    /// SSE event ID for this token.
    event_id: u64,

    /// Token text (owned by cache).
    text: []const u8,

    /// Timestamp when cached (milliseconds).
    timestamp_ms: i64,
};

/// A cached streaming session.
pub const SessionEntry = struct {
    /// Unique session identifier (owned).
    session_id: []const u8,

    /// Cached tokens (ring buffer behavior when full).
    tokens: std.ArrayListUnmanaged(CachedToken),

    /// When session was created.
    created_at_ms: i64,

    /// When session was last accessed.
    last_accessed_ms: i64,

    /// Backend type for this session.
    backend_type: backends.BackendType,

    /// Hash of the original prompt (for validation).
    prompt_hash: u64,

    /// Free all memory owned by this entry.
    pub fn deinit(self: *SessionEntry, allocator: std.mem.Allocator) void {
        allocator.free(self.session_id);
        for (self.tokens.items) |token| {
            allocator.free(token.text);
        }
        self.tokens.deinit(allocator);
        self.* = undefined;
    }
};

/// Thread-safe session cache with LRU eviction and TTL expiration.
pub const SessionCache = struct {
    allocator: std.mem.Allocator,
    config: SessionCacheConfig,
    sessions: std.StringHashMapUnmanaged(*SessionEntry),
    mutex: sync.Mutex,
    last_cleanup_ms: i64,
    time_provider: platform_time.TimeProvider,

    const Self = @This();

    /// Initialize a new session cache.
    pub fn init(allocator: std.mem.Allocator, config: SessionCacheConfig) Self {
        const time_provider = config.time_provider orelse platform_time.TimeProvider{};
        return .{
            .allocator = allocator,
            .config = config,
            .sessions = .{},
            .mutex = .{},
            .last_cleanup_ms = time_provider.nowMs(),
            .time_provider = time_provider,
        };
    }

    /// Deinitialize and free all cached sessions.
    pub fn deinit(self: *Self) void {
        var iter = self.sessions.valueIterator();
        while (iter.next()) |entry_ptr| {
            entry_ptr.*.deinit(self.allocator);
            self.allocator.destroy(entry_ptr.*);
        }
        self.sessions.deinit(self.allocator);
        self.* = undefined;
    }

    /// Store a token for a streaming session.
    ///
    /// Creates a new session if it doesn't exist. Evicts oldest tokens
    /// if the per-session limit is reached.
    pub fn storeToken(
        self: *Self,
        session_id: []const u8,
        event_id: u64,
        text: []const u8,
        backend_type: backends.BackendType,
        prompt_hash: u64,
    ) !void {
        self.mutex.lock();
        defer self.mutex.unlock();

        const now = self.time_provider.nowMs();

        // Maybe run cleanup
        self.maybeCleanupLocked(now);

        // Evict oldest session if at capacity BEFORE adding
        if (!self.sessions.contains(session_id) and self.sessions.count() >= self.config.max_sessions) {
            self.evictOldestLocked();
        }

        // Get or create session
        const gop = try self.sessions.getOrPut(self.allocator, session_id);
        if (!gop.found_existing) {
            // Create new session
            const entry = try self.allocator.create(SessionEntry);
            errdefer self.allocator.destroy(entry);

            entry.* = .{
                .session_id = try self.allocator.dupe(u8, session_id),
                .tokens = .empty,
                .created_at_ms = now,
                .last_accessed_ms = now,
                .backend_type = backend_type,
                .prompt_hash = prompt_hash,
            };
            gop.value_ptr.* = entry;
            gop.key_ptr.* = entry.session_id;
        }

        const entry = gop.value_ptr.*;
        entry.last_accessed_ms = now;

        // Evict oldest token if at per-session limit
        if (entry.tokens.items.len >= self.config.max_tokens_per_session) {
            const oldest = entry.tokens.orderedRemove(0);
            self.allocator.free(oldest.text);
        }

        // Store new token
        const token_copy = CachedToken{
            .event_id = event_id,
            .text = try self.allocator.dupe(u8, text),
            .timestamp_ms = now,
        };
        try entry.tokens.append(self.allocator, token_copy);
    }

    /// Get all tokens since a given event ID (exclusive).
    ///
    /// Returns null if session not found or expired.
    /// Caller must NOT free the returned slice (owned by cache).
    pub fn getTokensSince(
        self: *Self,
        session_id: []const u8,
        last_event_id: u64,
    ) ?[]const CachedToken {
        self.mutex.lock();
        defer self.mutex.unlock();

        const entry = self.sessions.get(session_id) orelse return null;

        // Check if expired
        const now = self.time_provider.nowMs();
        const ttl: i64 = @intCast(self.config.ttl_ms);
        if (now - entry.created_at_ms > ttl) {
            return null;
        }

        // Update access time
        entry.last_accessed_ms = now;

        // Find tokens after last_event_id
        var start_idx: usize = entry.tokens.items.len;
        for (entry.tokens.items, 0..) |token, i| {
            if (token.event_id > last_event_id) {
                start_idx = i;
                break;
            }
        }

        if (start_idx >= entry.tokens.items.len) {
            return &[_]CachedToken{};
        }

        return entry.tokens.items[start_idx..];
    }

    /// Get session metadata without tokens.
    pub fn getSessionInfo(
        self: *Self,
        session_id: []const u8,
    ) ?struct { backend_type: backends.BackendType, prompt_hash: u64, token_count: usize } {
        self.mutex.lock();
        defer self.mutex.unlock();

        const entry = self.sessions.get(session_id) orelse return null;

        // Check if expired
        const now = self.time_provider.nowMs();
        const ttl: i64 = @intCast(self.config.ttl_ms);
        if (now - entry.created_at_ms > ttl) {
            return null;
        }

        return .{
            .backend_type = entry.backend_type,
            .prompt_hash = entry.prompt_hash,
            .token_count = entry.tokens.items.len,
        };
    }

    /// Invalidate a session (remove from cache).
    pub fn invalidateSession(self: *Self, session_id: []const u8) void {
        self.mutex.lock();
        defer self.mutex.unlock();

        if (self.sessions.fetchRemove(session_id)) |removed| {
            removed.value.deinit(self.allocator);
            self.allocator.destroy(removed.value);
        }
    }

    /// Get cache statistics.
    pub fn getStats(self: *Self) struct {
        session_count: usize,
        total_tokens: usize,
        oldest_session_age_ms: i64,
    } {
        self.mutex.lock();
        defer self.mutex.unlock();

        const now = self.time_provider.nowMs();
        var total_tokens: usize = 0;
        var oldest_age: i64 = 0;

        var iter = self.sessions.valueIterator();
        while (iter.next()) |entry| {
            total_tokens += entry.*.tokens.items.len;
            const age = now - entry.*.created_at_ms;
            if (age > oldest_age) {
                oldest_age = age;
            }
        }

        return .{
            .session_count = self.sessions.count(),
            .total_tokens = total_tokens,
            .oldest_session_age_ms = oldest_age,
        };
    }

    /// Manually trigger cleanup of expired sessions.
    pub fn cleanup(self: *Self) void {
        self.mutex.lock();
        defer self.mutex.unlock();

        const now = self.time_provider.nowMs();
        self.forceCleanupLocked(now);
    }

    // --- Private methods ---

    fn maybeCleanupLocked(self: *Self, now: i64) void {
        const interval: i64 = @intCast(self.config.cleanup_interval_ms);
        if (now - self.last_cleanup_ms >= interval) {
            self.forceCleanupLocked(now);
        }
    }

    fn forceCleanupLocked(self: *Self, now: i64) void {
        const ttl: i64 = @intCast(self.config.ttl_ms);
        var to_remove = std.ArrayListUnmanaged([]const u8).empty;
        defer to_remove.deinit(self.allocator);

        var iter = self.sessions.iterator();
        while (iter.next()) |kv| {
            if (now - kv.value_ptr.*.created_at_ms > ttl) {
                to_remove.append(self.allocator, kv.key_ptr.*) catch continue;
            }
        }

        for (to_remove.items) |key| {
            if (self.sessions.fetchRemove(key)) |removed| {
                removed.value.deinit(self.allocator);
                self.allocator.destroy(removed.value);
            }
        }

        self.last_cleanup_ms = now;
    }

    fn evictOldestLocked(self: *Self) void {
        var oldest_time: i64 = std.math.maxInt(i64);
        var oldest_key: ?[]const u8 = null;

        var iter = self.sessions.iterator();
        while (iter.next()) |kv| {
            if (kv.value_ptr.*.last_accessed_ms < oldest_time) {
                oldest_time = kv.value_ptr.*.last_accessed_ms;
                oldest_key = kv.key_ptr.*;
            }
        }

        if (oldest_key) |key| {
            if (self.sessions.fetchRemove(key)) |removed| {
                removed.value.deinit(self.allocator);
                self.allocator.destroy(removed.value);
            }
        }
    }
};

/// Compute a hash of the prompt for session validation.
pub fn hashPrompt(prompt: []const u8) u64 {
    return std.hash.Wyhash.hash(0, prompt);
}

// ============================================================================
// Tests
// ============================================================================

const TestClock = struct {
    now_ms: i64 = 0,

    fn nowMs(ctx: ?*anyopaque) i64 {
        const clock: *TestClock = @ptrCast(@alignCast(ctx.?));
        return clock.now_ms;
    }
};

fn testTimeProvider(clock: *TestClock) platform_time.TimeProvider {
    return .{
        .ctx = @ptrCast(clock),
        .nowMsFn = TestClock.nowMs,
    };
}

test "SessionCache basic store and retrieve" {
    const allocator = std.testing.allocator;

    var cache = SessionCache.init(allocator, .{});
    defer cache.deinit();

    // Store tokens
    try cache.storeToken("session-1", 1, "Hello", .local, 12345);
    try cache.storeToken("session-1", 2, " world", .local, 12345);
    try cache.storeToken("session-1", 3, "!", .local, 12345);

    // Retrieve all tokens (since event 0)
    const tokens = cache.getTokensSince("session-1", 0);
    try std.testing.expect(tokens != null);
    try std.testing.expectEqual(@as(usize, 3), tokens.?.len);
    try std.testing.expectEqualStrings("Hello", tokens.?[0].text);
    try std.testing.expectEqualStrings(" world", tokens.?[1].text);
    try std.testing.expectEqualStrings("!", tokens.?[2].text);
}

test "SessionCache retrieve since specific event" {
    const allocator = std.testing.allocator;

    var cache = SessionCache.init(allocator, .{});
    defer cache.deinit();

    try cache.storeToken("session-1", 1, "Hello", .local, 12345);
    try cache.storeToken("session-1", 2, " world", .local, 12345);
    try cache.storeToken("session-1", 3, "!", .local, 12345);

    // Get tokens since event 1 (exclusive)
    const tokens = cache.getTokensSince("session-1", 1);
    try std.testing.expect(tokens != null);
    try std.testing.expectEqual(@as(usize, 2), tokens.?.len);
    try std.testing.expectEqualStrings(" world", tokens.?[0].text);
    try std.testing.expectEqualStrings("!", tokens.?[1].text);
}

test "SessionCache non-existent session returns null" {
    const allocator = std.testing.allocator;

    var cache = SessionCache.init(allocator, .{});
    defer cache.deinit();

    const tokens = cache.getTokensSince("non-existent", 0);
    try std.testing.expect(tokens == null);
}

test "SessionCache per-session token limit" {
    const allocator = std.testing.allocator;

    var cache = SessionCache.init(allocator, .{
        .max_tokens_per_session = 3,
    });
    defer cache.deinit();

    // Store 5 tokens - should evict oldest 2
    try cache.storeToken("session-1", 1, "A", .local, 0);
    try cache.storeToken("session-1", 2, "B", .local, 0);
    try cache.storeToken("session-1", 3, "C", .local, 0);
    try cache.storeToken("session-1", 4, "D", .local, 0);
    try cache.storeToken("session-1", 5, "E", .local, 0);

    const tokens = cache.getTokensSince("session-1", 0);
    try std.testing.expect(tokens != null);
    try std.testing.expectEqual(@as(usize, 3), tokens.?.len);
    try std.testing.expectEqualStrings("C", tokens.?[0].text);
    try std.testing.expectEqualStrings("D", tokens.?[1].text);
    try std.testing.expectEqualStrings("E", tokens.?[2].text);
}

test "SessionCache session eviction" {
    const allocator = std.testing.allocator;

    var clock = TestClock{};
    var cache = SessionCache.init(allocator, .{
        .max_sessions = 2,
        .time_provider = testTimeProvider(&clock),
    });
    defer cache.deinit();

    // Store 3 sessions - should evict oldest
    try cache.storeToken("session-1", 1, "A", .local, 0);
    clock.now_ms += 1;
    try cache.storeToken("session-2", 1, "B", .local, 0);
    clock.now_ms += 1;
    try cache.storeToken("session-3", 1, "C", .local, 0);

    // session-1 should be evicted
    try std.testing.expect(cache.getTokensSince("session-1", 0) == null);
    try std.testing.expect(cache.getTokensSince("session-2", 0) != null);
    try std.testing.expect(cache.getTokensSince("session-3", 0) != null);
}

test "SessionCache invalidate session" {
    const allocator = std.testing.allocator;

    var cache = SessionCache.init(allocator, .{});
    defer cache.deinit();

    try cache.storeToken("session-1", 1, "Hello", .local, 0);
    try std.testing.expect(cache.getTokensSince("session-1", 0) != null);

    cache.invalidateSession("session-1");
    try std.testing.expect(cache.getTokensSince("session-1", 0) == null);
}

test "SessionCache get session info" {
    const allocator = std.testing.allocator;

    var cache = SessionCache.init(allocator, .{});
    defer cache.deinit();

    try cache.storeToken("session-1", 1, "A", .openai, 99999);
    try cache.storeToken("session-1", 2, "B", .openai, 99999);

    const info = cache.getSessionInfo("session-1");
    try std.testing.expect(info != null);
    try std.testing.expectEqual(backends.BackendType.openai, info.?.backend_type);
    try std.testing.expectEqual(@as(u64, 99999), info.?.prompt_hash);
    try std.testing.expectEqual(@as(usize, 2), info.?.token_count);
}

test "SessionCache stats" {
    const allocator = std.testing.allocator;

    var cache = SessionCache.init(allocator, .{});
    defer cache.deinit();

    try cache.storeToken("session-1", 1, "A", .local, 0);
    try cache.storeToken("session-1", 2, "B", .local, 0);
    try cache.storeToken("session-2", 1, "C", .local, 0);

    const stats = cache.getStats();
    try std.testing.expectEqual(@as(usize, 2), stats.session_count);
    try std.testing.expectEqual(@as(usize, 3), stats.total_tokens);
}

test "hashPrompt consistency" {
    const hash1 = hashPrompt("Hello, world!");
    const hash2 = hashPrompt("Hello, world!");
    const hash3 = hashPrompt("Different prompt");

    try std.testing.expectEqual(hash1, hash2);
    try std.testing.expect(hash1 != hash3);
}
