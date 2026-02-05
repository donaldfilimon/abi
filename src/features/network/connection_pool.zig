//! Connection pooling for network resources.
//!
//! Provides efficient connection reuse and management for
//! HTTP clients and other network resources.

const std = @import("std");
const time = @import("../../services/shared/utils.zig");

/// Connection state.
pub const ConnectionState = enum {
    available,
    in_use,
    stale,
    closed,
};

/// Connection statistics.
pub const ConnectionStats = struct {
    created_at_ns: i64,
    last_used_at_ns: i64,
    use_count: u64,
    bytes_sent: u64,
    bytes_received: u64,
    errors: u64,
};

/// Pool configuration.
pub const ConnectionPoolConfig = struct {
    /// Maximum connections per host.
    max_connections_per_host: u32 = 10,
    /// Maximum total connections.
    max_total_connections: u32 = 100,
    /// Connection timeout (ns).
    connect_timeout_ns: u64 = 30_000_000_000,
    /// Idle timeout before closing (ns).
    idle_timeout_ns: u64 = 60_000_000_000,
    /// Maximum lifetime for a connection (ns).
    max_lifetime_ns: u64 = 300_000_000_000,
    /// Enable keepalive.
    enable_keepalive: bool = true,
    /// Keepalive interval (ns).
    keepalive_interval_ns: u64 = 15_000_000_000,
    /// Validate connections before use.
    validate_on_acquire: bool = true,
    /// Enable connection health checks.
    enable_health_check: bool = true,
    /// Health check interval (ns).
    health_check_interval_ns: u64 = 30_000_000_000,
};

/// Host key for connection pooling.
pub const HostKey = struct {
    host: [256]u8 = [_]u8{0} ** 256,
    host_len: usize = 0,
    port: u16 = 80,
    is_tls: bool = false,

    pub fn init(host: []const u8, port: u16, is_tls: bool) HostKey {
        var key = HostKey{ .port = port, .is_tls = is_tls };
        key.host_len = @min(host.len, 256);
        @memcpy(key.host[0..key.host_len], host[0..key.host_len]);
        return key;
    }

    pub fn getHost(self: *const HostKey) []const u8 {
        return self.host[0..self.host_len];
    }

    pub fn eql(self: HostKey, other: HostKey) bool {
        if (self.port != other.port) return false;
        if (self.is_tls != other.is_tls) return false;
        if (self.host_len != other.host_len) return false;
        return std.mem.eql(u8, self.host[0..self.host_len], other.host[0..other.host_len]);
    }

    pub fn hash(self: HostKey) u64 {
        var hasher = std.hash.Wyhash.init(0);
        hasher.update(self.host[0..self.host_len]);
        hasher.update(std.mem.asBytes(&self.port));
        hasher.update(std.mem.asBytes(&self.is_tls));
        return hasher.final();
    }
};

/// Pooled connection wrapper.
pub const PooledConnection = struct {
    id: u64,
    host_key: HostKey,
    state: ConnectionState,
    stats: ConnectionStats,
    user_data: ?*anyopaque,

    /// Mark connection as used.
    pub fn markUsed(self: *PooledConnection) void {
        self.stats.last_used_at_ns = @intCast(time.nowNanoseconds());
        self.stats.use_count += 1;
    }

    /// Record bytes sent.
    pub fn recordSent(self: *PooledConnection, bytes: u64) void {
        self.stats.bytes_sent += bytes;
    }

    /// Record bytes received.
    pub fn recordReceived(self: *PooledConnection, bytes: u64) void {
        self.stats.bytes_received += bytes;
    }

    /// Record error.
    pub fn recordError(self: *PooledConnection) void {
        self.stats.errors += 1;
    }

    /// Check if connection is healthy.
    pub fn isHealthy(self: *const PooledConnection, config: ConnectionPoolConfig) bool {
        const now_ns: i64 = @intCast(time.nowNanoseconds());

        // Check idle timeout
        const idle_time_ns: u64 = @intCast(@max(0, now_ns - self.stats.last_used_at_ns));
        if (idle_time_ns > config.idle_timeout_ns) {
            return false;
        }

        // Check max lifetime
        const lifetime_ns: u64 = @intCast(@max(0, now_ns - self.stats.created_at_ns));
        if (lifetime_ns > config.max_lifetime_ns) {
            return false;
        }

        // Check error rate (simple heuristic)
        if (self.stats.use_count > 10 and self.stats.errors * 2 > self.stats.use_count) {
            return false;
        }

        return self.state == .available or self.state == .in_use;
    }
};

/// Connection pool for managing reusable connections.
pub const ConnectionPool = struct {
    allocator: std.mem.Allocator,
    config: ConnectionPoolConfig,
    connections: std.ArrayListUnmanaged(PooledConnection),
    next_id: std.atomic.Value(u64),
    mutex: std.Thread.Mutex,
    total_created: std.atomic.Value(u64),
    total_reused: std.atomic.Value(u64),
    total_closed: std.atomic.Value(u64),

    /// Initialize connection pool.
    pub fn init(allocator: std.mem.Allocator, config: ConnectionPoolConfig) ConnectionPool {
        return .{
            .allocator = allocator,
            .config = config,
            .connections = .{},
            .next_id = std.atomic.Value(u64).init(1),
            .mutex = .{},
            .total_created = std.atomic.Value(u64).init(0),
            .total_reused = std.atomic.Value(u64).init(0),
            .total_closed = std.atomic.Value(u64).init(0),
        };
    }

    /// Deinitialize pool.
    pub fn deinit(self: *ConnectionPool) void {
        self.connections.deinit(self.allocator);
        self.* = undefined;
    }

    /// Acquire a connection from the pool.
    pub fn acquire(self: *ConnectionPool, host: []const u8, port: u16, is_tls: bool) !*PooledConnection {
        const key = HostKey.init(host, port, is_tls);
        return self.acquireByKey(key);
    }

    /// Acquire by host key.
    pub fn acquireByKey(self: *ConnectionPool, key: HostKey) !*PooledConnection {
        self.mutex.lock();
        defer self.mutex.unlock();

        // Try to find an available connection
        for (self.connections.items) |*conn| {
            if (conn.host_key.eql(key) and conn.state == .available) {
                if (!self.config.validate_on_acquire or conn.isHealthy(self.config)) {
                    conn.state = .in_use;
                    conn.markUsed();
                    _ = self.total_reused.fetchAdd(1, .monotonic);
                    return conn;
                } else {
                    // Mark stale for cleanup
                    conn.state = .stale;
                }
            }
        }

        // Check limits
        const host_count = self.countConnectionsForHost(key);
        if (host_count >= self.config.max_connections_per_host) {
            return error.PoolExhausted;
        }

        if (self.connections.items.len >= self.config.max_total_connections) {
            // Try to evict stale connections
            self.cleanupStale();

            if (self.connections.items.len >= self.config.max_total_connections) {
                return error.PoolExhausted;
            }
        }

        // Create new connection
        const id = self.next_id.fetchAdd(1, .monotonic);
        const now_ns: i64 = @intCast(time.nowNanoseconds());

        const conn = PooledConnection{
            .id = id,
            .host_key = key,
            .state = .in_use,
            .stats = .{
                .created_at_ns = now_ns,
                .last_used_at_ns = now_ns,
                .use_count = 1,
                .bytes_sent = 0,
                .bytes_received = 0,
                .errors = 0,
            },
            .user_data = null,
        };

        try self.connections.append(self.allocator, conn);
        _ = self.total_created.fetchAdd(1, .monotonic);

        return &self.connections.items[self.connections.items.len - 1];
    }

    /// Release a connection back to the pool.
    pub fn release(self: *ConnectionPool, conn: *PooledConnection) void {
        self.mutex.lock();
        defer self.mutex.unlock();

        if (conn.isHealthy(self.config)) {
            conn.state = .available;
        } else {
            conn.state = .stale;
        }
    }

    /// Close a connection and remove from pool.
    pub fn close(self: *ConnectionPool, conn_id: u64) void {
        self.mutex.lock();
        defer self.mutex.unlock();

        for (self.connections.items, 0..) |*conn, i| {
            if (conn.id == conn_id) {
                conn.state = .closed;
                _ = self.connections.swapRemove(i);
                _ = self.total_closed.fetchAdd(1, .monotonic);
                return;
            }
        }
    }

    /// Clean up stale and expired connections.
    pub fn cleanup(self: *ConnectionPool) void {
        self.mutex.lock();
        defer self.mutex.unlock();
        self.cleanupStale();
    }

    /// Get pool statistics.
    pub fn getStats(self: *ConnectionPool) PoolStats {
        self.mutex.lock();
        defer self.mutex.unlock();

        var available: u32 = 0;
        var in_use: u32 = 0;
        var stale: u32 = 0;

        for (self.connections.items) |conn| {
            switch (conn.state) {
                .available => available += 1,
                .in_use => in_use += 1,
                .stale => stale += 1,
                .closed => {},
            }
        }

        return .{
            .total_connections = @intCast(self.connections.items.len),
            .available_connections = available,
            .in_use_connections = in_use,
            .stale_connections = stale,
            .total_created = self.total_created.load(.monotonic),
            .total_reused = self.total_reused.load(.monotonic),
            .total_closed = self.total_closed.load(.monotonic),
            .max_connections = self.config.max_total_connections,
        };
    }

    /// Get connection by ID.
    pub fn getConnection(self: *ConnectionPool, id: u64) ?*PooledConnection {
        self.mutex.lock();
        defer self.mutex.unlock();

        for (self.connections.items) |*conn| {
            if (conn.id == id) return conn;
        }
        return null;
    }

    // Internal helpers
    fn countConnectionsForHost(self: *const ConnectionPool, key: HostKey) u32 {
        var count: u32 = 0;
        for (self.connections.items) |conn| {
            if (conn.host_key.eql(key) and conn.state != .closed) {
                count += 1;
            }
        }
        return count;
    }

    fn cleanupStale(self: *ConnectionPool) void {
        var i: usize = 0;
        while (i < self.connections.items.len) {
            if (self.connections.items[i].state == .stale or
                !self.connections.items[i].isHealthy(self.config))
            {
                _ = self.connections.swapRemove(i);
                _ = self.total_closed.fetchAdd(1, .monotonic);
            } else {
                i += 1;
            }
        }
    }
};

/// Pool statistics.
pub const PoolStats = struct {
    total_connections: u32,
    available_connections: u32,
    in_use_connections: u32,
    stale_connections: u32,
    total_created: u64,
    total_reused: u64,
    total_closed: u64,
    max_connections: u32,

    pub fn utilizationPercent(self: PoolStats) f64 {
        if (self.max_connections == 0) return 0;
        return @as(f64, @floatFromInt(self.in_use_connections)) /
            @as(f64, @floatFromInt(self.max_connections)) * 100.0;
    }

    pub fn reuseRate(self: PoolStats) f64 {
        const total = self.total_created + self.total_reused;
        if (total == 0) return 0;
        return @as(f64, @floatFromInt(self.total_reused)) / @as(f64, @floatFromInt(total));
    }
};

/// Connection pool builder.
pub const PoolBuilder = struct {
    config: ConnectionPoolConfig,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) PoolBuilder {
        return .{
            .config = .{},
            .allocator = allocator,
        };
    }

    pub fn maxConnectionsPerHost(self: *PoolBuilder, max: u32) *PoolBuilder {
        self.config.max_connections_per_host = max;
        return self;
    }

    pub fn maxTotalConnections(self: *PoolBuilder, max: u32) *PoolBuilder {
        self.config.max_total_connections = max;
        return self;
    }

    pub fn idleTimeout(self: *PoolBuilder, timeout_ns: u64) *PoolBuilder {
        self.config.idle_timeout_ns = timeout_ns;
        return self;
    }

    pub fn maxLifetime(self: *PoolBuilder, lifetime_ns: u64) *PoolBuilder {
        self.config.max_lifetime_ns = lifetime_ns;
        return self;
    }

    pub fn enableKeepalive(self: *PoolBuilder, enable: bool) *PoolBuilder {
        self.config.enable_keepalive = enable;
        return self;
    }

    pub fn validateOnAcquire(self: *PoolBuilder, validate: bool) *PoolBuilder {
        self.config.validate_on_acquire = validate;
        return self;
    }

    pub fn build(self: *PoolBuilder) ConnectionPool {
        return ConnectionPool.init(self.allocator, self.config);
    }
};

test "connection pool acquire and release" {
    const allocator = std.testing.allocator;
    var pool = ConnectionPool.init(allocator, .{
        .max_connections_per_host = 5,
        .max_total_connections = 10,
    });
    defer pool.deinit();

    // Acquire connection
    const conn = try pool.acquire("example.com", 443, true);
    try std.testing.expectEqual(ConnectionState.in_use, conn.state);

    // Release connection
    pool.release(conn);
    try std.testing.expectEqual(ConnectionState.available, conn.state);
}

test "connection pool reuse" {
    const allocator = std.testing.allocator;
    var pool = ConnectionPool.init(allocator, .{});
    defer pool.deinit();

    // Acquire and release
    const conn1 = try pool.acquire("example.com", 80, false);
    const id1 = conn1.id;
    pool.release(conn1);

    // Should reuse same connection
    const conn2 = try pool.acquire("example.com", 80, false);
    try std.testing.expectEqual(id1, conn2.id);

    const stats = pool.getStats();
    try std.testing.expectEqual(@as(u64, 1), stats.total_reused);
}

test "connection pool limits" {
    const allocator = std.testing.allocator;
    var pool = ConnectionPool.init(allocator, .{
        .max_connections_per_host = 2,
        .max_total_connections = 5,
    });
    defer pool.deinit();

    // Acquire up to limit
    const c1 = try pool.acquire("example.com", 80, false);
    const c2 = try pool.acquire("example.com", 80, false);
    _ = c1;
    _ = c2;

    // Should fail when exhausted
    const result = pool.acquire("example.com", 80, false);
    try std.testing.expectError(error.PoolExhausted, result);
}

test "host key equality" {
    const key1 = HostKey.init("example.com", 443, true);
    const key2 = HostKey.init("example.com", 443, true);
    const key3 = HostKey.init("example.com", 80, false);

    try std.testing.expect(key1.eql(key2));
    try std.testing.expect(!key1.eql(key3));
}

test "pool builder" {
    const allocator = std.testing.allocator;

    var builder = PoolBuilder.init(allocator);
    var pool = builder
        .maxConnectionsPerHost(20)
        .maxTotalConnections(200)
        .enableKeepalive(true)
        .build();
    defer pool.deinit();

    try std.testing.expectEqual(@as(u32, 20), pool.config.max_connections_per_host);
    try std.testing.expectEqual(@as(u32, 200), pool.config.max_total_connections);
}
