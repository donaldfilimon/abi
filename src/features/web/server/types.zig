//! Web Server Core Types
//!
//! Provides fundamental types for HTTP server configuration and connection management.
//! These types follow Zig 0.16 conventions and integrate with std.Io.Threaded.

const std = @import("std");
const Io = std.Io;
const time = @import("../../../services/shared/time.zig");

/// Configuration for the HTTP server.
pub const ServerConfig = struct {
    /// Host address to bind to.
    host: []const u8 = "127.0.0.1",
    /// Port to listen on.
    port: u16 = 8080,
    /// Maximum number of concurrent connections.
    max_connections: usize = 1024,
    /// Read timeout in milliseconds.
    read_timeout_ms: u64 = 30000,
    /// Write timeout in milliseconds.
    write_timeout_ms: u64 = 30000,
    /// Maximum size of HTTP headers in bytes.
    max_header_size: usize = 8192,
    /// Maximum size of request body in bytes.
    max_body_size: usize = 10 * 1024 * 1024,
    /// Enable keep-alive connections.
    keep_alive: bool = true,
    /// Keep-alive timeout in milliseconds.
    keep_alive_timeout_ms: u64 = 5000,
    /// Number of worker threads (0 = auto-detect).
    worker_threads: u32 = 0,

    /// Returns the address for binding.
    pub fn getBindAddress(self: ServerConfig) Io.net.IpAddress {
        return Io.net.IpAddress.parseIp4(self.host, self.port) catch
            Io.net.IpAddress.parseIp6(self.host, self.port) catch
            .{ .ip4 = .loopback(self.port) };
    }
};

/// Server lifecycle states.
pub const ServerState = enum {
    /// Server is stopped and not accepting connections.
    stopped,
    /// Server is in the process of starting up.
    starting,
    /// Server is running and accepting connections.
    running,
    /// Server is shutting down gracefully.
    stopping,
    /// Server encountered an error.
    errored,

    /// Returns true if the server can accept new connections.
    pub fn canAccept(self: ServerState) bool {
        return self == .running;
    }

    /// Returns true if the server is in a terminal state.
    pub fn isTerminal(self: ServerState) bool {
        return self == .stopped or self == .errored;
    }
};

/// Represents an active client connection.
pub const Connection = struct {
    /// Unique connection identifier.
    id: u64,
    /// Client socket address.
    address: Io.net.IpAddress,
    /// Connection creation timestamp (milliseconds since epoch).
    created_at: i64,
    /// Last activity timestamp.
    last_activity: i64,
    /// Number of requests handled on this connection.
    request_count: u64 = 0,
    /// Whether the connection should be kept alive.
    keep_alive: bool = true,
    /// Allocator for connection-specific allocations.
    allocator: std.mem.Allocator,

    /// Creates a new connection.
    pub fn init(allocator: std.mem.Allocator, id: u64, address: Io.net.IpAddress) Connection {
        const now = time.nowMs();
        return .{
            .id = id,
            .address = address,
            .created_at = now,
            .last_activity = now,
            .allocator = allocator,
        };
    }

    /// Updates the last activity timestamp.
    pub fn touch(self: *Connection) void {
        self.last_activity = time.nowMs();
    }

    /// Returns the connection age in milliseconds.
    pub fn ageMs(self: Connection) i64 {
        return time.nowMs() - self.created_at;
    }

    /// Returns the idle time in milliseconds.
    pub fn idleMs(self: Connection) i64 {
        return time.nowMs() - self.last_activity;
    }

    /// Checks if the connection has timed out.
    pub fn isTimedOut(self: Connection, timeout_ms: u64) bool {
        return @as(u64, @intCast(@max(0, self.idleMs()))) > timeout_ms;
    }

    /// Formats the connection address as a string.
    pub fn formatAddress(self: Connection, buf: []u8) []const u8 {
        var writer = Io.Writer.fixed(buf);
        self.address.format(&writer) catch return "<unknown>";
        return buf[0..writer.end];
    }
};

/// HTTP request method.
pub const Method = std.http.Method;

/// HTTP status codes.
pub const Status = std.http.Status;

/// Common HTTP headers.
pub const Header = struct {
    pub const content_type = "Content-Type";
    pub const content_length = "Content-Length";
    pub const connection = "Connection";
    pub const host = "Host";
    pub const user_agent = "User-Agent";
    pub const accept = "Accept";
    pub const authorization = "Authorization";
    pub const cache_control = "Cache-Control";
    pub const cookie = "Cookie";
    pub const set_cookie = "Set-Cookie";
    pub const location = "Location";
    pub const x_request_id = "X-Request-Id";
    pub const x_forwarded_for = "X-Forwarded-For";
    pub const x_real_ip = "X-Real-IP";
};

/// Common MIME types.
pub const MimeType = struct {
    pub const json = "application/json";
    pub const html = "text/html; charset=utf-8";
    pub const plain = "text/plain; charset=utf-8";
    pub const xml = "application/xml";
    pub const form = "application/x-www-form-urlencoded";
    pub const multipart = "multipart/form-data";
    pub const octet_stream = "application/octet-stream";
    pub const javascript = "application/javascript";
    pub const css = "text/css";
};

/// Server statistics.
pub const ServerStats = struct {
    /// Total connections accepted.
    total_connections: u64 = 0,
    /// Currently active connections.
    active_connections: u64 = 0,
    /// Total requests handled.
    total_requests: u64 = 0,
    /// Total bytes received.
    bytes_received: u64 = 0,
    /// Total bytes sent.
    bytes_sent: u64 = 0,
    /// Number of errors encountered.
    error_count: u64 = 0,
    /// Server start timestamp.
    started_at: i64 = 0,

    /// Returns server uptime in milliseconds.
    pub fn uptimeMs(self: ServerStats) i64 {
        if (self.started_at == 0) return 0;
        return time.nowMs() - self.started_at;
    }

    /// Returns requests per second (averaged over uptime).
    pub fn requestsPerSecond(self: ServerStats) f64 {
        const uptime_s = @as(f64, @floatFromInt(self.uptimeMs())) / 1000.0;
        if (uptime_s <= 0) return 0;
        return @as(f64, @floatFromInt(self.total_requests)) / uptime_s;
    }
};

/// Error types for server operations.
pub const ServerError = error{
    /// Server is already running.
    AlreadyRunning,
    /// Server is not running.
    NotRunning,
    /// Failed to bind to address.
    BindFailed,
    /// Connection limit reached.
    TooManyConnections,
    /// Request timeout.
    Timeout,
    /// Invalid request.
    InvalidRequest,
    /// Request too large.
    RequestTooLarge,
    /// Internal server error.
    InternalError,
    /// Connection closed by client.
    ConnectionClosed,
    /// SSL/TLS error.
    TlsError,
};

test "ServerConfig defaults" {
    const config = ServerConfig{};
    try std.testing.expectEqualStrings("127.0.0.1", config.host);
    try std.testing.expectEqual(@as(u16, 8080), config.port);
    try std.testing.expectEqual(@as(usize, 1024), config.max_connections);
}

test "ServerState transitions" {
    try std.testing.expect(ServerState.running.canAccept());
    try std.testing.expect(!ServerState.stopped.canAccept());
    try std.testing.expect(!ServerState.stopping.canAccept());
    try std.testing.expect(ServerState.stopped.isTerminal());
    try std.testing.expect(ServerState.errored.isTerminal());
    try std.testing.expect(!ServerState.running.isTerminal());
}

test "Connection lifecycle" {
    const allocator = std.testing.allocator;
    const addr: Io.net.IpAddress = .{ .ip4 = .loopback(12345) };
    var conn = Connection.init(allocator, 1, addr);

    try std.testing.expectEqual(@as(u64, 1), conn.id);
    try std.testing.expectEqual(@as(u64, 0), conn.request_count);
    try std.testing.expect(conn.keep_alive);

    // Test touch updates last_activity
    const old_activity = conn.last_activity;
    time.sleepMs(1);
    conn.touch();
    try std.testing.expect(conn.last_activity >= old_activity);
}

test "ServerStats calculations" {
    var stats = ServerStats{
        .started_at = time.nowMs() - 1000, // Started 1 second ago
        .total_requests = 100,
    };

    try std.testing.expect(stats.uptimeMs() >= 1000);
    try std.testing.expect(stats.requestsPerSecond() > 0);
}
