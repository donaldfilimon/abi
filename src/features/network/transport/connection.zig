//! Connection Management
//!
//! PeerConnection and PendingRequest types for managing remote peer connections.

const std = @import("std");
const protocol = @import("protocol.zig");
const platform_time = @import("../../../foundation/mod.zig").utils;
const time = platform_time;
const sync = @import("../../../foundation/mod.zig").sync;

const NetworkAddress = protocol.NetworkAddress;
const TransportConfig = protocol.TransportConfig;
const TransportError = protocol.TransportError;

/// Request state for pending requests.
pub const PendingRequest = struct {
    request_id: u64,
    sent_at_ns: i64,
    timeout_ns: i64,
    response_data: ?[]u8 = null,
    completed: std.atomic.Value(bool),
    error_code: ?TransportError = null,
    condition: sync.Condition,
    mutex: sync.Mutex,

    pub fn init(request_id: u64, timeout_ms: u64) PendingRequest {
        return .{
            .request_id = request_id,
            .sent_at_ns = @intCast(time.nowNanoseconds()),
            .timeout_ns = @intCast(timeout_ms * 1_000_000),
            .completed = std.atomic.Value(bool).init(false),
            .condition = .{},
            .mutex = .{},
        };
    }

    pub fn isTimedOut(self: *const PendingRequest) bool {
        const elapsed: i64 = @intCast(time.nowNanoseconds());
        return (elapsed - self.sent_at_ns) > self.timeout_ns;
    }

    pub fn complete(self: *PendingRequest, response: ?[]u8, err: ?TransportError) void {
        self.mutex.lock();
        defer self.mutex.unlock();

        self.response_data = response;
        self.error_code = err;
        self.completed.store(true, .release);
        self.condition.signal();
    }

    pub fn waitForCompletion(self: *PendingRequest, timeout_ns: u64) !?[]u8 {
        self.mutex.lock();
        defer self.mutex.unlock();

        while (!self.completed.load(.acquire)) {
            self.condition.timedWait(&self.mutex, timeout_ns) catch |err| {
                if (err == error.Timeout) {
                    return TransportError.RequestTimeout;
                }
                return TransportError.Cancelled;
            };
        }

        if (self.error_code) |e| {
            return e;
        }

        return self.response_data;
    }
};

/// Connection state for a remote peer.
pub const PeerConnection = struct {
    allocator: std.mem.Allocator,
    address: []const u8,
    port: u16,
    socket_fd: ?std.posix.socket_t = null,
    state: ConnectionState,
    last_activity_ns: i64,
    bytes_sent: u64,
    bytes_received: u64,
    requests_sent: u64,
    requests_failed: u64,
    consecutive_failures: u32,
    mutex: sync.Mutex,

    pub const ConnectionState = enum {
        disconnected,
        connecting,
        connected,
        error_state,
        closing,
    };

    pub fn init(allocator: std.mem.Allocator, address: []const u8, port: u16) !PeerConnection {
        const addr_copy = try allocator.dupe(u8, address);
        return .{
            .allocator = allocator,
            .address = addr_copy,
            .port = port,
            .state = .disconnected,
            .last_activity_ns = @intCast(time.nowNanoseconds()),
            .bytes_sent = 0,
            .bytes_received = 0,
            .requests_sent = 0,
            .requests_failed = 0,
            .consecutive_failures = 0,
            .mutex = .{},
        };
    }

    pub fn deinit(self: *PeerConnection) void {
        self.close();
        self.allocator.free(self.address);
        self.* = undefined;
    }

    pub fn connect(self: *PeerConnection, config: TransportConfig) !void {
        self.mutex.lock();
        defer self.mutex.unlock();

        if (self.state == .connected) return;

        self.state = .connecting;

        // Parse address
        const addr = NetworkAddress.parseIp4(self.address, self.port) catch {
            // Try IPv6
            const addr6 = NetworkAddress.parseIp6(self.address, self.port) catch {
                self.state = .error_state;
                return TransportError.AddressParseError;
            };
            return self.connectToAddress(addr6, config);
        };

        return self.connectToAddress(addr, config);
    }

    fn connectToAddress(self: *PeerConnection, addr: NetworkAddress, config: TransportConfig) !void {
        const sock = std.c.socket(addr.family(), std.c.SOCK.STREAM, 0);
        if (sock < 0) {
            self.state = .error_state;
            return TransportError.ConnectionFailed;
        }
        const socket: std.posix.fd_t = @intCast(sock);
        errdefer _ = std.c.close(socket);

        // Set socket options
        if (config.enable_keepalive) {
            const one: c_int = 1;
            _ = std.c.setsockopt(socket, std.c.SOL.SOCKET, std.c.SO.KEEPALIVE, std.mem.asBytes(&one), @sizeOf(c_int));
        }

        // Set connect timeout via SO_RCVTIMEO/SO_SNDTIMEO
        const timeout_s = config.connect_timeout_ms / 1000;
        const timeout_us = (config.connect_timeout_ms % 1000) * 1000;
        const timeval = std.c.timeval{
            .sec = @intCast(timeout_s),
            .usec = @intCast(timeout_us),
        };
        _ = std.c.setsockopt(socket, std.c.SOL.SOCKET, std.c.SO.RCVTIMEO, std.mem.asBytes(&timeval), @sizeOf(std.c.timeval));
        _ = std.c.setsockopt(socket, std.c.SOL.SOCKET, std.c.SO.SNDTIMEO, std.mem.asBytes(&timeval), @sizeOf(std.c.timeval));

        // Connect
        if (std.c.connect(socket, &addr.any, addr.getOsSockLen()) < 0) {
            _ = std.c.close(socket);
            self.state = .error_state;
            self.consecutive_failures += 1;
            return TransportError.ConnectionFailed;
        }

        self.socket_fd = socket;
        self.state = .connected;
        self.consecutive_failures = 0;
        self.last_activity_ns = @intCast(time.nowNanoseconds());
    }

    pub fn close(self: *PeerConnection) void {
        self.mutex.lock();
        defer self.mutex.unlock();

        if (self.socket_fd) |fd| {
            self.state = .closing;
            _ = std.c.close(fd);
            self.socket_fd = null;
        }
        self.state = .disconnected;
    }

    pub fn isConnected(self: *const PeerConnection) bool {
        return self.state == .connected and self.socket_fd != null;
    }

    pub fn send(self: *PeerConnection, data: []const u8) !void {
        self.mutex.lock();
        defer self.mutex.unlock();

        const socket = self.socket_fd orelse return TransportError.NotConnected;

        var total_sent: usize = 0;
        while (total_sent < data.len) {
            const remaining = data[total_sent..];
            const rc = std.c.send(socket, remaining.ptr, remaining.len, 0);
            if (rc < 0) {
                self.requests_failed += 1;
                self.consecutive_failures += 1;
                self.state = .error_state;
                return TransportError.SendFailed;
            }
            const sent: usize = @intCast(rc);

            if (sent == 0) {
                self.state = .error_state;
                return TransportError.ConnectionClosed;
            }

            total_sent += sent;
        }

        self.bytes_sent += data.len;
        self.last_activity_ns = @intCast(time.nowNanoseconds());
    }

    pub fn receive(self: *PeerConnection, buffer: []u8) !usize {
        self.mutex.lock();
        defer self.mutex.unlock();

        const socket = self.socket_fd orelse return TransportError.NotConnected;

        const rc = std.c.recv(socket, buffer.ptr, buffer.len, 0);
        if (rc < 0) {
            self.consecutive_failures += 1;
            self.state = .error_state;
            return TransportError.ReceiveFailed;
        }
        const received: usize = @intCast(rc);

        if (received == 0) {
            self.state = .disconnected;
            return TransportError.ConnectionClosed;
        }

        self.bytes_received += received;
        self.last_activity_ns = @intCast(time.nowNanoseconds());
        return received;
    }

    pub fn receiveExact(self: *PeerConnection, buffer: []u8) !void {
        var total_received: usize = 0;
        while (total_received < buffer.len) {
            const remaining = buffer[total_received..];
            const received = try self.receive(remaining);
            total_received += received;
        }
    }
};

test "peer connection init and deinit" {
    const allocator = std.testing.allocator;

    var peer = try PeerConnection.init(allocator, "127.0.0.1", 9000);
    defer peer.deinit();

    try std.testing.expectEqualStrings("127.0.0.1", peer.address);
    try std.testing.expectEqual(@as(u16, 9000), peer.port);
    try std.testing.expect(!peer.isConnected());
}
