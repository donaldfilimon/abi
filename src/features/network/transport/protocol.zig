//! Transport Protocol Types
//!
//! Shared types, enums, error types, and protocol constants for the transport layer.

const std = @import("std");

/// Network address wrapper for low-level posix socket operations.
/// Replaces the removed `std.net.Address` in Zig 0.16 by wrapping
/// `std.Io.net.IpAddress` parsing with a sockaddr layout for posix calls.
pub const NetworkAddress = struct {
    /// Raw sockaddr storage large enough for IPv4 or IPv6.
    any: std.posix.sockaddr,
    addr_len: std.posix.socklen_t,

    /// Parse an IPv4 address string and port into a NetworkAddress.
    pub fn parseIp4(host: []const u8, port: u16) !NetworkAddress {
        const Io = std.Io;
        const ip4 = Io.net.Ip4Address.parse(host, port) catch return error.InvalidAddress;
        var addr: NetworkAddress = undefined;
        // Build sockaddr.in from parsed IPv4 bytes
        const sin: *std.c.sockaddr.in = @ptrCast(@alignCast(&addr.any));
        sin.* = .{
            .port = std.mem.nativeToBig(u16, ip4.port),
            .addr = @bitCast(ip4.bytes),
        };
        addr.addr_len = @sizeOf(std.c.sockaddr.in);
        return addr;
    }

    /// Parse an IPv6 address string and port into a NetworkAddress.
    pub fn parseIp6(host: []const u8, port: u16) !NetworkAddress {
        const Io = std.Io;
        const ip6 = Io.net.Ip6Address.parse(host, port) catch return error.InvalidAddress;
        var addr: NetworkAddress = undefined;
        const sin6: *std.c.sockaddr.in6 = @ptrCast(@alignCast(&addr.any));
        sin6.* = .{
            .port = std.mem.nativeToBig(u16, ip6.port),
            .flowinfo = ip6.flow,
            .addr = ip6.bytes,
            .scope_id = if (@hasField(@TypeOf(ip6.interface), "index")) ip6.interface.index else 0,
        };
        addr.addr_len = @sizeOf(std.c.sockaddr.in6);
        return addr;
    }

    /// Return the address family as a raw integer (AF_INET or AF_INET6).
    pub fn family(self: *const NetworkAddress) c_uint {
        return @intCast(self.any.family);
    }

    /// Return the sockaddr length for posix calls.
    pub fn getOsSockLen(self: *const NetworkAddress) std.posix.socklen_t {
        return self.addr_len;
    }
};

/// Transport configuration.
pub const TransportConfig = struct {
    /// Local address to bind to.
    listen_address: []const u8 = "0.0.0.0",
    /// Local port to listen on.
    listen_port: u16 = 9000,
    /// Maximum concurrent connections.
    max_connections: u32 = 256,
    /// Connection timeout in milliseconds.
    connect_timeout_ms: u64 = 5000,
    /// Read/write timeout in milliseconds.
    io_timeout_ms: u64 = 30000,
    /// Maximum message size in bytes.
    max_message_size: usize = 16 * 1024 * 1024, // 16MB
    /// Enable TCP keepalive.
    enable_keepalive: bool = true,
    /// Keepalive interval in seconds.
    keepalive_interval_s: u32 = 30,
    /// Enable connection pooling.
    enable_pooling: bool = true,
    /// Maximum retries for failed requests.
    max_retries: u32 = 3,
    /// Base retry delay in milliseconds.
    retry_delay_ms: u64 = 100,
    /// Enable request/response logging.
    enable_logging: bool = true,
};

/// Message types for the transport protocol.
pub const MessageType = enum(u8) {
    // Raft consensus messages
    raft_vote_request = 1,
    raft_vote_response = 2,
    raft_append_entries = 3,
    raft_append_response = 4,
    raft_install_snapshot = 5,
    raft_snapshot_response = 6,

    // Database RPC messages
    db_search_request = 10,
    db_search_response = 11,
    db_insert_request = 12,
    db_insert_response = 13,
    db_delete_request = 14,
    db_delete_response = 15,
    db_update_request = 16,
    db_update_response = 17,
    db_batch_request = 18,
    db_batch_response = 19,

    // Cluster management messages
    cluster_join = 20,
    cluster_join_ack = 21,
    cluster_leave = 22,
    cluster_leave_ack = 23,
    cluster_heartbeat = 24,
    cluster_heartbeat_ack = 25,
    cluster_state_sync = 26,
    cluster_state_ack = 27,

    // Generic RPC
    rpc_request = 30,
    rpc_response = 31,
    rpc_error = 32,

    // Control messages
    ping = 100,
    pong = 101,
    close = 102,
};

/// Message header for framing.
pub const MessageHeader = extern struct {
    /// Magic number for validation.
    magic: u32 = MAGIC_NUMBER,
    /// Protocol version.
    version: u8 = PROTOCOL_VERSION,
    /// Message type.
    message_type: u8,
    /// Flags (reserved).
    flags: u16 = 0,
    /// Request ID for correlation.
    request_id: u64,
    /// Payload length.
    payload_length: u32,
    /// CRC32 checksum of payload (0 if disabled).
    checksum: u32 = 0,

    pub const SIZE: usize = @sizeOf(MessageHeader);
};

pub const MAGIC_NUMBER: u32 = 0x41424954; // "ABIT"
pub const PROTOCOL_VERSION: u8 = 1;

/// Errors for transport operations.
pub const TransportError = error{
    ConnectionFailed,
    ConnectionClosed,
    ConnectionTimeout,
    InvalidMessage,
    InvalidMagic,
    InvalidVersion,
    MessageTooLarge,
    ChecksumMismatch,
    RequestTimeout,
    SendFailed,
    ReceiveFailed,
    PoolExhausted,
    NotConnected,
    AlreadyStarted,
    NotStarted,
    AddressParseError,
    BindFailed,
    ListenFailed,
    AcceptFailed,
    OutOfMemory,
    Cancelled,
};

/// Address parsing utilities.
pub fn parseAddress(address: []const u8) !struct { host: []const u8, port: u16 } {
    // Handle IPv6 with brackets: [::1]:8080
    if (address[0] == '[') {
        const bracket_end = std.mem.indexOf(u8, address, "]") orelse return TransportError.AddressParseError;
        const host = address[1..bracket_end];

        if (address.len > bracket_end + 1 and address[bracket_end + 1] == ':') {
            const port_str = address[bracket_end + 2 ..];
            const port = std.fmt.parseInt(u16, port_str, 10) catch return TransportError.AddressParseError;
            return .{ .host = host, .port = port };
        }

        return .{ .host = host, .port = 9000 };
    }

    // Handle IPv4: 192.168.1.1:8080
    if (std.mem.lastIndexOf(u8, address, ":")) |colon_pos| {
        const host = address[0..colon_pos];
        const port_str = address[colon_pos + 1 ..];
        const port = std.fmt.parseInt(u16, port_str, 10) catch return TransportError.AddressParseError;
        return .{ .host = host, .port = port };
    }

    return .{ .host = address, .port = 9000 };
}

test "message header size" {
    try std.testing.expectEqual(@as(usize, 24), MessageHeader.SIZE);
}

test "address parsing ipv4" {
    const result = try parseAddress("192.168.1.1:8080");
    try std.testing.expectEqualStrings("192.168.1.1", result.host);
    try std.testing.expectEqual(@as(u16, 8080), result.port);
}

test "address parsing ipv4 default port" {
    const result = try parseAddress("192.168.1.1");
    try std.testing.expectEqualStrings("192.168.1.1", result.host);
    try std.testing.expectEqual(@as(u16, 9000), result.port);
}
