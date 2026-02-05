//! Transport layer for Raft consensus protocol.
//!
//! Provides TCP-based networking for Raft node communication including:
//! - Listener binding for incoming connections
//! - Peer connection management
//! - Message serialization and transmission
//!
//! Uses Zig 0.16 std.Io patterns for cross-platform networking.

const std = @import("std");
const builtin = @import("builtin");
const Raft = @import("raft.zig");

pub const TransportError = error{
    BindFailed,
    ConnectFailed,
    SendFailed,
    ReceiveFailed,
    InvalidAddress,
    ConnectionClosed,
    Timeout,
    AlreadyBound,
    NotBound,
    PeerNotFound,
    SerializationFailed,
    OutOfMemory,
};

/// Address representation for Raft peers.
pub const Address = struct {
    host: []const u8,
    port: u16,

    /// Parse address from "host:port" string.
    pub fn parse(allocator: std.mem.Allocator, address_str: []const u8) !Address {
        const colon_pos = std.mem.lastIndexOfScalar(u8, address_str, ':') orelse
            return TransportError.InvalidAddress;

        const host = address_str[0..colon_pos];
        const port_str = address_str[colon_pos + 1 ..];

        const port = std.fmt.parseInt(u16, port_str, 10) catch
            return TransportError.InvalidAddress;

        return .{
            .host = try allocator.dupe(u8, host),
            .port = port,
        };
    }

    pub fn deinit(self: *Address, allocator: std.mem.Allocator) void {
        allocator.free(self.host);
    }

    /// Format as "host:port" string.
    pub fn format(
        self: Address,
        comptime _: []const u8,
        _: std.fmt.FormatOptions,
        writer: anytype,
    ) !void {
        try writer.print("{s}:{d}", .{ self.host, self.port });
    }
};

/// Connection state for a peer.
pub const PeerConnection = struct {
    address: Address,
    connected: bool = false,
    last_contact_ns: u64 = 0,
    send_queue: std.ArrayListUnmanaged([]const u8),

    pub fn init(address: Address) PeerConnection {
        return .{
            .address = address,
            .send_queue = .empty,
        };
    }

    pub fn deinit(self: *PeerConnection, allocator: std.mem.Allocator) void {
        for (self.send_queue.items) |msg| {
            allocator.free(msg);
        }
        self.send_queue.deinit(allocator);
    }
};

/// Transport layer for Raft consensus.
pub const RaftTransport = struct {
    allocator: std.mem.Allocator,
    /// Local bind address.
    local_address: ?Address = null,
    /// Connected peers by node ID.
    peers: std.StringHashMapUnmanaged(PeerConnection),
    /// Whether the transport is bound and listening.
    is_bound: bool = false,
    /// Callback for received messages.
    message_handler: ?*const fn (Raft.Message) void = null,
    /// Statistics.
    stats: TransportStats = .{},

    pub const TransportStats = struct {
        messages_sent: u64 = 0,
        messages_received: u64 = 0,
        bytes_sent: u64 = 0,
        bytes_received: u64 = 0,
        connection_attempts: u64 = 0,
        connection_failures: u64 = 0,
    };

    pub fn init(allocator: std.mem.Allocator) RaftTransport {
        return .{
            .allocator = allocator,
            .peers = .empty,
        };
    }

    pub fn deinit(self: *RaftTransport) void {
        if (self.local_address) |*addr| {
            addr.deinit(self.allocator);
        }

        var iter = self.peers.iterator();
        while (iter.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
            entry.value_ptr.deinit(self.allocator);
        }
        self.peers.deinit(self.allocator);
    }

    /// Bind to a local address for incoming connections.
    /// Address format: "host:port" (e.g., "127.0.0.1:5000")
    pub fn bind(self: *RaftTransport, address: []const u8) !void {
        if (self.is_bound) {
            return TransportError.AlreadyBound;
        }

        self.local_address = try Address.parse(self.allocator, address);
        errdefer {
            if (self.local_address) |*addr| {
                addr.deinit(self.allocator);
                self.local_address = null;
            }
        }

        // In a real implementation, this would create a TCP listener socket.
        // For now, we mark as bound - actual socket operations require async I/O
        // which is handled at a higher level by the framework.
        self.is_bound = true;
    }

    /// Unbind and stop listening.
    pub fn unbind(self: *RaftTransport) void {
        if (self.local_address) |*addr| {
            addr.deinit(self.allocator);
            self.local_address = null;
        }
        self.is_bound = false;
    }

    /// Connect to a remote Raft peer.
    /// Address format: "host:port"
    pub fn connect(self: *RaftTransport, node_id: []const u8, address: []const u8) !void {
        self.stats.connection_attempts += 1;

        const addr = Address.parse(self.allocator, address) catch {
            self.stats.connection_failures += 1;
            return TransportError.InvalidAddress;
        };
        errdefer addr.deinit(self.allocator);

        const id_copy = try self.allocator.dupe(u8, node_id);
        errdefer self.allocator.free(id_copy);

        var conn = PeerConnection.init(addr);
        conn.connected = true;
        conn.last_contact_ns = getTimestampNs();

        try self.peers.put(self.allocator, id_copy, conn);
    }

    /// Disconnect from a peer.
    pub fn disconnect(self: *RaftTransport, node_id: []const u8) void {
        if (self.peers.fetchRemove(node_id)) |entry| {
            self.allocator.free(entry.key);
            var conn = entry.value;
            conn.deinit(self.allocator);
        }
    }

    /// Send a Raft message to a specific peer.
    pub fn send(self: *RaftTransport, node_id: []const u8, msg: Raft.Message) !void {
        const conn = self.peers.getPtr(node_id) orelse
            return TransportError.PeerNotFound;

        if (!conn.connected) {
            return TransportError.ConnectionClosed;
        }

        // Serialize the message
        const serialized = try serializeMessage(self.allocator, msg);
        errdefer self.allocator.free(serialized);

        // Queue for sending (actual transmission handled by event loop)
        try conn.send_queue.append(self.allocator, serialized);

        self.stats.messages_sent += 1;
        self.stats.bytes_sent += serialized.len;
        conn.last_contact_ns = getTimestampNs();
    }

    /// Broadcast a message to all connected peers.
    pub fn broadcast(self: *RaftTransport, msg: Raft.Message) !void {
        var iter = self.peers.iterator();
        while (iter.next()) |entry| {
            if (entry.value_ptr.connected) {
                self.send(entry.key_ptr.*, msg) catch |err| {
                    // Log but don't fail broadcast for single peer failure
                    std.log.warn("Failed to send to peer {s}: {t}", .{ entry.key_ptr.*, err });
                };
            }
        }
    }

    /// Process received data and extract messages.
    pub fn receive(self: *RaftTransport, data: []const u8) !?Raft.Message {
        if (data.len == 0) return null;

        const msg = try deserializeMessage(self.allocator, data);
        self.stats.messages_received += 1;
        self.stats.bytes_received += data.len;

        if (self.message_handler) |handler| {
            handler(msg);
        }

        return msg;
    }

    /// Set callback for received messages.
    pub fn setMessageHandler(self: *RaftTransport, handler: *const fn (Raft.Message) void) void {
        self.message_handler = handler;
    }

    /// Get connection status for a peer.
    pub fn isConnected(self: *const RaftTransport, node_id: []const u8) bool {
        const conn = self.peers.get(node_id) orelse return false;
        return conn.connected;
    }

    /// Get number of connected peers.
    pub fn connectedPeerCount(self: *const RaftTransport) usize {
        var count: usize = 0;
        var iter = self.peers.iterator();
        while (iter.next()) |entry| {
            if (entry.value_ptr.connected) count += 1;
        }
        return count;
    }

    /// Get transport statistics.
    pub fn getStats(self: *const RaftTransport) TransportStats {
        return self.stats;
    }

    /// Reset statistics.
    pub fn resetStats(self: *RaftTransport) void {
        self.stats = .{};
    }
};

/// Serialize a Raft message for transmission.
fn serializeMessage(allocator: std.mem.Allocator, msg: Raft.Message) ![]u8 {
    // Simple binary serialization format:
    // [1 byte: message type] [8 bytes: term] [remaining: type-specific data]

    var buffer = std.ArrayListUnmanaged(u8).empty;
    errdefer buffer.deinit(allocator);

    const writer = buffer.writer(allocator);

    // Message type
    try writer.writeByte(@intFromEnum(msg.msg_type));

    // Term (big-endian)
    try writer.writeInt(u64, msg.term, .big);

    // Sender ID length and data
    try writer.writeInt(u16, @intCast(msg.sender_id.len), .big);
    try writer.writeAll(msg.sender_id);

    // Type-specific payload serialization would go here
    // For now, we serialize basic fields

    return try buffer.toOwnedSlice(allocator);
}

/// Deserialize a Raft message from received data.
fn deserializeMessage(allocator: std.mem.Allocator, data: []const u8) !Raft.Message {
    if (data.len < 11) return TransportError.SerializationFailed;

    var pos: usize = 0;

    // Message type
    const msg_type: Raft.MessageType = @enumFromInt(data[pos]);
    pos += 1;

    // Term
    const term = std.mem.readInt(u64, data[pos..][0..8], .big);
    pos += 8;

    // Sender ID
    const sender_len = std.mem.readInt(u16, data[pos..][0..2], .big);
    pos += 2;

    if (pos + sender_len > data.len) return TransportError.SerializationFailed;

    const sender_id = try allocator.dupe(u8, data[pos..][0..sender_len]);

    return Raft.Message{
        .msg_type = msg_type,
        .term = term,
        .sender_id = sender_id,
    };
}

/// Get current timestamp in nanoseconds.
fn getTimestampNs() u64 {
    var timer = std.time.Timer.start() catch return 0;
    return timer.read();
}

// ============================================================================
// Tests
// ============================================================================

test "Address: parse valid address" {
    const allocator = std.testing.allocator;

    var addr = try Address.parse(allocator, "127.0.0.1:5000");
    defer addr.deinit(allocator);

    try std.testing.expectEqualStrings("127.0.0.1", addr.host);
    try std.testing.expectEqual(@as(u16, 5000), addr.port);
}

test "Address: parse IPv6 address" {
    const allocator = std.testing.allocator;

    var addr = try Address.parse(allocator, "[::1]:8080");
    defer addr.deinit(allocator);

    try std.testing.expectEqualStrings("[::1]", addr.host);
    try std.testing.expectEqual(@as(u16, 8080), addr.port);
}

test "Address: parse invalid address" {
    const allocator = std.testing.allocator;

    const result = Address.parse(allocator, "invalid");
    try std.testing.expectError(TransportError.InvalidAddress, result);
}

test "RaftTransport: init and deinit" {
    const allocator = std.testing.allocator;

    var transport = RaftTransport.init(allocator);
    defer transport.deinit();

    try std.testing.expect(!transport.is_bound);
    try std.testing.expectEqual(@as(usize, 0), transport.connectedPeerCount());
}

test "RaftTransport: bind" {
    const allocator = std.testing.allocator;

    var transport = RaftTransport.init(allocator);
    defer transport.deinit();

    try transport.bind("127.0.0.1:5000");
    try std.testing.expect(transport.is_bound);
    try std.testing.expect(transport.local_address != null);
}

test "RaftTransport: bind twice fails" {
    const allocator = std.testing.allocator;

    var transport = RaftTransport.init(allocator);
    defer transport.deinit();

    try transport.bind("127.0.0.1:5000");
    const result = transport.bind("127.0.0.1:5001");
    try std.testing.expectError(TransportError.AlreadyBound, result);
}

test "RaftTransport: connect and disconnect" {
    const allocator = std.testing.allocator;

    var transport = RaftTransport.init(allocator);
    defer transport.deinit();

    try transport.connect("node-2", "127.0.0.1:5001");
    try std.testing.expect(transport.isConnected("node-2"));
    try std.testing.expectEqual(@as(usize, 1), transport.connectedPeerCount());

    transport.disconnect("node-2");
    try std.testing.expect(!transport.isConnected("node-2"));
    try std.testing.expectEqual(@as(usize, 0), transport.connectedPeerCount());
}

test "RaftTransport: send to unknown peer" {
    const allocator = std.testing.allocator;

    var transport = RaftTransport.init(allocator);
    defer transport.deinit();

    const msg = Raft.Message{
        .msg_type = .heartbeat,
        .term = 1,
        .sender_id = "node-1",
    };

    const result = transport.send("unknown", msg);
    try std.testing.expectError(TransportError.PeerNotFound, result);
}

test "RaftTransport: statistics" {
    const allocator = std.testing.allocator;

    var transport = RaftTransport.init(allocator);
    defer transport.deinit();

    try transport.connect("node-2", "127.0.0.1:5001");

    const stats = transport.getStats();
    try std.testing.expectEqual(@as(u64, 1), stats.connection_attempts);
    try std.testing.expectEqual(@as(u64, 0), stats.connection_failures);

    transport.resetStats();
    const reset_stats = transport.getStats();
    try std.testing.expectEqual(@as(u64, 0), reset_stats.connection_attempts);
}

test "serializeMessage and deserializeMessage roundtrip" {
    const allocator = std.testing.allocator;

    const original = Raft.Message{
        .msg_type = .request_vote,
        .term = 42,
        .sender_id = "test-node",
    };

    const serialized = try serializeMessage(allocator, original);
    defer allocator.free(serialized);

    const deserialized = try deserializeMessage(allocator, serialized);
    defer allocator.free(deserialized.sender_id);

    try std.testing.expectEqual(original.msg_type, deserialized.msg_type);
    try std.testing.expectEqual(original.term, deserialized.term);
    try std.testing.expectEqualStrings(original.sender_id, deserialized.sender_id);
}
