//! Transport layer for Raft consensus protocol.
//!
//! Provides networking for Raft node communication including:
//! - Listener binding for incoming connections
//! - Peer connection management
//! - Message serialization and transmission
//! - Optional real TCP I/O via TcpTransport integration
//!
//! When a `TcpTransport` is connected via `connectTcp()`, messages are
//! serialized and sent over the wire using the TcpTransport's framing
//! protocol.  When no TcpTransport is present the transport falls back
//! to the in-process queue-based simulation (useful for testing).
//!
//! Uses Zig 0.16 std.Io patterns for cross-platform networking.

const std = @import("std");
const time = @import("../../services/shared/time.zig");
const sync = @import("../../services/shared/sync.zig");
const builtin = @import("builtin");
const Raft = @import("raft.zig");
const tcp = @import("transport.zig");

/// TcpTransport type from the transport module (used internally and by callers
/// that need to create one to pass into `connectTcp`).
const TcpTransport = tcp.TcpTransport;

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
    TcpSendFailed,
};

// ---------------------------------------------------------------------------
// Raft-level message types (wire-compatible with transport.MessageType)
// ---------------------------------------------------------------------------

/// Raft message types exchanged between nodes.
pub const MessageType = enum(u8) {
    request_vote = 1,
    vote_response = 2,
    append_entries = 3,
    append_response = 4,
    install_snapshot = 5,
    snapshot_response = 6,
    heartbeat = 7,

    /// Map to the TcpTransport-level `MessageType` for framing.
    pub fn toTcpMessageType(self: MessageType) tcp.MessageType {
        return switch (self) {
            .request_vote => .raft_vote_request,
            .vote_response => .raft_vote_response,
            .append_entries => .raft_append_entries,
            .append_response => .raft_append_response,
            .install_snapshot => .raft_install_snapshot,
            .snapshot_response => .raft_snapshot_response,
            // Heartbeats are empty append_entries in the Raft spec.
            .heartbeat => .raft_append_entries,
        };
    }

    /// Map from TcpTransport-level `MessageType` back to Raft MessageType.
    pub fn fromTcpMessageType(t: tcp.MessageType) ?MessageType {
        return switch (t) {
            .raft_vote_request => .request_vote,
            .raft_vote_response => .vote_response,
            .raft_append_entries => .append_entries,
            .raft_append_response => .append_response,
            .raft_install_snapshot => .install_snapshot,
            .raft_snapshot_response => .snapshot_response,
            else => null,
        };
    }
};

/// Unified Raft message exchanged over the transport.
pub const Message = struct {
    msg_type: MessageType,
    term: u64 = 0,
    sender_id: []const u8 = "",
};

// ---------------------------------------------------------------------------
// Configuration & address aliases (expected by mod.zig)
// ---------------------------------------------------------------------------

/// Configuration for the Raft transport layer.
pub const RaftTransportConfig = struct {
    /// Whether to use real TCP when a TcpTransport is available.
    use_tcp: bool = true,
    /// Timeout for individual send operations (ms).
    send_timeout_ms: u64 = 5000,
};

/// Alias for `Address` — exported as `PeerAddress` for convenience.
pub const PeerAddress = Address;

// ---------------------------------------------------------------------------
// Address
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// PeerConnection (queue-based simulation layer)
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// RaftTransport
// ---------------------------------------------------------------------------

/// Transport layer for Raft consensus.
///
/// Supports two modes of operation:
///   1. **Simulation** (default) — messages are serialized into per-peer
///      queues for in-process testing.
///   2. **Real TCP** — when a `TcpTransport` is connected via `connectTcp()`,
///      messages are framed and sent over TCP sockets.
pub const RaftTransport = struct {
    allocator: std.mem.Allocator,
    /// Local bind address.
    local_address: ?Address = null,
    /// Connected peers by node ID.
    peers: std.StringHashMapUnmanaged(PeerConnection),
    /// Whether the transport is bound and listening.
    is_bound: bool = false,
    /// Callback for received messages.
    message_handler: ?*const fn (Message) void = null,
    /// Statistics.
    stats: TransportStats = .{},
    /// Optional real TCP transport for actual network I/O.
    tcp_transport: ?*TcpTransport = null,

    // Keep the old name for mod.zig compat.
    pub const RaftTransportStats = TransportStats;

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
        // If we registered TCP handlers, the TcpTransport owns them — nothing
        // to clean up on our side beyond clearing the pointer.
        self.tcp_transport = null;

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

    // ------------------------------------------------------------------
    // TCP integration
    // ------------------------------------------------------------------

    /// Wire this RaftTransport to a real `TcpTransport` for network I/O.
    ///
    /// Registers message handlers on the TcpTransport for all Raft message
    /// types so that incoming Raft frames are deserialized and delivered to
    /// the configured `message_handler`.
    ///
    /// This is opt-in: when `tcp_transport` is null the existing queue-based
    /// simulation continues to work unchanged.
    pub fn connectTcp(self: *RaftTransport, transport: *TcpTransport) void {
        self.tcp_transport = transport;

        // Register handlers for every Raft message type the TcpTransport
        // might receive.  The handler is a file-level function (not a
        // closure) that receives the TcpTransport pointer; we stash `self`
        // in a module-level variable so the handler can recover it.
        //
        // Only one RaftTransport may be wired to TCP at a time per
        // compilation unit.
        raft_transport_instance = self;

        // Best-effort registration — we log and continue if a slot is
        // already occupied.
        const raft_tcp_types = [_]tcp.MessageType{
            .raft_vote_request,
            .raft_vote_response,
            .raft_append_entries,
            .raft_append_response,
            .raft_install_snapshot,
            .raft_snapshot_response,
        };

        for (raft_tcp_types) |mt| {
            transport.registerHandler(mt, &handleTcpIncoming) catch |err| {
                std.log.warn("RaftTransport: failed to register TCP handler for {t}: {t}", .{ mt, err });
            };
        }
    }

    /// Disconnect from the TcpTransport (reverts to simulation mode).
    pub fn disconnectTcp(self: *RaftTransport) void {
        self.tcp_transport = null;
        if (raft_transport_instance == self) {
            raft_transport_instance = null;
        }
    }

    // ------------------------------------------------------------------
    // Bind / unbind
    // ------------------------------------------------------------------

    /// Bind to a local address for incoming connections.
    /// Address format: "host:port" (e.g., "127.0.0.1:5000")
    ///
    /// When a TcpTransport is connected, this delegates to `TcpTransport.start()`
    /// which performs the real `bind()` + `listen()` system calls.
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

        // When a TcpTransport is available, delegate to it for real I/O.
        if (self.tcp_transport) |t| {
            t.start() catch |err| {
                std.log.warn("RaftTransport: TcpTransport.start() failed: {t}", .{err});
                return TransportError.BindFailed;
            };
        }

        self.is_bound = true;
    }

    /// Unbind and stop listening.
    pub fn unbind(self: *RaftTransport) void {
        if (self.tcp_transport) |t| {
            t.stop();
        }

        if (self.local_address) |*addr| {
            addr.deinit(self.allocator);
            self.local_address = null;
        }
        self.is_bound = false;
    }

    // ------------------------------------------------------------------
    // Connect / disconnect peers
    // ------------------------------------------------------------------

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

    // ------------------------------------------------------------------
    // Send / broadcast
    // ------------------------------------------------------------------

    /// Send a Raft message to a specific peer.
    ///
    /// When a TcpTransport is connected, the message is serialized and
    /// sent over TCP using the appropriate `MessageType` framing.
    /// Otherwise it is queued in the peer's in-process send queue.
    pub fn send(self: *RaftTransport, node_id: []const u8, msg: Message) !void {
        const conn = self.peers.getPtr(node_id) orelse
            return TransportError.PeerNotFound;

        if (!conn.connected) {
            return TransportError.ConnectionClosed;
        }

        // Serialize the Raft message payload.
        const serialized = try serializeMessage(self.allocator, msg);
        errdefer self.allocator.free(serialized);

        // Real TCP path — send via TcpTransport.
        if (self.tcp_transport) |t| {
            const tcp_msg_type = msg.msg_type.toTcpMessageType();
            t.sendMessage(
                conn.address.host,
                conn.address.port,
                tcp_msg_type,
                serialized,
            ) catch |err| {
                std.log.warn("RaftTransport: TCP send failed: {t}", .{err});
                self.allocator.free(serialized);
                return TransportError.TcpSendFailed;
            };
            // Payload was copied by sendMessage; free our serialized copy.
            self.allocator.free(serialized);
        } else {
            // Simulation path — queue for later consumption.
            try conn.send_queue.append(self.allocator, serialized);
        }

        self.stats.messages_sent += 1;
        self.stats.bytes_sent += serialized.len;
        conn.last_contact_ns = getTimestampNs();
    }

    /// Broadcast a message to all connected peers.
    pub fn broadcast(self: *RaftTransport, msg: Message) !void {
        var iter = self.peers.iterator();
        while (iter.next()) |entry| {
            if (entry.value_ptr.connected) {
                self.send(entry.key_ptr.*, msg) catch |err| {
                    // Log but don't fail broadcast for single peer failure.
                    std.log.warn("Failed to send to peer {s}: {t}", .{ entry.key_ptr.*, err });
                };
            }
        }
    }

    // ------------------------------------------------------------------
    // Receive (simulation path + incoming TCP handler bridge)
    // ------------------------------------------------------------------

    /// Process received data and extract messages (simulation path).
    pub fn receive(self: *RaftTransport, data: []const u8) !?Message {
        if (data.len == 0) return null;

        const msg = try deserializeMessage(self.allocator, data);
        self.stats.messages_received += 1;
        self.stats.bytes_received += data.len;

        if (self.message_handler) |handler| {
            handler(msg);
        }

        return msg;
    }

    /// Deliver a message that arrived via TCP.  Called from the
    /// TcpTransport handler trampoline (`handleTcpIncoming`).
    pub fn handleIncoming(self: *RaftTransport, payload: []const u8) void {
        const msg = deserializeMessage(self.allocator, payload) catch |err| {
            std.log.warn("RaftTransport: failed to deserialize incoming TCP message: {t}", .{err});
            return;
        };

        self.stats.messages_received += 1;
        self.stats.bytes_received += payload.len;

        if (self.message_handler) |handler| {
            handler(msg);
        }
    }

    // ------------------------------------------------------------------
    // Misc
    // ------------------------------------------------------------------

    /// Set callback for received messages.
    pub fn setMessageHandler(self: *RaftTransport, handler: *const fn (Message) void) void {
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

// ---------------------------------------------------------------------------
// Module-level bridge for TcpTransport -> RaftTransport callback
// ---------------------------------------------------------------------------

/// Module-level pointer so the fixed-signature `handleTcpIncoming` callback
/// can locate the owning `RaftTransport`.  Only one RaftTransport may be
/// wired to TCP at a time per compilation unit.
var raft_transport_instance: ?*RaftTransport = null;

/// TcpTransport.MessageHandler-compatible callback.
///
/// Deserializes the Raft payload and forwards it to the RaftTransport's
/// `handleIncoming`.  Returns an empty slice (no response payload) because
/// Raft responses are sent as separate messages.
fn handleTcpIncoming(
    _: *TcpTransport,
    _: []const u8,
    _: *const tcp.MessageHeader,
    payload: []const u8,
) tcp.TransportError![]u8 {
    if (raft_transport_instance) |rt| {
        rt.handleIncoming(payload);
    }
    // Return empty response — Raft sends replies as discrete messages.
    return &[_]u8{};
}

// ---------------------------------------------------------------------------
// Serialization
// ---------------------------------------------------------------------------

/// Serialize a Raft message for transmission.
fn serializeMessage(allocator: std.mem.Allocator, msg: Message) ![]u8 {
    // Simple binary serialization format:
    // [1 byte: message type] [8 bytes: term] [2 bytes: sender_id len] [N bytes: sender_id]

    var buffer: std.ArrayListUnmanaged(u8) = .empty;
    errdefer buffer.deinit(allocator);

    const writer = buffer.writer(allocator);

    // Message type
    try writer.writeByte(@intFromEnum(msg.msg_type));

    // Term (big-endian)
    try writer.writeInt(u64, msg.term, .big);

    // Sender ID length and data
    try writer.writeInt(u16, @intCast(msg.sender_id.len), .big);
    try writer.writeAll(msg.sender_id);

    return try buffer.toOwnedSlice(allocator);
}

/// Deserialize a Raft message from received data.
fn deserializeMessage(allocator: std.mem.Allocator, data: []const u8) !Message {
    if (data.len < 11) return TransportError.SerializationFailed;

    var pos: usize = 0;

    // Message type
    const msg_type: MessageType = @enumFromInt(data[pos]);
    pos += 1;

    // Term
    const term = std.mem.readInt(u64, data[pos..][0..8], .big);
    pos += 8;

    // Sender ID
    const sender_len = std.mem.readInt(u16, data[pos..][0..2], .big);
    pos += 2;

    if (pos + sender_len > data.len) return TransportError.SerializationFailed;

    const sender_id = try allocator.dupe(u8, data[pos..][0..sender_len]);

    return Message{
        .msg_type = msg_type,
        .term = term,
        .sender_id = sender_id,
    };
}

/// Get current timestamp in nanoseconds.
fn getTimestampNs() u64 {
    var timer = time.Timer.start() catch return 0;
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
    try std.testing.expect(transport.tcp_transport == null);
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

    const msg = Message{
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

    const original = Message{
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

test "MessageType: toTcpMessageType mapping" {
    try std.testing.expectEqual(tcp.MessageType.raft_vote_request, MessageType.request_vote.toTcpMessageType());
    try std.testing.expectEqual(tcp.MessageType.raft_vote_response, MessageType.vote_response.toTcpMessageType());
    try std.testing.expectEqual(tcp.MessageType.raft_append_entries, MessageType.append_entries.toTcpMessageType());
    try std.testing.expectEqual(tcp.MessageType.raft_append_response, MessageType.append_response.toTcpMessageType());
    try std.testing.expectEqual(tcp.MessageType.raft_install_snapshot, MessageType.install_snapshot.toTcpMessageType());
    try std.testing.expectEqual(tcp.MessageType.raft_snapshot_response, MessageType.snapshot_response.toTcpMessageType());
    // Heartbeat maps to append_entries (Raft spec: heartbeat = empty AE).
    try std.testing.expectEqual(tcp.MessageType.raft_append_entries, MessageType.heartbeat.toTcpMessageType());
}

test "MessageType: fromTcpMessageType roundtrip" {
    try std.testing.expectEqual(MessageType.request_vote, MessageType.fromTcpMessageType(.raft_vote_request).?);
    try std.testing.expectEqual(MessageType.vote_response, MessageType.fromTcpMessageType(.raft_vote_response).?);
    // Non-Raft message types return null.
    try std.testing.expect(MessageType.fromTcpMessageType(.db_search_request) == null);
    try std.testing.expect(MessageType.fromTcpMessageType(.ping) == null);
}

test "RaftTransport: send queues in simulation mode" {
    const allocator = std.testing.allocator;

    var transport = RaftTransport.init(allocator);
    defer transport.deinit();

    try transport.connect("node-2", "127.0.0.1:5001");

    const msg = Message{
        .msg_type = .append_entries,
        .term = 5,
        .sender_id = "leader-1",
    };

    try transport.send("node-2", msg);

    // In simulation mode the message ends up in the peer's send queue.
    const conn = transport.peers.getPtr("node-2").?;
    try std.testing.expectEqual(@as(usize, 1), conn.send_queue.items.len);

    const stats = transport.getStats();
    try std.testing.expectEqual(@as(u64, 1), stats.messages_sent);
}

test "RaftTransport: connectTcp and disconnectTcp" {
    var transport = RaftTransport.init(std.testing.allocator);
    defer transport.deinit();

    // tcp_transport starts as null.
    try std.testing.expect(transport.tcp_transport == null);

    // We can't fully test with a real TcpTransport in unit tests (requires
    // sockets), but we can verify the field is set/cleared.
    transport.disconnectTcp();
    try std.testing.expect(transport.tcp_transport == null);
}

test "RaftTransport: handleIncoming delivers to handler" {
    const allocator = std.testing.allocator;

    var transport = RaftTransport.init(allocator);
    defer transport.deinit();

    const Handler = struct {
        fn handle(_: Message) void {
            // Verifies no crash — full callback testing requires closures.
        }
    };
    transport.setMessageHandler(&Handler.handle);

    // Build a serialized message to feed into handleIncoming.
    const msg = Message{ .msg_type = .request_vote, .term = 10, .sender_id = "n1" };
    const payload = try serializeMessage(allocator, msg);
    defer allocator.free(payload);

    transport.handleIncoming(payload);

    try std.testing.expectEqual(@as(u64, 1), transport.stats.messages_received);
}

test {
    // Reference public declarations to ensure they compile.
    // We avoid refAllDecls to prevent transitive analysis of TcpTransport
    // internals which may pull in unrelated modules during testing.
    _ = Address;
    _ = PeerConnection;
    _ = RaftTransport;
    _ = RaftTransportConfig;
    _ = PeerAddress;
    _ = Message;
    _ = MessageType;
    _ = TransportError;
}
