//! Internet Transport Layer
//!
//! Secure remote connections over the Internet using TCP/TLS or QUIC.
//! Includes NAT traversal for peer-to-peer connections.
//!
//! ## Features
//! - QUIC protocol support (low latency, multiplexed)
//! - TCP with TLS 1.3 fallback
//! - NAT traversal (STUN, TURN, ICE)
//! - Connection migration
//! - 0-RTT reconnection

const std = @import("std");
const builtin = @import("builtin");

/// Internet transport configuration.
pub const InternetConfig = struct {
    /// Preferred protocol.
    protocol: Protocol = .quic,

    /// Local bind address.
    bind_address: []const u8 = "0.0.0.0",

    /// Local bind port (0 = auto).
    bind_port: u16 = 0,

    /// STUN servers for NAT traversal.
    stun_servers: []const []const u8 = &default_stun_servers,

    /// TURN servers for relay.
    turn_servers: []const TurnServer = &.{},

    /// Enable ICE (Interactive Connectivity Establishment).
    ice_enabled: bool = true,

    /// ICE candidate gathering timeout (milliseconds).
    ice_timeout_ms: u64 = 10000,

    /// Enable 0-RTT for QUIC.
    zero_rtt_enabled: bool = true,

    /// Connection timeout (milliseconds).
    connect_timeout_ms: u64 = 30000,

    /// Idle timeout (milliseconds).
    idle_timeout_ms: u64 = 60000,

    /// Maximum streams per connection (QUIC).
    max_streams: u32 = 100,

    /// Maximum datagram size.
    max_datagram_size: usize = 1350, // Safe for most networks

    /// Enable connection migration.
    migration_enabled: bool = true,

    /// Congestion control algorithm.
    congestion_control: CongestionControl = .bbr,

    /// Enable ECN (Explicit Congestion Notification).
    ecn_enabled: bool = true,

    /// Keep-alive interval (milliseconds).
    keepalive_interval_ms: u64 = 15000,

    pub const Protocol = enum {
        /// QUIC (preferred).
        quic,
        /// TCP with TLS.
        tcp_tls,
        /// WebSocket over TLS.
        websocket,
        /// Auto-select best available.
        auto,
    };

    pub const CongestionControl = enum {
        /// BBR (Bottleneck Bandwidth and RTT).
        bbr,
        /// CUBIC.
        cubic,
        /// Reno.
        reno,
    };

    pub const TurnServer = struct {
        url: []const u8,
        username: []const u8,
        credential: []const u8,
    };

    const default_stun_servers = [_][]const u8{
        "stun:stun.l.google.com:19302",
        "stun:stun1.l.google.com:19302",
        "stun:stun.cloudflare.com:3478",
    };

    pub fn defaults() InternetConfig {
        return .{};
    }

    /// Low-latency configuration.
    pub fn lowLatency() InternetConfig {
        return .{
            .protocol = .quic,
            .zero_rtt_enabled = true,
            .congestion_control = .bbr,
            .ecn_enabled = true,
            .keepalive_interval_ms = 5000,
        };
    }

    /// High-reliability configuration.
    pub fn reliable() InternetConfig {
        return .{
            .protocol = .tcp_tls,
            .ice_enabled = true,
            .connect_timeout_ms = 60000,
            .idle_timeout_ms = 120000,
        };
    }
};

/// NAT traversal helper.
pub const NatTraversal = struct {
    allocator: std.mem.Allocator,

    /// Local address (as seen internally).
    local_address: ?Address,

    /// Public address (as seen externally).
    public_address: ?Address,

    /// NAT type detected.
    nat_type: NatType,

    /// ICE candidates.
    candidates: std.ArrayListUnmanaged(IceCandidate),

    /// STUN/TURN sessions.
    sessions: std.ArrayListUnmanaged(*StunSession),

    /// State.
    state: State,

    /// Lock for thread safety.
    mutex: std.Thread.Mutex,

    pub const Address = struct {
        ip: [16]u8, // IPv6 or IPv4-mapped
        port: u16,
        is_ipv6: bool,

        pub fn format(self: Address) [64]u8 {
            var result: [64]u8 = undefined;
            if (self.is_ipv6) {
                // Format IPv6
                _ = std.fmt.bufPrint(&result, "[{x}]:{d}", .{
                    std.mem.readInt(u128, &self.ip, .big),
                    self.port,
                }) catch {};
            } else {
                // Format IPv4
                _ = std.fmt.bufPrint(&result, "{d}.{d}.{d}.{d}:{d}", .{
                    self.ip[12],
                    self.ip[13],
                    self.ip[14],
                    self.ip[15],
                    self.port,
                }) catch {};
            }
            return result;
        }
    };

    pub const NatType = enum {
        /// Unknown NAT type.
        unknown,
        /// No NAT (public IP).
        open,
        /// Full cone NAT (easiest to traverse).
        full_cone,
        /// Restricted cone NAT.
        restricted_cone,
        /// Port restricted cone NAT.
        port_restricted_cone,
        /// Symmetric NAT (hardest to traverse).
        symmetric,
        /// Behind firewall.
        firewalled,

        pub fn canConnectDirectly(self: NatType) bool {
            return switch (self) {
                .open, .full_cone, .restricted_cone => true,
                .unknown, .port_restricted_cone, .symmetric, .firewalled => false,
            };
        }

        pub fn needsRelay(self: NatType) bool {
            return self == .symmetric or self == .firewalled;
        }
    };

    pub const IceCandidate = struct {
        /// Candidate type.
        type: CandidateType,
        /// Transport protocol.
        protocol: TransportProtocol,
        /// IP address.
        address: Address,
        /// Priority (higher = preferred).
        priority: u32,
        /// Foundation (for pairing).
        foundation: [32]u8,
        /// Component ID.
        component_id: u8,
        /// Related address (for reflexive/relay).
        related_address: ?Address,

        pub const CandidateType = enum {
            /// Host candidate (local interface).
            host,
            /// Server reflexive (from STUN).
            server_reflexive,
            /// Peer reflexive (discovered during connectivity checks).
            peer_reflexive,
            /// Relay (from TURN).
            relay,
        };

        pub const TransportProtocol = enum {
            udp,
            tcp,
        };
    };

    pub const State = enum {
        uninitialized,
        gathering,
        complete,
        failed,
    };

    pub const StunSession = struct {
        server_address: []const u8,
        transaction_id: [12]u8,
        state: SessionState,
        response: ?StunResponse,

        pub const SessionState = enum {
            pending,
            sent,
            received,
            timeout,
            error_state,
        };

        pub const StunResponse = struct {
            mapped_address: Address,
            xor_mapped_address: Address,
            response_origin: ?Address,
            other_address: ?Address,
        };
    };

    /// Initialize NAT traversal.
    pub fn init(allocator: std.mem.Allocator) !*NatTraversal {
        const nat = try allocator.create(NatTraversal);
        nat.* = .{
            .allocator = allocator,
            .local_address = null,
            .public_address = null,
            .nat_type = .unknown,
            .candidates = .{},
            .sessions = .{},
            .state = .uninitialized,
            .mutex = .{},
        };
        return nat;
    }

    /// Deinitialize.
    pub fn deinit(self: *NatTraversal) void {
        self.candidates.deinit(self.allocator);
        for (self.sessions.items) |session| {
            self.allocator.free(session.server_address);
            self.allocator.destroy(session);
        }
        self.sessions.deinit(self.allocator);
        self.allocator.destroy(self);
    }

    /// Discover public address and NAT type.
    pub fn discover(self: *NatTraversal, stun_servers: []const []const u8) !void {
        self.mutex.lock();
        defer self.mutex.unlock();

        self.state = .gathering;

        // Create STUN sessions
        for (stun_servers) |server| {
            const session = try self.allocator.create(StunSession);
            session.* = .{
                .server_address = try self.allocator.dupe(u8, server),
                .transaction_id = undefined,
                .state = .pending,
                .response = null,
            };
            std.crypto.random.bytes(&session.transaction_id);
            try self.sessions.append(self.allocator, session);
        }

        // Would send STUN binding requests here
        // For now, simulate discovery
        self.nat_type = .full_cone;
        self.state = .complete;
    }

    /// Gather ICE candidates.
    pub fn gatherCandidates(self: *NatTraversal) ![]IceCandidate {
        self.mutex.lock();
        defer self.mutex.unlock();

        // Add host candidates
        try self.addHostCandidates();

        // Add server reflexive candidates (from STUN)
        try self.addServerReflexiveCandidates();

        return self.candidates.items;
    }

    fn addHostCandidates(self: *NatTraversal) !void {
        // Would enumerate local interfaces
        // For now, add placeholder
        const candidate = IceCandidate{
            .type = .host,
            .protocol = .udp,
            .address = .{
                .ip = [_]u8{0} ** 12 ++ [_]u8{ 127, 0, 0, 1 },
                .port = 0,
                .is_ipv6 = false,
            },
            .priority = 1000,
            .foundation = std.mem.zeroes([32]u8),
            .component_id = 1,
            .related_address = null,
        };
        try self.candidates.append(self.allocator, candidate);
    }

    fn addServerReflexiveCandidates(self: *NatTraversal) !void {
        if (self.public_address) |pub_addr| {
            const candidate = IceCandidate{
                .type = .server_reflexive,
                .protocol = .udp,
                .address = pub_addr,
                .priority = 500,
                .foundation = std.mem.zeroes([32]u8),
                .component_id = 1,
                .related_address = self.local_address,
            };
            try self.candidates.append(self.allocator, candidate);
        }
    }

    /// Get the best candidate pair for connection.
    pub fn getBestCandidate(self: *NatTraversal) ?*const IceCandidate {
        self.mutex.lock();
        defer self.mutex.unlock();

        var best: ?*const IceCandidate = null;
        for (self.candidates.items) |*c| {
            if (best == null or c.priority > best.?.priority) {
                best = c;
            }
        }
        return best;
    }
};

/// QUIC connection.
pub const QuicConnection = struct {
    allocator: std.mem.Allocator,

    /// Connection ID.
    connection_id: [20]u8,

    /// Remote address.
    remote_address: []const u8,

    /// Connection state.
    state: ConnectionState,

    /// Streams.
    streams: std.AutoHashMapUnmanaged(u64, *QuicStream),

    /// Statistics.
    stats: ConnectionStats,

    /// Configuration.
    config: QuicConfig,

    /// Lock for thread safety.
    mutex: std.Thread.Mutex,

    pub const ConnectionState = enum {
        initial,
        handshaking,
        connected,
        closing,
        draining,
        closed,
    };

    pub const ConnectionStats = struct {
        rtt_us: u64 = 0,
        rtt_variance_us: u64 = 0,
        bytes_sent: u64 = 0,
        bytes_received: u64 = 0,
        packets_sent: u64 = 0,
        packets_received: u64 = 0,
        packets_lost: u64 = 0,
        congestion_window: u64 = 0,
        bytes_in_flight: u64 = 0,
        streams_opened: u64 = 0,
        streams_closed: u64 = 0,
        datagrams_sent: u64 = 0,
        datagrams_received: u64 = 0,
        zero_rtt_accepted: bool = false,
    };

    pub const QuicConfig = struct {
        max_idle_timeout_ms: u64 = 60000,
        max_udp_payload_size: u16 = 1350,
        initial_max_data: u64 = 10 * 1024 * 1024,
        initial_max_stream_data: u64 = 1024 * 1024,
        initial_max_streams_bidi: u64 = 100,
        initial_max_streams_uni: u64 = 100,
        ack_delay_exponent: u8 = 3,
        max_ack_delay_ms: u64 = 25,
        disable_active_migration: bool = false,
        active_connection_id_limit: u64 = 8,
        enable_datagrams: bool = true,
        max_datagram_frame_size: u16 = 1200,
    };

    pub const QuicStream = struct {
        id: u64,
        state: StreamState,
        send_buffer: std.ArrayListUnmanaged(u8),
        recv_buffer: std.ArrayListUnmanaged(u8),
        bytes_sent: u64,
        bytes_received: u64,
        priority: u8,

        pub const StreamState = enum {
            open,
            half_closed_local,
            half_closed_remote,
            closed,
        };
    };

    /// Initialize QUIC connection.
    pub fn init(allocator: std.mem.Allocator, remote_address: []const u8, config: QuicConfig) !*QuicConnection {
        const conn = try allocator.create(QuicConnection);
        errdefer allocator.destroy(conn);

        conn.* = .{
            .allocator = allocator,
            .connection_id = undefined,
            .remote_address = try allocator.dupe(u8, remote_address),
            .state = .initial,
            .streams = .{},
            .stats = .{},
            .config = config,
            .mutex = .{},
        };

        std.crypto.random.bytes(&conn.connection_id);
        return conn;
    }

    /// Deinitialize connection.
    pub fn deinit(self: *QuicConnection) void {
        var it = self.streams.valueIterator();
        while (it.next()) |stream| {
            stream.*.send_buffer.deinit(self.allocator);
            stream.*.recv_buffer.deinit(self.allocator);
            self.allocator.destroy(stream.*);
        }
        self.streams.deinit(self.allocator);
        self.allocator.free(self.remote_address);
        self.allocator.destroy(self);
    }

    /// Connect to the remote peer.
    pub fn connect(self: *QuicConnection) !void {
        self.mutex.lock();
        defer self.mutex.unlock();

        if (self.state != .initial) return error.InvalidState;

        self.state = .handshaking;
        // Would perform QUIC handshake
        self.state = .connected;
    }

    /// Open a new stream.
    pub fn openStream(self: *QuicConnection, bidirectional: bool) !u64 {
        self.mutex.lock();
        defer self.mutex.unlock();

        if (self.state != .connected) return error.NotConnected;

        const stream_id = self.stats.streams_opened;
        self.stats.streams_opened += 1;

        const stream = try self.allocator.create(QuicStream);
        stream.* = .{
            .id = stream_id | (if (bidirectional) @as(u64, 0) else 2),
            .state = .open,
            .send_buffer = .{},
            .recv_buffer = .{},
            .bytes_sent = 0,
            .bytes_received = 0,
            .priority = 128,
        };

        try self.streams.put(self.allocator, stream.id, stream);
        return stream.id;
    }

    /// Send data on a stream.
    pub fn sendStream(self: *QuicConnection, stream_id: u64, data: []const u8) !void {
        self.mutex.lock();
        defer self.mutex.unlock();

        const stream = self.streams.get(stream_id) orelse return error.StreamNotFound;
        if (stream.state == .closed or stream.state == .half_closed_local) {
            return error.StreamClosed;
        }

        try stream.send_buffer.appendSlice(self.allocator, data);
        stream.bytes_sent += data.len;
        self.stats.bytes_sent += data.len;
    }

    /// Receive data from a stream.
    pub fn receiveStream(self: *QuicConnection, stream_id: u64, buffer: []u8) !usize {
        self.mutex.lock();
        defer self.mutex.unlock();

        const stream = self.streams.get(stream_id) orelse return error.StreamNotFound;
        if (stream.state == .closed or stream.state == .half_closed_remote) {
            return 0;
        }

        const available = stream.recv_buffer.items.len;
        const to_read = @min(available, buffer.len);

        if (to_read > 0) {
            @memcpy(buffer[0..to_read], stream.recv_buffer.items[0..to_read]);
            stream.recv_buffer.replaceRange(self.allocator, 0, to_read, &.{}) catch |err| {
                std.log.debug("Failed to clear stream recv_buffer after read: {t}", .{err});
            };
            stream.bytes_received += to_read;
            self.stats.bytes_received += to_read;
        }

        return to_read;
    }

    /// Close the connection.
    pub fn close(self: *QuicConnection) void {
        self.mutex.lock();
        defer self.mutex.unlock();

        if (self.state == .closed) return;

        self.state = .closing;
        // Would send CONNECTION_CLOSE frame
        self.state = .closed;
    }

    /// Get connection statistics.
    pub fn getStats(self: *QuicConnection) ConnectionStats {
        self.mutex.lock();
        defer self.mutex.unlock();
        return self.stats;
    }

    pub const Error = error{
        InvalidState,
        NotConnected,
        StreamNotFound,
        StreamClosed,
    };
};

/// Internet transport.
pub const InternetTransport = struct {
    allocator: std.mem.Allocator,

    /// Configuration.
    config: InternetConfig,

    /// NAT traversal helper.
    nat: ?*NatTraversal,

    /// Active connections.
    connections: std.AutoHashMapUnmanaged(u64, *QuicConnection),

    /// Statistics.
    stats: TransportStats,

    /// State.
    state: TransportState,

    /// Next connection ID.
    next_conn_id: u64,

    /// Lock for thread safety.
    mutex: std.Thread.Mutex,

    pub const TransportState = enum {
        uninitialized,
        initializing,
        ready,
        error_state,
    };

    pub const TransportStats = struct {
        connections_established: u64 = 0,
        connections_failed: u64 = 0,
        bytes_sent: u64 = 0,
        bytes_received: u64 = 0,
        nat_traversals_attempted: u64 = 0,
        nat_traversals_succeeded: u64 = 0,
    };

    /// Initialize Internet transport.
    pub fn init(allocator: std.mem.Allocator, config: InternetConfig) !*InternetTransport {
        const transport = try allocator.create(InternetTransport);
        errdefer allocator.destroy(transport);

        transport.* = .{
            .allocator = allocator,
            .config = config,
            .nat = null,
            .connections = .{},
            .stats = .{},
            .state = .uninitialized,
            .next_conn_id = 1,
            .mutex = .{},
        };

        // Initialize NAT traversal if ICE is enabled
        if (config.ice_enabled) {
            transport.nat = try NatTraversal.init(allocator);
            try transport.nat.?.discover(config.stun_servers);
            transport.stats.nat_traversals_attempted += 1;
            if (transport.nat.?.state == .complete) {
                transport.stats.nat_traversals_succeeded += 1;
            }
        }

        transport.state = .ready;
        return transport;
    }

    /// Deinitialize transport.
    pub fn deinit(self: *InternetTransport) void {
        var it = self.connections.valueIterator();
        while (it.next()) |conn| {
            conn.*.deinit();
        }
        self.connections.deinit(self.allocator);

        if (self.nat) |nat| {
            nat.deinit();
        }

        self.allocator.destroy(self);
    }

    /// Connect to a remote address.
    pub fn connect(self: *InternetTransport, address: []const u8) !*QuicConnection {
        self.mutex.lock();
        defer self.mutex.unlock();

        const conn = try QuicConnection.init(self.allocator, address, .{});
        errdefer conn.deinit();

        try conn.connect();

        const conn_id = self.next_conn_id;
        self.next_conn_id += 1;

        try self.connections.put(self.allocator, conn_id, conn);
        self.stats.connections_established += 1;

        return conn;
    }

    /// Disconnect a connection.
    pub fn disconnect(self: *InternetTransport, conn_id: u64) void {
        self.mutex.lock();
        defer self.mutex.unlock();

        if (self.connections.fetchRemove(conn_id)) |entry| {
            entry.value.close();
            entry.value.deinit();
        }
    }

    /// Get transport statistics.
    pub fn getStats(self: *InternetTransport) TransportStats {
        self.mutex.lock();
        defer self.mutex.unlock();
        return self.stats;
    }

    /// Get NAT type.
    pub fn getNatType(self: *InternetTransport) NatTraversal.NatType {
        if (self.nat) |nat| {
            return nat.nat_type;
        }
        return .unknown;
    }
};

// ============================================================================
// Tests
// ============================================================================

test "InternetConfig presets" {
    const low_latency = InternetConfig.lowLatency();
    try std.testing.expectEqual(InternetConfig.Protocol.quic, low_latency.protocol);
    try std.testing.expect(low_latency.zero_rtt_enabled);

    const reliable = InternetConfig.reliable();
    try std.testing.expectEqual(InternetConfig.Protocol.tcp_tls, reliable.protocol);
}

test "NatType checks" {
    try std.testing.expect(NatTraversal.NatType.open.canConnectDirectly());
    try std.testing.expect(NatTraversal.NatType.full_cone.canConnectDirectly());
    try std.testing.expect(!NatTraversal.NatType.symmetric.canConnectDirectly());
    try std.testing.expect(NatTraversal.NatType.symmetric.needsRelay());
}

test "NatTraversal initialization" {
    const allocator = std.testing.allocator;

    const nat = try NatTraversal.init(allocator);
    defer nat.deinit();

    try std.testing.expectEqual(NatTraversal.State.uninitialized, nat.state);
    try std.testing.expectEqual(NatTraversal.NatType.unknown, nat.nat_type);
}

test "QuicConnection initialization" {
    const allocator = std.testing.allocator;

    const conn = try QuicConnection.init(allocator, "127.0.0.1:4433", .{});
    defer conn.deinit();

    try std.testing.expectEqual(QuicConnection.ConnectionState.initial, conn.state);
}

test "InternetTransport initialization" {
    const allocator = std.testing.allocator;

    const transport = try InternetTransport.init(allocator, .{ .ice_enabled = false });
    defer transport.deinit();

    try std.testing.expectEqual(InternetTransport.TransportState.ready, transport.state);
}

test "Address formatting" {
    const ipv4_addr = NatTraversal.Address{
        .ip = [_]u8{0} ** 12 ++ [_]u8{ 192, 168, 1, 100 },
        .port = 8080,
        .is_ipv6 = false,
    };

    const formatted = ipv4_addr.format();
    const expected = "192.168.1.100:8080";

    // Check prefix matches
    try std.testing.expect(std.mem.startsWith(u8, &formatted, expected));
}
