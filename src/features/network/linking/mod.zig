//! Secure Linking Module
//!
//! Provides secure, high-performance connections between machines for
//! unified memory and distributed compute. Supports multiple transport
//! backends including Thunderbolt (high-speed local) and Internet (secure remote).
//!
//! ## Features
//! - Multiple transport backends (Thunderbolt, Internet/QUIC, RDMA)
//! - End-to-end encryption with mTLS
//! - NAT traversal for Internet connections
//! - Automatic failover between transports
//! - Bandwidth aggregation across multiple links
//!
//! ## Usage
//!
//! ```zig
//! const linking = @import("linking/mod.zig");
//!
//! var manager = try linking.LinkManager.init(allocator, config);
//! defer manager.deinit();
//!
//! // Create a secure link to another machine
//! const link = try manager.connect("192.168.1.100:9000", .{
//!     .transport = .auto,
//!     .encryption = .tls_1_3,
//! });
//!
//! // Use the link for memory operations
//! try link.send(data);
//! const response = try link.receive();
//! ```

const std = @import("std");
const build_options = @import("build_options");
const shared_utils = @import("../../../services/shared/utils.zig");

// Sub-module imports
pub const secure_channel = @import("secure_channel.zig");
pub const thunderbolt = @import("thunderbolt.zig");
pub const internet = @import("internet.zig");

// Re-exports
pub const SecureChannel = secure_channel.SecureChannel;
pub const ChannelConfig = secure_channel.ChannelConfig;
pub const ChannelState = secure_channel.ChannelState;
pub const ChannelStats = secure_channel.ChannelStats;
pub const EncryptionType = secure_channel.EncryptionType;

pub const ThunderboltTransport = thunderbolt.ThunderboltTransport;
pub const ThunderboltConfig = thunderbolt.ThunderboltConfig;
pub const ThunderboltDevice = thunderbolt.ThunderboltDevice;

pub const InternetTransport = internet.InternetTransport;
pub const InternetConfig = internet.InternetConfig;
pub const NatTraversal = internet.NatTraversal;
pub const QuicConnection = internet.QuicConnection;

/// Link transport type.
pub const TransportType = enum {
    /// Automatically select best available transport.
    auto,
    /// Thunderbolt 3/4 direct connection (up to 40 Gbps).
    thunderbolt,
    /// Internet via TCP with TLS.
    internet_tcp,
    /// Internet via QUIC (preferred for Internet).
    internet_quic,
    /// RDMA over Converged Ethernet (RoCE).
    rdma_roce,
    /// RDMA over InfiniBand.
    rdma_infiniband,
    /// Local loopback (for testing).
    loopback,

    /// Get the maximum theoretical bandwidth (bytes/second).
    pub fn maxBandwidth(self: TransportType) u64 {
        return switch (self) {
            .thunderbolt => 5_000_000_000, // 40 Gbps
            .rdma_infiniband => 25_000_000_000, // 200 Gbps
            .rdma_roce => 12_500_000_000, // 100 Gbps
            .internet_quic, .internet_tcp => 1_250_000_000, // 10 Gbps theoretical
            .auto, .loopback => std.math.maxInt(u64),
        };
    }

    /// Get typical latency (microseconds).
    pub fn typicalLatency(self: TransportType) u64 {
        return switch (self) {
            .thunderbolt => 1, // ~1 us
            .rdma_infiniband => 1, // ~1 us
            .rdma_roce => 2, // ~2 us
            .loopback => 0,
            .internet_tcp => 1000, // ~1 ms (varies)
            .internet_quic => 500, // ~0.5 ms
            .auto => 100,
        };
    }
};

/// Link configuration.
pub const LinkConfig = struct {
    /// Preferred transport type.
    transport: TransportType = .auto,

    /// Encryption configuration.
    encryption: EncryptionConfig = .{},

    /// Connection timeout (milliseconds).
    connect_timeout_ms: u64 = 10000,

    /// Operation timeout (milliseconds).
    operation_timeout_ms: u64 = 5000,

    /// Enable automatic reconnection.
    auto_reconnect: bool = true,

    /// Maximum reconnection attempts.
    max_reconnect_attempts: u32 = 5,

    /// Reconnection backoff (milliseconds).
    reconnect_backoff_ms: u64 = 1000,

    /// Enable bandwidth aggregation across multiple paths.
    bandwidth_aggregation: bool = false,

    /// Enable compression.
    compression_enabled: bool = true,

    /// Compression level (1-9, higher = more compression).
    compression_level: u4 = 6,

    /// Buffer size for transfers.
    buffer_size: usize = 1024 * 1024, // 1 MB

    /// Enable keepalive.
    keepalive_enabled: bool = true,

    /// Keepalive interval (milliseconds).
    keepalive_interval_ms: u64 = 30000,

    /// Maximum queued messages.
    max_queue_size: usize = 1024,

    pub const EncryptionConfig = struct {
        /// Encryption type.
        encryption_type: EncryptionType = .tls_1_3,
        /// Path to certificate file.
        cert_path: ?[]const u8 = null,
        /// Path to private key file.
        key_path: ?[]const u8 = null,
        /// Path to CA certificate for verification.
        ca_path: ?[]const u8 = null,
        /// Require mutual TLS.
        require_mtls: bool = true,
        /// Allowed cipher suites.
        cipher_suites: ?[]const CipherSuite = null,

        pub const CipherSuite = enum {
            tls_aes_256_gcm_sha384,
            tls_aes_128_gcm_sha256,
            tls_chacha20_poly1305_sha256,
        };
    };

    pub fn defaults() LinkConfig {
        return .{};
    }

    /// High-performance configuration for local links.
    pub fn highPerformance() LinkConfig {
        return .{
            .transport = .thunderbolt,
            .encryption = .{ .encryption_type = .none }, // Physical security
            .compression_enabled = false,
            .buffer_size = 16 * 1024 * 1024, // 16 MB
            .keepalive_interval_ms = 1000,
        };
    }

    /// Secure configuration for Internet links.
    pub fn secure() LinkConfig {
        return .{
            .transport = .internet_quic,
            .encryption = .{
                .encryption_type = .tls_1_3,
                .require_mtls = true,
            },
            .compression_enabled = true,
            .compression_level = 9,
        };
    }
};

/// Link state.
pub const LinkState = enum {
    disconnected,
    connecting,
    authenticating,
    connected,
    degraded,
    reconnecting,
    failed,

    pub fn isConnected(self: LinkState) bool {
        return self == .connected or self == .degraded;
    }
};

/// Link statistics.
pub const LinkStats = struct {
    /// Bytes sent.
    bytes_sent: u64 = 0,
    /// Bytes received.
    bytes_received: u64 = 0,
    /// Messages sent.
    messages_sent: u64 = 0,
    /// Messages received.
    messages_received: u64 = 0,
    /// Send errors.
    send_errors: u64 = 0,
    /// Receive errors.
    receive_errors: u64 = 0,
    /// Connection established timestamp.
    connected_at_ms: i64 = 0,
    /// Last activity timestamp.
    last_activity_ms: i64 = 0,
    /// Round-trip time (microseconds).
    rtt_us: u64 = 0,
    /// Bandwidth estimate (bytes/second).
    bandwidth_bps: u64 = 0,
    /// Packet loss rate (0.0-1.0).
    packet_loss: f32 = 0.0,
    /// Reconnection count.
    reconnections: u32 = 0,

    pub fn recordSend(self: *LinkStats, bytes: usize) void {
        self.bytes_sent += bytes;
        self.messages_sent += 1;
        self.last_activity_ms = shared_utils.unixMs();
    }

    pub fn recordReceive(self: *LinkStats, bytes: usize) void {
        self.bytes_received += bytes;
        self.messages_received += 1;
        self.last_activity_ms = shared_utils.unixMs();
    }

    pub fn recordError(self: *LinkStats, is_send: bool) void {
        if (is_send) {
            self.send_errors += 1;
        } else {
            self.receive_errors += 1;
        }
    }
};

/// A secure link to a remote machine.
pub const Link = struct {
    /// Link identifier.
    id: u64,

    /// Remote node identifier.
    remote_node_id: u64,

    /// Remote address.
    remote_address: []const u8,

    /// Current state.
    state: LinkState,

    /// Active transport type.
    transport_type: TransportType,

    /// Secure channel.
    channel: ?*SecureChannel,

    /// Configuration.
    config: LinkConfig,

    /// Statistics.
    stats: LinkStats,

    /// Allocator.
    allocator: std.mem.Allocator,

    /// Message queue for outgoing messages.
    send_queue: std.ArrayListUnmanaged(QueuedMessage),

    /// Lock for thread safety.
    mutex: std.Thread.Mutex,

    pub const QueuedMessage = struct {
        data: []const u8,
        priority: u8,
        timestamp: i64,
    };

    /// Initialize a new link.
    pub fn init(
        allocator: std.mem.Allocator,
        id: u64,
        remote_address: []const u8,
        config: LinkConfig,
    ) !*Link {
        const link = try allocator.create(Link);
        errdefer allocator.destroy(link);

        link.* = .{
            .id = id,
            .remote_node_id = 0,
            .remote_address = try allocator.dupe(u8, remote_address),
            .state = .disconnected,
            .transport_type = config.transport,
            .channel = null,
            .config = config,
            .stats = .{},
            .allocator = allocator,
            .send_queue = .{},
            .mutex = .{},
        };

        return link;
    }

    /// Deinitialize link.
    pub fn deinit(self: *Link) void {
        self.mutex.lock();
        defer self.mutex.unlock();

        if (self.channel) |ch| {
            ch.close();
            self.allocator.destroy(ch);
        }

        for (self.send_queue.items) |msg| {
            self.allocator.free(msg.data);
        }
        self.send_queue.deinit(self.allocator);

        self.allocator.free(self.remote_address);
        self.allocator.destroy(self);
    }

    /// Connect to the remote machine.
    pub fn connect(self: *Link) LinkError!void {
        self.mutex.lock();
        defer self.mutex.unlock();

        if (self.state.isConnected()) return;

        self.state = .connecting;

        // Select transport
        if (self.config.transport == .auto) {
            self.transport_type = self.selectBestTransport();
        }

        // Create secure channel
        self.channel = SecureChannel.init(self.allocator, .{
            .encryption = self.config.encryption.encryption_type,
        }) catch return error.ConnectionFailed;

        // Perform connection handshake
        self.channel.?.connect(self.remote_address) catch |err| {
            std.log.debug("Connection handshake failed for {s}: {t}", .{ self.remote_address, err });
            self.state = .failed;
            return error.ConnectionFailed;
        };

        self.state = .connected;
        self.stats.connected_at_ms = shared_utils.unixMs();
    }

    /// Disconnect from the remote machine.
    pub fn disconnect(self: *Link) void {
        self.mutex.lock();
        defer self.mutex.unlock();

        if (self.channel) |ch| {
            ch.close();
            self.allocator.destroy(ch);
            self.channel = null;
        }

        self.state = .disconnected;
    }

    /// Send data to the remote machine.
    pub fn send(self: *Link, data: []const u8) LinkError!void {
        self.mutex.lock();
        defer self.mutex.unlock();

        if (!self.state.isConnected()) {
            return error.NotConnected;
        }

        const channel = self.channel orelse return error.NotConnected;

        channel.send(data) catch |err| {
            std.log.debug("Send failed on link {d}: {t}", .{ self.id, err });
            self.stats.recordError(true);
            if (self.config.auto_reconnect) {
                self.state = .reconnecting;
            }
            return error.SendFailed;
        };

        self.stats.recordSend(data.len);
    }

    /// Receive data from the remote machine.
    pub fn receive(self: *Link, buffer: []u8) LinkError!usize {
        self.mutex.lock();
        defer self.mutex.unlock();

        if (!self.state.isConnected()) {
            return error.NotConnected;
        }

        const channel = self.channel orelse return error.NotConnected;

        const len = channel.receive(buffer) catch |err| {
            std.log.debug("Receive failed on link {d}: {t}", .{ self.id, err });
            self.stats.recordError(false);
            return error.ReceiveFailed;
        };

        self.stats.recordReceive(len);
        return len;
    }

    /// Get link statistics.
    pub fn getStats(self: *Link) LinkStats {
        self.mutex.lock();
        defer self.mutex.unlock();
        return self.stats;
    }

    /// Select the best available transport.
    fn selectBestTransport(self: *Link) TransportType {
        _ = self;
        // Check for Thunderbolt first
        if (thunderbolt.isAvailable()) {
            return .thunderbolt;
        }

        // Default to QUIC for Internet
        return .internet_quic;
    }
};

/// Link error types.
pub const LinkError = error{
    ConnectionFailed,
    NotConnected,
    SendFailed,
    ReceiveFailed,
    AuthenticationFailed,
    Timeout,
    Disconnected,
    QueueFull,
    InvalidAddress,
    TransportUnavailable,
    EncryptionFailed,
};

/// Link manager for handling multiple connections.
pub const LinkManager = struct {
    allocator: std.mem.Allocator,

    /// Active links indexed by ID.
    links: std.AutoHashMapUnmanaged(u64, *Link),

    /// Links by remote address for lookup.
    links_by_address: std.StringHashMapUnmanaged(*Link),

    /// Default configuration for new links.
    default_config: LinkConfig,

    /// Manager statistics.
    stats: ManagerStats,

    /// Next link ID.
    next_link_id: u64,

    /// Lock for thread safety.
    mutex: std.Thread.Mutex,

    pub const ManagerStats = struct {
        links_created: u64 = 0,
        links_closed: u64 = 0,
        total_bytes_sent: u64 = 0,
        total_bytes_received: u64 = 0,
        active_links: u64 = 0,
    };

    /// Initialize link manager.
    pub fn init(allocator: std.mem.Allocator, config: LinkConfig) !*LinkManager {
        if (!isEnabled()) return error.LinkingDisabled;

        const manager = try allocator.create(LinkManager);
        manager.* = .{
            .allocator = allocator,
            .links = .{},
            .links_by_address = .{},
            .default_config = config,
            .stats = .{},
            .next_link_id = 1,
            .mutex = .{},
        };

        return manager;
    }

    /// Deinitialize manager.
    pub fn deinit(self: *LinkManager) void {
        self.mutex.lock();
        defer self.mutex.unlock();

        var it = self.links.valueIterator();
        while (it.next()) |link| {
            link.*.deinit();
        }
        self.links.deinit(self.allocator);
        self.links_by_address.deinit(self.allocator);

        self.allocator.destroy(self);
    }

    /// Create and connect a new link.
    pub fn connect(
        self: *LinkManager,
        remote_address: []const u8,
        config: ?LinkConfig,
    ) LinkError!*Link {
        self.mutex.lock();
        defer self.mutex.unlock();

        // Check if link already exists
        if (self.links_by_address.get(remote_address)) |existing| {
            if (existing.state.isConnected()) {
                return existing;
            }
        }

        const link_config = config orelse self.default_config;
        const link_id = self.next_link_id;
        self.next_link_id += 1;

        const link = Link.init(
            self.allocator,
            link_id,
            remote_address,
            link_config,
        ) catch return error.ConnectionFailed;

        link.connect() catch {
            link.deinit();
            return error.ConnectionFailed;
        };

        self.links.put(self.allocator, link_id, link) catch {
            link.deinit();
            return error.ConnectionFailed;
        };

        self.stats.links_created += 1;
        self.stats.active_links += 1;

        return link;
    }

    /// Get a link by ID.
    pub fn getLink(self: *LinkManager, link_id: u64) ?*Link {
        self.mutex.lock();
        defer self.mutex.unlock();
        return self.links.get(link_id);
    }

    /// Close a link.
    pub fn closeLink(self: *LinkManager, link_id: u64) void {
        self.mutex.lock();
        defer self.mutex.unlock();

        if (self.links.fetchRemove(link_id)) |entry| {
            const link = entry.value;
            _ = self.links_by_address.remove(link.remote_address);
            link.deinit();
            self.stats.links_closed += 1;
            self.stats.active_links -= 1;
        }
    }

    /// Get all active links.
    pub fn listLinks(self: *LinkManager, allocator: std.mem.Allocator) ![]u64 {
        self.mutex.lock();
        defer self.mutex.unlock();

        var list = std.ArrayListUnmanaged(u64){};
        errdefer list.deinit(allocator);

        var it = self.links.keyIterator();
        while (it.next()) |key| {
            try list.append(allocator, key.*);
        }

        return list.toOwnedSlice(allocator);
    }

    /// Get manager statistics.
    pub fn getStats(self: *LinkManager) ManagerStats {
        self.mutex.lock();
        defer self.mutex.unlock();
        return self.stats;
    }

    pub const Error = error{
        LinkingDisabled,
    };
};

/// Check if linking is enabled at compile time.
pub fn isEnabled() bool {
    return build_options.enable_network;
}

// ============================================================================
// Tests
// ============================================================================

test "TransportType bandwidth and latency" {
    try std.testing.expect(TransportType.thunderbolt.maxBandwidth() > TransportType.internet_tcp.maxBandwidth());
    try std.testing.expect(TransportType.thunderbolt.typicalLatency() < TransportType.internet_tcp.typicalLatency());
}

test "LinkConfig presets" {
    const hp = LinkConfig.highPerformance();
    try std.testing.expectEqual(TransportType.thunderbolt, hp.transport);
    try std.testing.expect(!hp.compression_enabled);

    const secure = LinkConfig.secure();
    try std.testing.expectEqual(TransportType.internet_quic, secure.transport);
    try std.testing.expect(secure.encryption.require_mtls);
}

test "LinkState checks" {
    try std.testing.expect(LinkState.connected.isConnected());
    try std.testing.expect(LinkState.degraded.isConnected());
    try std.testing.expect(!LinkState.disconnected.isConnected());
    try std.testing.expect(!LinkState.failed.isConnected());
}

test "LinkStats recording" {
    var stats = LinkStats{};

    stats.recordSend(1000);
    try std.testing.expectEqual(@as(u64, 1000), stats.bytes_sent);
    try std.testing.expectEqual(@as(u64, 1), stats.messages_sent);

    stats.recordReceive(500);
    try std.testing.expectEqual(@as(u64, 500), stats.bytes_received);
    try std.testing.expectEqual(@as(u64, 1), stats.messages_received);

    stats.recordError(true);
    try std.testing.expectEqual(@as(u64, 1), stats.send_errors);
}
