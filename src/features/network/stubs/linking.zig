const std = @import("std");

// Sub-module namespace stubs (matching real linking/mod.zig public API)
pub const secure_channel = struct {
    pub const SecureChannel = linking_root.SecureChannel;
    pub const ChannelConfig = linking_root.ChannelConfig;
    pub const ChannelState = linking_root.ChannelState;
    pub const ChannelStats = linking_root.ChannelStats;
    pub const EncryptionType = linking_root.EncryptionType;
};

pub const thunderbolt = struct {
    pub const ThunderboltTransport = linking_root.ThunderboltTransport;
    pub const ThunderboltConfig = linking_root.ThunderboltConfig;
    pub const ThunderboltDevice = linking_root.ThunderboltDevice;
    pub fn isAvailable() bool {
        return false;
    }
};

pub const internet = struct {
    pub const InternetTransport = linking_root.InternetTransport;
    pub const InternetConfig = linking_root.InternetConfig;
    pub const NatTraversal = linking_root.NatTraversal;
    pub const QuicConnection = linking_root.QuicConnection;
};

const linking_root = @This();

// Re-exports matching linking/mod.zig top-level public API

pub const SecureChannel = struct {
    pub fn init(_: std.mem.Allocator, _: ChannelConfig) !*@This() {
        return error.NetworkDisabled;
    }
    pub fn deinit(_: *@This()) void {}
    pub fn connect(_: *@This(), _: []const u8) !void {
        return error.NetworkDisabled;
    }
    pub fn accept(_: *@This(), _: *anyopaque) !void {
        return error.NetworkDisabled;
    }
    pub fn send(_: *@This(), _: []const u8) !void {
        return error.NetworkDisabled;
    }
    pub fn receive(_: *@This(), _: []u8) !usize {
        return error.NetworkDisabled;
    }
    pub fn close(_: *@This()) void {}
    pub fn rotateKeys(_: *@This()) !void {
        return error.NetworkDisabled;
    }
    pub fn getStats(_: *@This()) ChannelStats {
        return .{};
    }
    pub fn needsRekey(_: *@This()) bool {
        return false;
    }
};

pub const ChannelConfig = struct {
    encryption: EncryptionType = .tls_1_3,
    psk: ?[32]u8 = null,
    local_cert: ?[]const u8 = null,
    local_key: ?[]const u8 = null,
    ca_cert: ?[]const u8 = null,
    verify_peer: bool = true,
    peer_hostname: ?[]const u8 = null,
    max_message_size: usize = 16 * 1024 * 1024,
    authenticate_messages: bool = true,
    replay_protection: bool = true,
    handshake_timeout_ms: u64 = 10000,
    session_resumption: bool = true,
    key_rotation_interval_s: u64 = 3600,
};

pub const ChannelState = enum {
    uninitialized,
    handshaking,
    established,
    rekeying,
    error_state,
    closed,

    pub fn isReady(self: ChannelState) bool {
        return self == .established or self == .rekeying;
    }
};

pub const ChannelStats = struct {
    bytes_encrypted: u64 = 0,
    bytes_decrypted: u64 = 0,
    encrypt_ops: u64 = 0,
    decrypt_ops: u64 = 0,
    auth_failures: u64 = 0,
    replay_attacks: u64 = 0,
    key_rotations: u64 = 0,
    handshakes: u64 = 0,
    resumptions: u64 = 0,
    established_at_ms: i64 = 0,
    last_activity_ms: i64 = 0,
};

pub const EncryptionType = enum {
    none,
    tls_1_2,
    tls_1_3,
    noise_xx,
    wireguard,
    chacha20_poly1305,
    aes_256_gcm,

    pub fn keySize(self: EncryptionType) usize {
        return switch (self) {
            .none => 0,
            .tls_1_2, .tls_1_3 => 32,
            .noise_xx => 32,
            .wireguard => 32,
            .chacha20_poly1305 => 32,
            .aes_256_gcm => 32,
        };
    }

    pub fn nonceSize(self: EncryptionType) usize {
        return switch (self) {
            .none => 0,
            .tls_1_2, .tls_1_3 => 12,
            .noise_xx => 8,
            .wireguard => 8,
            .chacha20_poly1305 => 12,
            .aes_256_gcm => 12,
        };
    }

    pub fn tagSize(self: EncryptionType) usize {
        return switch (self) {
            .none => 0,
            .tls_1_2, .tls_1_3 => 16,
            .noise_xx => 16,
            .wireguard => 16,
            .chacha20_poly1305 => 16,
            .aes_256_gcm => 16,
        };
    }
};

pub const TransportType = enum {
    auto,
    thunderbolt,
    internet_tcp,
    internet_quic,
    rdma_roce,
    rdma_infiniband,
    loopback,

    pub fn maxBandwidth(self: TransportType) u64 {
        return switch (self) {
            .thunderbolt => 5_000_000_000,
            .rdma_infiniband => 25_000_000_000,
            .rdma_roce => 12_500_000_000,
            .internet_quic, .internet_tcp => 1_250_000_000,
            .auto, .loopback => std.math.maxInt(u64),
        };
    }

    pub fn typicalLatency(self: TransportType) u64 {
        return switch (self) {
            .thunderbolt => 1,
            .rdma_infiniband => 1,
            .rdma_roce => 2,
            .loopback => 0,
            .internet_tcp => 1000,
            .internet_quic => 500,
            .auto => 100,
        };
    }
};

pub const LinkConfig = struct {
    transport: TransportType = .auto,
    encryption: EncryptionConfig = .{},
    connect_timeout_ms: u64 = 10000,
    operation_timeout_ms: u64 = 5000,
    auto_reconnect: bool = true,
    max_reconnect_attempts: u32 = 5,
    reconnect_backoff_ms: u64 = 1000,
    bandwidth_aggregation: bool = false,
    compression_enabled: bool = true,
    compression_level: u4 = 6,
    buffer_size: usize = 1024 * 1024,
    keepalive_enabled: bool = true,
    keepalive_interval_ms: u64 = 30000,
    max_queue_size: usize = 1024,

    pub const EncryptionConfig = struct {
        encryption_type: EncryptionType = .tls_1_3,
        cert_path: ?[]const u8 = null,
        key_path: ?[]const u8 = null,
        ca_path: ?[]const u8 = null,
        require_mtls: bool = true,
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

    pub fn highPerformance() LinkConfig {
        return .{
            .transport = .thunderbolt,
            .encryption = .{ .encryption_type = .none },
            .compression_enabled = false,
            .buffer_size = 16 * 1024 * 1024,
            .keepalive_interval_ms = 1000,
        };
    }

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

pub const LinkStats = struct {
    bytes_sent: u64 = 0,
    bytes_received: u64 = 0,
    messages_sent: u64 = 0,
    messages_received: u64 = 0,
    send_errors: u64 = 0,
    receive_errors: u64 = 0,
    connected_at_ms: i64 = 0,
    last_activity_ms: i64 = 0,
    rtt_us: u64 = 0,
    bandwidth_bps: u64 = 0,
    packet_loss: f32 = 0.0,
    reconnections: u32 = 0,

    pub fn recordSend(_: *LinkStats, _: usize) void {}
    pub fn recordReceive(_: *LinkStats, _: usize) void {}
    pub fn recordError(_: *LinkStats, _: bool) void {}
};

pub const Link = struct {
    id: u64 = 0,
    remote_node_id: u64 = 0,
    remote_address: []const u8 = "",
    state: LinkState = .disconnected,
    transport_type: TransportType = .auto,
    channel: ?*SecureChannel = null,
    config: LinkConfig = .{},
    stats: LinkStats = .{},
    allocator: std.mem.Allocator = undefined,
    send_queue: std.ArrayListUnmanaged(QueuedMessage) = .{},
    mutex: @import("../../../services/shared/sync.zig").Mutex = .{},

    pub const QueuedMessage = struct {
        data: []const u8,
        priority: u8,
        timestamp: i64,
    };

    pub fn init(_: std.mem.Allocator, _: u64, _: []const u8, _: LinkConfig) !*Link {
        return error.NetworkDisabled;
    }
    pub fn deinit(_: *Link) void {}
    pub fn connect(_: *Link) LinkError!void {
        return error.ConnectionFailed;
    }
    pub fn disconnect(_: *Link) void {}
    pub fn send(_: *Link, _: []const u8) LinkError!void {
        return error.NotConnected;
    }
    pub fn receive(_: *Link, _: []u8) LinkError!usize {
        return error.NotConnected;
    }
    pub fn getStats(self: *Link) LinkStats {
        return self.stats;
    }
};

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

pub const LinkManager = struct {
    allocator: std.mem.Allocator = undefined,
    links: std.AutoHashMapUnmanaged(u64, *Link) = .{},
    links_by_address: std.StringHashMapUnmanaged(*Link) = .{},
    default_config: LinkConfig = .{},
    stats: ManagerStats = .{},
    next_link_id: u64 = 1,
    mutex: @import("../../../services/shared/sync.zig").Mutex = .{},

    pub const ManagerStats = struct {
        links_created: u64 = 0,
        links_closed: u64 = 0,
        total_bytes_sent: u64 = 0,
        total_bytes_received: u64 = 0,
        active_links: u64 = 0,
    };

    pub fn init(_: std.mem.Allocator, _: LinkConfig) !*LinkManager {
        return error.NetworkDisabled;
    }
    pub fn deinit(_: *LinkManager) void {}
    pub fn connect_link(_: *LinkManager, _: []const u8, _: ?LinkConfig) LinkError!*Link {
        return error.ConnectionFailed;
    }
    pub fn getLink(_: *LinkManager, _: u64) ?*Link {
        return null;
    }
    pub fn closeLink(_: *LinkManager, _: u64) void {}
    pub fn listLinks(_: *LinkManager, _: std.mem.Allocator) ![]u64 {
        return &.{};
    }
    pub fn getStats(self: *LinkManager) ManagerStats {
        return self.stats;
    }

    pub const Error = error{
        LinkingDisabled,
    };
};

pub const ThunderboltTransport = struct {
    pub fn init(_: std.mem.Allocator, _: ThunderboltConfig) !*@This() {
        return error.NetworkDisabled;
    }
    pub fn deinit(_: *@This()) void {}
};

pub const ThunderboltConfig = struct {
    dma_enabled: bool = true,
    max_dma_size: usize = 4 * 1024 * 1024,
    dma_timeout_ms: u64 = 1000,
    p2p_enabled: bool = true,
    buffer_pool_size: usize = 16 * 1024 * 1024,
    num_channels: u8 = 4,
    hotplug_enabled: bool = true,
    security_level: SecurityLevel = .user_authorized,
    power_delivery: bool = true,
    max_power_watts: u16 = 100,

    pub const SecurityLevel = enum {
        none,
        user_authorized,
        secure_boot,
        dp_only,
    };

    pub fn defaults() ThunderboltConfig {
        return .{};
    }

    pub fn highPerformance() ThunderboltConfig {
        return .{
            .dma_enabled = true,
            .max_dma_size = 16 * 1024 * 1024,
            .num_channels = 8,
            .security_level = .user_authorized,
        };
    }
};

pub const ThunderboltDevice = struct {
    id: u64 = 0,
    name: [64]u8 = std.mem.zeroes([64]u8),
    vendor_id: u16 = 0,
    device_id: u16 = 0,
    generation: Generation = .thunderbolt_4,
    state: ConnectionState = .disconnected,
    link_speed_gbps: u8 = 0,
    lanes: u8 = 0,
    power_watts: u16 = 0,
    security: SecurityStatus = .none,
    uuid: [16]u8 = std.mem.zeroes([16]u8),
    route_string: u64 = 0,
    protocols: Protocols = .{},

    pub const Generation = enum(u8) {
        thunderbolt_1 = 1,
        thunderbolt_2 = 2,
        thunderbolt_3 = 3,
        thunderbolt_4 = 4,
        usb4 = 5,

        pub fn maxBandwidth(self: Generation) u64 {
            return switch (self) {
                .thunderbolt_1 => 10_000_000_000 / 8,
                .thunderbolt_2 => 20_000_000_000 / 8,
                .thunderbolt_3, .thunderbolt_4 => 40_000_000_000 / 8,
                .usb4 => 80_000_000_000 / 8,
            };
        }
    };

    pub const ConnectionState = enum {
        disconnected,
        connecting,
        authenticating,
        connected,
        suspended,
        error_state,
    };

    pub const SecurityStatus = enum {
        none,
        user_authorized,
        device_authorized,
        secure_boot,
        rejected,
    };

    pub const Protocols = struct {
        pcie: bool = false,
        display_port: bool = false,
        usb3: bool = false,
        thunderbolt_networking: bool = false,
    };

    pub fn getName(self: *const ThunderboltDevice) []const u8 {
        const len = std.mem.indexOfScalar(u8, &self.name, 0) orelse self.name.len;
        return self.name[0..len];
    }

    pub fn supportsDma(self: *const ThunderboltDevice) bool {
        return self.protocols.pcie and
            (self.security == .user_authorized or
                self.security == .device_authorized or
                self.security == .secure_boot);
    }
};

pub const InternetTransport = struct {
    pub fn init(_: std.mem.Allocator, _: InternetConfig) !*@This() {
        return error.NetworkDisabled;
    }
    pub fn deinit(_: *@This()) void {}
};

pub const InternetConfig = struct {
    protocol: Protocol = .quic,
    bind_address: []const u8 = "0.0.0.0",
    bind_port: u16 = 0,
    stun_servers: []const []const u8 = &.{},
    turn_servers: []const TurnServer = &.{},
    ice_enabled: bool = true,
    ice_timeout_ms: u64 = 10000,
    zero_rtt_enabled: bool = true,
    connect_timeout_ms: u64 = 30000,
    idle_timeout_ms: u64 = 60000,
    max_streams: u32 = 100,
    max_datagram_size: usize = 1350,
    migration_enabled: bool = true,
    congestion_control: CongestionControl = .bbr,
    ecn_enabled: bool = true,
    keepalive_interval_ms: u64 = 15000,

    pub const Protocol = enum { quic, tcp_tls, websocket, auto };
    pub const CongestionControl = enum { bbr, cubic, reno };
    pub const TurnServer = struct {
        url: []const u8,
        username: []const u8,
        credential: []const u8,
    };

    pub fn defaults() InternetConfig {
        return .{};
    }

    pub fn lowLatency() InternetConfig {
        return .{
            .protocol = .quic,
            .zero_rtt_enabled = true,
            .congestion_control = .bbr,
            .ecn_enabled = true,
            .keepalive_interval_ms = 5000,
        };
    }

    pub fn reliable() InternetConfig {
        return .{
            .protocol = .tcp_tls,
            .ice_enabled = true,
            .connect_timeout_ms = 60000,
            .idle_timeout_ms = 120000,
        };
    }
};

pub const NatTraversal = struct {
    pub fn init(_: std.mem.Allocator) !*@This() {
        return error.NetworkDisabled;
    }
    pub fn deinit(_: *@This()) void {}
};

pub const QuicConnection = struct {
    pub fn init(_: std.mem.Allocator, _: []const u8, _: anytype) !*@This() {
        return error.NetworkDisabled;
    }
    pub fn deinit(_: *@This()) void {}
};

pub fn isEnabled() bool {
    return false;
}

test {
    std.testing.refAllDecls(@This());
}
