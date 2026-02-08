const std = @import("std");

pub const LinkManager = struct {
    pub fn init(_: std.mem.Allocator, _: LinkConfig) !@This() {
        return error.NetworkDisabled;
    }
    pub fn deinit(_: *@This()) void {}
};

pub const Link = struct {
    id: []const u8 = "",
    state: LinkState = .disconnected,
    transport: TransportType = .tcp,
};

pub const LinkConfig = struct {
    max_links: usize = 64,
    timeout_ms: u64 = 10_000,
};

pub const LinkState = enum { disconnected, connecting, connected, error_state };

pub const LinkStats = struct {
    bytes_sent: u64 = 0,
    bytes_received: u64 = 0,
    messages_sent: u64 = 0,
    messages_received: u64 = 0,
    errors: u64 = 0,
};

pub const TransportType = enum { tcp, udp, thunderbolt, internet, quic };

pub const SecureChannel = struct {
    pub fn init(_: std.mem.Allocator, _: ChannelConfig) !@This() {
        return error.NetworkDisabled;
    }
    pub fn deinit(_: *@This()) void {}
};

pub const ChannelConfig = struct {
    encryption: EncryptionType = .tls,
};

pub const EncryptionType = enum { none, tls, dtls, noise };

pub const ThunderboltTransport = struct {
    pub fn init(_: std.mem.Allocator, _: ThunderboltConfig) !@This() {
        return error.NetworkDisabled;
    }
    pub fn deinit(_: *@This()) void {}
};

pub const ThunderboltConfig = struct {
    device_id: u32 = 0,
    max_bandwidth_gbps: f64 = 40.0,
};

pub const ThunderboltDevice = struct {
    id: u32 = 0,
    name: []const u8 = "",
    available: bool = false,
};

pub const InternetTransport = struct {
    pub fn init(_: std.mem.Allocator, _: InternetConfig) !@This() {
        return error.NetworkDisabled;
    }
    pub fn deinit(_: *@This()) void {}
};

pub const InternetConfig = struct {
    bind_address: []const u8 = "0.0.0.0",
    port: u16 = 0,
    enable_nat_traversal: bool = false,
};

pub const NatTraversal = struct {
    pub fn init(_: std.mem.Allocator) !@This() {
        return error.NetworkDisabled;
    }
    pub fn deinit(_: *@This()) void {}
};

pub const QuicConnection = struct {
    pub fn init(_: std.mem.Allocator) !@This() {
        return error.NetworkDisabled;
    }
    pub fn deinit(_: *@This()) void {}
};
