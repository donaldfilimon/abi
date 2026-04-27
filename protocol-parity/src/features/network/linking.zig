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
//! const linking = @import("linking.zig");
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

// Delegate to the real implementations in the linking/ subdirectory.
// This file previously contained void/placeholder stubs; it now re-exports
// the full types from linking/mod.zig (which itself imports from
// secure_channel.zig, thunderbolt.zig, and internet.zig).
const real = @import("linking/mod.zig");

// Sub-module namespaces
pub const secure_channel = real.secure_channel;
pub const thunderbolt = real.thunderbolt;
pub const internet = real.internet;

// Re-exports — secure channel types
pub const SecureChannel = real.SecureChannel;
pub const ChannelConfig = real.ChannelConfig;
pub const ChannelState = real.ChannelState;
pub const ChannelStats = real.ChannelStats;
pub const EncryptionType = real.EncryptionType;

// Re-exports — thunderbolt transport types
pub const ThunderboltTransport = real.ThunderboltTransport;
pub const ThunderboltConfig = real.ThunderboltConfig;
pub const ThunderboltDevice = real.ThunderboltDevice;

// Re-exports — internet transport types
pub const InternetTransport = real.InternetTransport;
pub const InternetConfig = real.InternetConfig;
pub const NatTraversal = real.NatTraversal;
pub const QuicConnection = real.QuicConnection;

// Re-exports — link management types
pub const TransportType = real.TransportType;
pub const LinkConfig = real.LinkConfig;
pub const LinkState = real.LinkState;
pub const LinkStats = real.LinkStats;
pub const Link = real.Link;
pub const LinkError = real.LinkError;
pub const LinkManager = real.LinkManager;

// Re-export isEnabled
pub const isEnabled = real.isEnabled;

// ============================================================================
// Tests
// ============================================================================

test "encryption type key size" {
    try std.testing.expectEqual(@as(usize, 0), EncryptionType.none.keySize());
    try std.testing.expectEqual(@as(usize, 32), EncryptionType.tls_1_3.keySize());
    try std.testing.expectEqual(@as(usize, 32), EncryptionType.chacha20_poly1305.keySize());
    try std.testing.expectEqual(@as(usize, 32), EncryptionType.aes_256_gcm.keySize());
}

test "internet config default values" {
    const config = InternetConfig{};
    try std.testing.expectEqual(InternetConfig.Protocol.quic, config.protocol);
    try std.testing.expectEqualStrings("0.0.0.0", config.bind_address);
}

test "thunderbolt config default values" {
    const config = ThunderboltConfig{};
    try std.testing.expect(config.dma_enabled);
    try std.testing.expectEqual(@as(usize, 4 * 1024 * 1024), config.max_dma_size);
}

test "link config default transport" {
    const config = LinkConfig{};
    try std.testing.expectEqual(TransportType.auto, config.transport);
}

test {
    std.testing.refAllDecls(@This());
}
