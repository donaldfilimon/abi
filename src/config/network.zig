//! Network Configuration
//!
//! Configuration for distributed compute, unified memory,
//! and inter-node linking.

const std = @import("std");

/// Distributed network configuration.
pub const NetworkConfig = struct {
    /// Node bind address.
    bind_address: []const u8 = "0.0.0.0",

    /// Node bind port.
    port: u16 = 8080,

    /// Enable node discovery.
    discovery_enabled: bool = true,

    /// Known peer addresses for bootstrapping.
    bootstrap_peers: []const []const u8 = &.{},

    /// Enable Raft consensus.
    consensus_enabled: bool = false,

    /// Node role in the cluster.
    role: Role = .worker,

    /// Unified memory configuration.
    unified_memory: ?UnifiedMemoryConfig = null,

    /// Linking configuration.
    linking: ?LinkingConfig = null,

    pub const Role = enum {
        coordinator,
        worker,
        observer,
    };

    pub fn defaults() NetworkConfig {
        return .{};
    }

    /// Standalone node (no clustering).
    pub fn standalone() NetworkConfig {
        return .{
            .discovery_enabled = false,
            .consensus_enabled = false,
        };
    }

    /// Distributed compute with unified memory.
    pub fn distributed() NetworkConfig {
        return .{
            .discovery_enabled = true,
            .consensus_enabled = true,
            .unified_memory = UnifiedMemoryConfig.defaults(),
            .linking = LinkingConfig.defaults(),
        };
    }
};

/// Unified memory configuration for distributed compute.
pub const UnifiedMemoryConfig = struct {
    const max_shared_memory_default: usize = @intCast(@min(
        @as(u64, 16 * 1024 * 1024 * 1024),
        @as(u64, std.math.maxInt(usize)),
    ));

    /// Maximum number of registered memory regions.
    max_regions: usize = 256,

    /// Maximum total shared memory size (bytes).
    max_shared_memory: usize = max_shared_memory_default, // 16 GB (clamped)

    /// Enable memory coherence protocol.
    coherence_enabled: bool = true,

    /// Coherence protocol to use.
    coherence_protocol: CoherenceProtocol = .mesi,

    /// Enable encryption for memory transfers.
    encrypt_transfers: bool = true,

    /// Enable compression for memory transfers.
    compress_transfers: bool = true,

    /// Page size for memory regions.
    page_size: usize = 4096,

    /// Enable RDMA when available.
    rdma_enabled: bool = true,

    /// Timeout for remote memory operations (ms).
    operation_timeout_ms: u64 = 5000,

    /// Enable memory prefetching.
    prefetch_enabled: bool = true,

    pub const CoherenceProtocol = enum {
        mesi,
        moesi,
        directory,
        none,
    };

    pub fn defaults() UnifiedMemoryConfig {
        return .{};
    }

    /// High-performance for local Thunderbolt links.
    pub fn thunderbolt() UnifiedMemoryConfig {
        return .{
            .coherence_protocol = .moesi,
            .encrypt_transfers = false,
            .compress_transfers = false,
            .rdma_enabled = true,
            .operation_timeout_ms = 100,
        };
    }

    /// Secure for Internet links.
    pub fn internet() UnifiedMemoryConfig {
        return .{
            .coherence_protocol = .directory,
            .encrypt_transfers = true,
            .compress_transfers = true,
            .rdma_enabled = false,
            .operation_timeout_ms = 30000,
        };
    }
};

/// Linking configuration for inter-node communication.
pub const LinkingConfig = struct {
    /// Preferred transport type.
    transport: Transport = .auto,

    /// Enable automatic reconnection.
    auto_reconnect: bool = true,

    /// Maximum reconnection attempts.
    max_reconnect_attempts: u32 = 5,

    /// Enable bandwidth aggregation.
    bandwidth_aggregation: bool = false,

    /// Enable compression.
    compression_enabled: bool = true,

    /// Enable keepalive.
    keepalive_enabled: bool = true,

    /// Keepalive interval (milliseconds).
    keepalive_interval_ms: u64 = 30000,

    /// Encryption settings.
    encryption: EncryptionConfig = .{},

    /// Thunderbolt-specific settings.
    thunderbolt_settings: ThunderboltConfig = .{},

    /// Internet-specific settings.
    internet_settings: InternetLinkConfig = .{},

    pub const Transport = enum {
        auto,
        thunderbolt,
        internet_tcp,
        internet_quic,
        rdma_roce,
        rdma_infiniband,
    };

    pub const EncryptionConfig = struct {
        /// Encryption type.
        encryption_type: EncryptionType = .tls_1_3,
        /// Require mutual TLS.
        require_mtls: bool = true,
        /// Certificate path.
        cert_path: ?[]const u8 = null,
        /// Key path.
        key_path: ?[]const u8 = null,
        /// CA path.
        ca_path: ?[]const u8 = null,

        pub const EncryptionType = enum {
            none,
            tls_1_2,
            tls_1_3,
            noise_xx,
            wireguard,
        };
    };

    pub const ThunderboltConfig = struct {
        /// Enable DMA.
        dma_enabled: bool = true,
        /// Maximum DMA transfer size.
        max_dma_size: usize = 4 * 1024 * 1024,
        /// Enable peer-to-peer.
        p2p_enabled: bool = true,
        /// Security level.
        security_level: SecurityLevel = .user_authorized,

        pub const SecurityLevel = enum {
            none,
            user_authorized,
            secure_boot,
        };
    };

    pub const InternetLinkConfig = struct {
        /// Enable ICE for NAT traversal.
        ice_enabled: bool = true,
        /// STUN servers.
        stun_servers: []const []const u8 = &default_stun_servers,
        /// Enable QUIC 0-RTT.
        zero_rtt_enabled: bool = true,
        /// Congestion control algorithm.
        congestion_control: CongestionControl = .bbr,

        pub const CongestionControl = enum {
            bbr,
            cubic,
            reno,
        };

        const default_stun_servers = [_][]const u8{
            "stun:stun.l.google.com:19302",
            "stun:stun.cloudflare.com:3478",
        };
    };

    pub fn defaults() LinkingConfig {
        return .{};
    }

    /// High-performance local linking.
    pub fn highPerformance() LinkingConfig {
        return .{
            .transport = .thunderbolt,
            .compression_enabled = false,
            .encryption = .{ .encryption_type = .none },
        };
    }

    /// Secure Internet linking.
    pub fn secure() LinkingConfig {
        return .{
            .transport = .internet_quic,
            .compression_enabled = true,
            .encryption = .{
                .encryption_type = .tls_1_3,
                .require_mtls = true,
            },
        };
    }
};
