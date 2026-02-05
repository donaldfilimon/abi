const std = @import("std");

pub const NetworkConfig = struct {
    bind_address: []const u8 = "0.0.0.0",
    port: u16 = 8080,
    discovery_enabled: bool = false,
    bootstrap_peers: []const []const u8 = &.{},
    consensus_enabled: bool = false,
    role: Role = .worker,
    unified_memory: ?UnifiedMemoryConfig = null,
    linking: ?LinkingConfig = null,

    pub const Role = enum {
        coordinator,
        worker,
        observer,
    };

    pub fn defaults() NetworkConfig {
        return .{};
    }

    pub fn standalone() NetworkConfig {
        return .{};
    }

    pub fn distributed() NetworkConfig {
        return .{};
    }
};

pub const UnifiedMemoryConfig = struct {
    max_regions: usize = 0,
    max_shared_memory: usize = 0,
    coherence_enabled: bool = false,
    coherence_protocol: CoherenceProtocol = .none,
    encrypt_transfers: bool = false,
    compress_transfers: bool = false,
    page_size: usize = 4096,
    rdma_enabled: bool = false,
    operation_timeout_ms: u64 = 5000,
    prefetch_enabled: bool = false,

    pub const CoherenceProtocol = enum {
        mesi,
        moesi,
        directory,
        none,
    };

    pub fn defaults() UnifiedMemoryConfig {
        return .{};
    }

    pub fn thunderbolt() UnifiedMemoryConfig {
        return .{};
    }

    pub fn internet() UnifiedMemoryConfig {
        return .{};
    }
};

pub const LinkingConfig = struct {
    transport: Transport = .auto,
    auto_reconnect: bool = false,
    max_reconnect_attempts: u32 = 0,
    bandwidth_aggregation: bool = false,
    compression_enabled: bool = false,
    keepalive_enabled: bool = false,
    keepalive_interval_ms: u64 = 30000,
    encryption: EncryptionConfig = .{},
    thunderbolt_settings: ThunderboltConfig = .{},
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
        encryption_type: EncryptionType = .none,
        require_mtls: bool = false,
        cert_path: ?[]const u8 = null,
        key_path: ?[]const u8 = null,
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
        dma_enabled: bool = false,
        max_dma_size: usize = 0,
        p2p_enabled: bool = false,
        security_level: SecurityLevel = .none,

        pub const SecurityLevel = enum {
            none,
            user_authorized,
            secure_boot,
        };
    };

    pub const InternetLinkConfig = struct {
        ice_enabled: bool = false,
        stun_servers: []const []const u8 = &.{},
        zero_rtt_enabled: bool = false,
        congestion_control: CongestionControl = .cubic,

        pub const CongestionControl = enum {
            bbr,
            cubic,
            reno,
        };
    };

    pub fn defaults() LinkingConfig {
        return .{};
    }

    pub fn highPerformance() LinkingConfig {
        return .{};
    }

    pub fn secure() LinkingConfig {
        return .{};
    }
};
