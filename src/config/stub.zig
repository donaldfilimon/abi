//! Configuration Module Stub
//!
//! Placeholder implementations when configuration features are disabled.
//! Provides identical API surface to mod.zig with minimal functionality.

const std = @import("std");

// ============================================================================
// GPU Configuration Stub
// ============================================================================

pub const GpuConfig = struct {
    backend: Backend = .auto,
    device_index: u32 = 0,
    memory_limit: ?usize = null,
    async_enabled: bool = false,
    cache_kernels: bool = false,
    recovery: RecoveryConfig = .{},

    pub const Backend = enum {
        auto,
        vulkan,
        cuda,
        metal,
        webgpu,
        opengl,
        fpga,
        cpu,
    };

    pub fn defaults() GpuConfig {
        return .{};
    }

    pub fn autoSelectBackend() Backend {
        return .cpu;
    }
};

pub const RecoveryConfig = struct {
    enabled: bool = false,
    max_retries: u32 = 0,
    fallback_to_cpu: bool = true,
};

// ============================================================================
// AI Configuration Stub
// ============================================================================

pub const AiConfig = struct {
    llm: ?LlmConfig = null,
    embeddings: ?EmbeddingsConfig = null,
    agents: ?AgentsConfig = null,
    training: ?TrainingConfig = null,
    personas: ?PersonasConfig = null,
    auto_discover: bool = false,
    model_paths: []const []const u8 = &.{},
    adaptive_config: bool = false,
    warmup_diagnostics: bool = false,

    pub fn defaults() AiConfig {
        return .{};
    }

    pub fn withAutoDiscovery() AiConfig {
        return .{};
    }

    pub fn llmOnly(config: LlmConfig) AiConfig {
        _ = config;
        return .{};
    }

    pub fn embeddingsOnly(config: EmbeddingsConfig) AiConfig {
        _ = config;
        return .{};
    }
};

pub const LlmConfig = struct {
    model_path: ?[]const u8 = null,
    model_name: []const u8 = "gpt2",
    context_size: u32 = 2048,
    threads: ?u32 = null,
    use_gpu: bool = false,
    batch_size: u32 = 512,

    pub fn defaults() LlmConfig {
        return .{};
    }
};

pub const EmbeddingsConfig = struct {
    model: []const u8 = "default",
    dimension: u32 = 384,
    normalize: bool = true,

    pub fn defaults() EmbeddingsConfig {
        return .{};
    }
};

pub const AgentsConfig = struct {
    max_agents: u32 = 0,
    timeout_ms: u64 = 30000,
    persistent_memory: bool = false,

    pub fn defaults() AgentsConfig {
        return .{};
    }
};

pub const TrainingConfig = struct {
    epochs: u32 = 0,
    batch_size: u32 = 0,
    learning_rate: f32 = 0.0,
    optimizer: Optimizer = .adamw,
    checkpoint_dir: ?[]const u8 = null,
    checkpoint_frequency: u32 = 1,

    pub const Optimizer = enum {
        sgd,
        adam,
        adamw,
        rmsprop,
    };

    pub fn defaults() TrainingConfig {
        return .{};
    }
};

pub const PersonasConfig = struct {
    pub fn defaults() PersonasConfig {
        return .{};
    }
};

// ============================================================================
// Database Configuration Stub
// ============================================================================

pub const DatabaseConfig = struct {
    path: []const u8 = "./abi.db",
    index_type: IndexType = .flat,
    wal_enabled: bool = false,
    cache_size: usize = 0,
    auto_optimize: bool = false,

    pub const IndexType = enum {
        hnsw,
        ivf_pq,
        flat,
    };

    pub fn defaults() DatabaseConfig {
        return .{};
    }

    pub fn inMemory() DatabaseConfig {
        return .{ .path = ":memory:" };
    }
};

// ============================================================================
// Network Configuration Stub
// ============================================================================

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

// ============================================================================
// Observability Configuration Stub
// ============================================================================

pub const ObservabilityConfig = struct {
    metrics_enabled: bool = false,
    tracing_enabled: bool = false,
    profiling_enabled: bool = false,
    metrics_endpoint: ?[]const u8 = null,
    trace_sample_rate: f32 = 0.0,

    pub fn defaults() ObservabilityConfig {
        return .{};
    }

    pub fn full() ObservabilityConfig {
        return .{};
    }
};

// ============================================================================
// Web Configuration Stub
// ============================================================================

pub const WebConfig = struct {
    bind_address: []const u8 = "127.0.0.1",
    port: u16 = 3000,
    cors_enabled: bool = false,
    timeout_ms: u64 = 30000,
    max_body_size: usize = 0,

    pub fn defaults() WebConfig {
        return .{};
    }
};

// ============================================================================
// Plugin Configuration Stub
// ============================================================================

pub const PluginConfig = struct {
    paths: []const []const u8 = &.{},
    auto_discover: bool = false,
    load: []const []const u8 = &.{},
    allow_untrusted: bool = false,

    pub fn defaults() PluginConfig {
        return .{};
    }

    pub fn withPaths(paths: []const []const u8) PluginConfig {
        _ = paths;
        return .{};
    }
};

// ============================================================================
// Config Loader Stub
// ============================================================================

pub const LoadError = error{
    ConfigDisabled,
    InvalidValue,
    MissingRequired,
    ParseError,
    OutOfMemory,
};

pub const ConfigLoader = struct {
    allocator: std.mem.Allocator,

    const Self = @This();

    pub fn init(allocator: std.mem.Allocator) Self {
        return .{ .allocator = allocator };
    }

    pub fn deinit(self: *Self) void {
        _ = self;
    }

    pub fn load(self: *Self) LoadError!Config {
        _ = self;
        return error.ConfigDisabled;
    }

    pub fn loadWithBase(self: *Self, base: Config) LoadError!Config {
        _ = self;
        _ = base;
        return error.ConfigDisabled;
    }
};

// ============================================================================
// Feature Enum
// ============================================================================

pub const Feature = enum {
    gpu,
    ai,
    llm,
    embeddings,
    agents,
    training,
    database,
    network,
    observability,
    web,
    personas,
    cloud,

    pub fn name(self: Feature) []const u8 {
        return @tagName(self);
    }

    pub fn description(self: Feature) []const u8 {
        _ = self;
        return "Feature disabled";
    }

    pub fn isCompileTimeEnabled(self: Feature) bool {
        _ = self;
        return false;
    }
};

// ============================================================================
// Unified Config Stub
// ============================================================================

pub const Config = struct {
    gpu: ?GpuConfig = null,
    ai: ?AiConfig = null,
    database: ?DatabaseConfig = null,
    network: ?NetworkConfig = null,
    observability: ?ObservabilityConfig = null,
    web: ?WebConfig = null,
    plugins: PluginConfig = .{},

    pub fn defaults() Config {
        return .{};
    }

    pub fn minimal() Config {
        return .{};
    }

    pub fn isEnabled(self: Config, feature: Feature) bool {
        _ = self;
        _ = feature;
        return false;
    }

    pub fn enabledFeatures(self: Config, allocator: std.mem.Allocator) ![]Feature {
        _ = self;
        return allocator.alloc(Feature, 0);
    }
};

// ============================================================================
// Builder Pattern Stub
// ============================================================================

pub const Builder = struct {
    allocator: std.mem.Allocator,
    config: Config,

    pub fn init(allocator: std.mem.Allocator) Builder {
        return .{
            .allocator = allocator,
            .config = Config.minimal(),
        };
    }

    pub fn withDefaults(self: *Builder) *Builder {
        return self;
    }

    pub fn withGpu(self: *Builder, cfg: GpuConfig) *Builder {
        _ = cfg;
        return self;
    }

    pub fn withGpuDefaults(self: *Builder) *Builder {
        return self;
    }

    pub fn withAi(self: *Builder, cfg: AiConfig) *Builder {
        _ = cfg;
        return self;
    }

    pub fn withAiDefaults(self: *Builder) *Builder {
        return self;
    }

    pub fn withLlm(self: *Builder, cfg: LlmConfig) *Builder {
        _ = cfg;
        return self;
    }

    pub fn withDatabase(self: *Builder, cfg: DatabaseConfig) *Builder {
        _ = cfg;
        return self;
    }

    pub fn withDatabaseDefaults(self: *Builder) *Builder {
        return self;
    }

    pub fn withNetwork(self: *Builder, cfg: NetworkConfig) *Builder {
        _ = cfg;
        return self;
    }

    pub fn withNetworkDefaults(self: *Builder) *Builder {
        return self;
    }

    pub fn withObservability(self: *Builder, cfg: ObservabilityConfig) *Builder {
        _ = cfg;
        return self;
    }

    pub fn withObservabilityDefaults(self: *Builder) *Builder {
        return self;
    }

    pub fn withWeb(self: *Builder, cfg: WebConfig) *Builder {
        _ = cfg;
        return self;
    }

    pub fn withWebDefaults(self: *Builder) *Builder {
        return self;
    }

    pub fn withPlugins(self: *Builder, cfg: PluginConfig) *Builder {
        _ = cfg;
        return self;
    }

    pub fn build(self: *Builder) Config {
        return self.config;
    }
};

// ============================================================================
// Validation Stub
// ============================================================================

pub const ConfigError = error{
    ConfigDisabled,
    FeatureDisabled,
    InvalidConfig,
    MissingRequired,
    ConflictingConfig,
};

pub fn validate(config: Config) ConfigError!void {
    _ = config;
    return error.ConfigDisabled;
}
