//! Unified Configuration System
//!
//! Single source of truth for all ABI framework configuration.
//! Supports both struct literal and builder pattern APIs.
//!
//! ## Usage
//!
//! ```zig
//! // Minimal - everything auto-detected with defaults
//! var fw = try abi.init(allocator);
//!
//! // Struct literal style
//! var fw = try abi.init(allocator, .{
//!     .gpu = .{ .backend = .vulkan },
//!     .ai = .{ .llm = .{} },
//! });
//!
//! // Builder style
//! var fw = try abi.Framework.builder(allocator)
//!     .withGpu(.{ .backend = .cuda })
//!     .withAi(.{ .llm = .{ .model_path = "./models/llama.gguf" } })
//!     .withDatabase(.{ .path = "./data" })
//!     .build();
//! ```

const std = @import("std");
const build_options = @import("build_options");

// Re-export ConfigLoader from shared utilities for file-based configuration
const shared_config = @import("shared/utils/config.zig");
pub const ConfigLoader = shared_config.ConfigLoader;

/// Unified configuration for the ABI framework.
/// Each field being non-null enables that feature with the specified settings.
/// A null field means the feature is disabled.
pub const Config = struct {
    /// GPU acceleration settings. Set to enable GPU features.
    gpu: ?GpuConfig = null,

    /// AI settings with independent sub-features.
    ai: ?AiConfig = null,

    /// Vector database settings.
    database: ?DatabaseConfig = null,

    /// Distributed network settings.
    network: ?NetworkConfig = null,

    /// Observability and monitoring settings.
    observability: ?ObservabilityConfig = null,

    /// Web/HTTP utilities settings.
    web: ?WebConfig = null,

    /// Plugin configuration.
    plugins: PluginConfig = .{},

    /// Returns a config with all compile-time enabled features activated with defaults.
    pub fn defaults() Config {
        return .{
            .gpu = if (build_options.enable_gpu) GpuConfig.defaults() else null,
            .ai = if (build_options.enable_ai) AiConfig.defaults() else null,
            .database = if (build_options.enable_database) DatabaseConfig.defaults() else null,
            .network = if (build_options.enable_network) NetworkConfig.defaults() else null,
            .observability = if (build_options.enable_profiling) ObservabilityConfig.defaults() else null,
            .web = if (build_options.enable_web) WebConfig.defaults() else null,
            .plugins = .{},
        };
    }

    /// Returns a minimal config with no features enabled.
    pub fn minimal() Config {
        return .{};
    }

    /// Check if a feature is enabled in this config.
    pub fn isEnabled(self: Config, feature: Feature) bool {
        return switch (feature) {
            .gpu => self.gpu != null,
            .ai => self.ai != null,
            .llm => if (self.ai) |ai| ai.llm != null else false,
            .embeddings => if (self.ai) |ai| ai.embeddings != null else false,
            .agents => if (self.ai) |ai| ai.agents != null else false,
            .training => if (self.ai) |ai| ai.training != null else false,
            .database => self.database != null,
            .network => self.network != null,
            .observability => self.observability != null,
            .web => self.web != null,
        };
    }

    /// Get list of enabled features.
    pub fn enabledFeatures(self: Config, allocator: std.mem.Allocator) ![]Feature {
        var list = std.ArrayListUnmanaged(Feature){};
        errdefer list.deinit(allocator);

        inline for (std.meta.fields(Feature)) |field| {
            const feature: Feature = @enumFromInt(field.value);
            if (self.isEnabled(feature)) {
                try list.append(allocator, feature);
            }
        }

        return list.toOwnedSlice(allocator);
    }
};

/// Feature identifiers for the framework.
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

    pub fn name(self: Feature) []const u8 {
        return @tagName(self);
    }

    pub fn description(self: Feature) []const u8 {
        return switch (self) {
            .gpu => "GPU acceleration and compute",
            .ai => "AI core functionality",
            .llm => "Local LLM inference",
            .embeddings => "Vector embeddings generation",
            .agents => "AI agent runtime",
            .training => "Model training pipelines",
            .database => "Vector database (WDBX)",
            .network => "Distributed compute network",
            .observability => "Metrics, tracing, profiling",
            .web => "Web/HTTP utilities",
        };
    }

    /// Check if this feature is available at compile time.
    pub fn isCompileTimeEnabled(self: Feature) bool {
        return switch (self) {
            .gpu => build_options.enable_gpu,
            .ai, .llm, .embeddings, .agents, .training => build_options.enable_ai,
            .database => build_options.enable_database,
            .network => build_options.enable_network,
            .observability => build_options.enable_profiling,
            .web => build_options.enable_web,
        };
    }
};

// ============================================================================
// GPU Configuration
// ============================================================================

pub const GpuConfig = struct {
    /// GPU backend to use. Auto-detect by default.
    backend: Backend = .auto,

    /// Preferred device index (0 = first available).
    device_index: u32 = 0,

    /// Maximum GPU memory to use (null = no limit).
    memory_limit: ?usize = null,

    /// Enable async operations.
    async_enabled: bool = true,

    /// Enable kernel caching.
    cache_kernels: bool = true,

    /// Recovery settings for GPU failures.
    recovery: RecoveryConfig = .{},

    pub const Backend = enum {
        auto,
        vulkan,
        cuda,
        metal,
        webgpu,
        opengl,
        cpu,
    };

    pub const RecoveryConfig = struct {
        enabled: bool = true,
        max_retries: u32 = 3,
        fallback_to_cpu: bool = true,
    };

    pub fn defaults() GpuConfig {
        return .{};
    }

    /// Select the best backend based on availability.
    pub fn autoSelectBackend() Backend {
        if (build_options.gpu_cuda) return .cuda;
        if (build_options.gpu_vulkan) return .vulkan;
        if (build_options.gpu_metal) return .metal;
        if (build_options.gpu_webgpu) return .webgpu;
        if (build_options.gpu_opengl) return .opengl;
        return .cpu;
    }
};

// ============================================================================
// AI Configuration
// ============================================================================

pub const AiConfig = struct {
    /// LLM inference settings. Set to enable local LLM.
    llm: ?LlmConfig = null,

    /// Embeddings generation settings.
    embeddings: ?EmbeddingsConfig = null,

    /// Agent runtime settings.
    agents: ?AgentsConfig = null,

    /// Training pipeline settings.
    training: ?TrainingConfig = null,

    pub fn defaults() AiConfig {
        return .{
            .llm = if (build_options.enable_llm) LlmConfig.defaults() else null,
            .embeddings = EmbeddingsConfig.defaults(),
            .agents = AgentsConfig.defaults(),
            .training = null, // Training not enabled by default
        };
    }

    /// Enable only LLM inference.
    pub fn llmOnly(config: LlmConfig) AiConfig {
        return .{ .llm = config };
    }

    /// Enable only embeddings.
    pub fn embeddingsOnly(config: EmbeddingsConfig) AiConfig {
        return .{ .embeddings = config };
    }
};

pub const LlmConfig = struct {
    /// Path to model file (GGUF format).
    model_path: ?[]const u8 = null,

    /// Model to use from registry.
    model_name: []const u8 = "gpt2",

    /// Context window size.
    context_size: u32 = 2048,

    /// Number of threads for inference.
    threads: ?u32 = null,

    /// Use GPU acceleration if available.
    use_gpu: bool = true,

    /// Batch size for inference.
    batch_size: u32 = 512,

    pub fn defaults() LlmConfig {
        return .{};
    }
};

pub const EmbeddingsConfig = struct {
    /// Embedding model to use.
    model: []const u8 = "default",

    /// Output embedding dimension.
    dimension: u32 = 384,

    /// Normalize output vectors.
    normalize: bool = true,

    pub fn defaults() EmbeddingsConfig {
        return .{};
    }
};

pub const AgentsConfig = struct {
    /// Maximum concurrent agents.
    max_agents: u32 = 16,

    /// Default agent timeout in milliseconds.
    timeout_ms: u64 = 30000,

    /// Enable agent memory/context persistence.
    persistent_memory: bool = false,

    pub fn defaults() AgentsConfig {
        return .{};
    }
};

pub const TrainingConfig = struct {
    /// Number of training epochs.
    epochs: u32 = 10,

    /// Training batch size.
    batch_size: u32 = 32,

    /// Learning rate.
    learning_rate: f32 = 0.001,

    /// Optimizer to use.
    optimizer: Optimizer = .adamw,

    /// Checkpoint directory.
    checkpoint_dir: ?[]const u8 = null,

    /// Checkpoint frequency (epochs).
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

// ============================================================================
// Database Configuration
// ============================================================================

pub const DatabaseConfig = struct {
    /// Database file path.
    path: []const u8 = "./abi.db",

    /// Index type for vector search.
    index_type: IndexType = .hnsw,

    /// Enable write-ahead logging.
    wal_enabled: bool = true,

    /// Cache size in bytes.
    cache_size: usize = 64 * 1024 * 1024, // 64MB

    /// Auto-optimize on startup.
    auto_optimize: bool = false,

    pub const IndexType = enum {
        hnsw,
        ivf_pq,
        flat,
    };

    pub fn defaults() DatabaseConfig {
        return .{};
    }

    /// In-memory database configuration.
    pub fn inMemory() DatabaseConfig {
        return .{
            .path = ":memory:",
            .wal_enabled = false,
        };
    }
};

// ============================================================================
// Network Configuration
// ============================================================================

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
};

// ============================================================================
// Observability Configuration
// ============================================================================

pub const ObservabilityConfig = struct {
    /// Enable metrics collection.
    metrics_enabled: bool = true,

    /// Enable distributed tracing.
    tracing_enabled: bool = true,

    /// Enable performance profiling.
    profiling_enabled: bool = false,

    /// Metrics export endpoint.
    metrics_endpoint: ?[]const u8 = null,

    /// Trace sampling rate (0.0 - 1.0).
    trace_sample_rate: f32 = 0.1,

    pub fn defaults() ObservabilityConfig {
        return .{};
    }

    /// Full observability (all features enabled).
    pub fn full() ObservabilityConfig {
        return .{
            .metrics_enabled = true,
            .tracing_enabled = true,
            .profiling_enabled = true,
            .trace_sample_rate = 1.0,
        };
    }
};

// ============================================================================
// Web Configuration
// ============================================================================

pub const WebConfig = struct {
    /// HTTP server bind address.
    bind_address: []const u8 = "127.0.0.1",

    /// HTTP server port.
    port: u16 = 3000,

    /// Enable CORS.
    cors_enabled: bool = true,

    /// Request timeout in milliseconds.
    timeout_ms: u64 = 30000,

    /// Maximum request body size.
    max_body_size: usize = 10 * 1024 * 1024, // 10MB

    pub fn defaults() WebConfig {
        return .{};
    }
};

// ============================================================================
// Plugin Configuration
// ============================================================================

pub const PluginConfig = struct {
    /// Paths to search for plugins.
    paths: []const []const u8 = &.{},

    /// Auto-discover plugins in paths.
    auto_discover: bool = false,

    /// Plugins to load by name.
    load: []const []const u8 = &.{},
};

// ============================================================================
// Builder API
// ============================================================================

/// Fluent builder for constructing Config instances.
pub const Builder = struct {
    allocator: std.mem.Allocator,
    config: Config,

    pub fn init(allocator: std.mem.Allocator) Builder {
        return .{
            .allocator = allocator,
            .config = .{},
        };
    }

    /// Start with default configuration for all compile-time enabled features.
    pub fn withDefaults(self: *Builder) *Builder {
        self.config = Config.defaults();
        return self;
    }

    /// Enable GPU with specified configuration.
    pub fn withGpu(self: *Builder, gpu_config: GpuConfig) *Builder {
        self.config.gpu = gpu_config;
        return self;
    }

    /// Enable GPU with default configuration.
    pub fn withGpuDefaults(self: *Builder) *Builder {
        self.config.gpu = GpuConfig.defaults();
        return self;
    }

    /// Enable AI with specified configuration.
    pub fn withAi(self: *Builder, ai_config: AiConfig) *Builder {
        self.config.ai = ai_config;
        return self;
    }

    /// Enable AI with default configuration.
    pub fn withAiDefaults(self: *Builder) *Builder {
        self.config.ai = AiConfig.defaults();
        return self;
    }

    /// Enable LLM only (convenience method).
    pub fn withLlm(self: *Builder, llm_config: LlmConfig) *Builder {
        if (self.config.ai == null) {
            self.config.ai = .{};
        }
        self.config.ai.?.llm = llm_config;
        return self;
    }

    /// Enable database with specified configuration.
    pub fn withDatabase(self: *Builder, db_config: DatabaseConfig) *Builder {
        self.config.database = db_config;
        return self;
    }

    /// Enable database with default configuration.
    pub fn withDatabaseDefaults(self: *Builder) *Builder {
        self.config.database = DatabaseConfig.defaults();
        return self;
    }

    /// Enable network with specified configuration.
    pub fn withNetwork(self: *Builder, net_config: NetworkConfig) *Builder {
        self.config.network = net_config;
        return self;
    }

    /// Enable network with default configuration.
    pub fn withNetworkDefaults(self: *Builder) *Builder {
        self.config.network = NetworkConfig.defaults();
        return self;
    }

    /// Enable observability with specified configuration.
    pub fn withObservability(self: *Builder, obs_config: ObservabilityConfig) *Builder {
        self.config.observability = obs_config;
        return self;
    }

    /// Enable observability with default configuration.
    pub fn withObservabilityDefaults(self: *Builder) *Builder {
        self.config.observability = ObservabilityConfig.defaults();
        return self;
    }

    /// Enable web with specified configuration.
    pub fn withWeb(self: *Builder, web_config: WebConfig) *Builder {
        self.config.web = web_config;
        return self;
    }

    /// Enable web with default configuration.
    pub fn withWebDefaults(self: *Builder) *Builder {
        self.config.web = WebConfig.defaults();
        return self;
    }

    /// Configure plugins.
    pub fn withPlugins(self: *Builder, plugin_config: PluginConfig) *Builder {
        self.config.plugins = plugin_config;
        return self;
    }

    /// Finalize and return the configuration.
    pub fn build(self: *Builder) Config {
        return self.config;
    }
};

// ============================================================================
// Errors
// ============================================================================

pub const ConfigError = error{
    /// Feature is disabled at compile time.
    FeatureDisabled,
    /// Invalid configuration value.
    InvalidConfig,
    /// Required configuration missing.
    MissingRequired,
    /// Configuration conflict detected.
    ConflictingConfig,
};

/// Validate configuration against compile-time constraints.
pub fn validate(config: Config) ConfigError!void {
    // Check GPU config against compile-time flags
    if (config.gpu != null and !build_options.enable_gpu) {
        return ConfigError.FeatureDisabled;
    }

    // Check AI config
    if (config.ai != null and !build_options.enable_ai) {
        return ConfigError.FeatureDisabled;
    }

    // Check LLM specifically
    if (config.ai) |ai| {
        if (ai.llm != null and !build_options.enable_llm) {
            return ConfigError.FeatureDisabled;
        }
    }

    // Check database config
    if (config.database != null and !build_options.enable_database) {
        return ConfigError.FeatureDisabled;
    }

    // Check network config
    if (config.network != null and !build_options.enable_network) {
        return ConfigError.FeatureDisabled;
    }

    // Check web config
    if (config.web != null and !build_options.enable_web) {
        return ConfigError.FeatureDisabled;
    }

    // Check observability config
    if (config.observability != null and !build_options.enable_profiling) {
        return ConfigError.FeatureDisabled;
    }
}

// ============================================================================
// Tests
// ============================================================================

test "Config.defaults creates valid configuration" {
    const config = Config.defaults();
    try validate(config);
}

test "Config.minimal creates empty configuration" {
    const config = Config.minimal();
    try std.testing.expect(config.gpu == null);
    try std.testing.expect(config.ai == null);
    try std.testing.expect(config.database == null);
}

test "Builder creates valid configuration" {
    var builder = Builder.init(std.testing.allocator);
    const config = builder
        .withGpuDefaults()
        .withAiDefaults()
        .build();

    if (build_options.enable_gpu) {
        try std.testing.expect(config.gpu != null);
    }
    if (build_options.enable_ai) {
        try std.testing.expect(config.ai != null);
    }
}

test "Config.isEnabled correctly reports feature state" {
    const config = Config{
        .gpu = .{},
        .ai = .{ .llm = .{} },
    };

    try std.testing.expect(config.isEnabled(.gpu));
    try std.testing.expect(config.isEnabled(.ai));
    try std.testing.expect(config.isEnabled(.llm));
    try std.testing.expect(!config.isEnabled(.database));
    try std.testing.expect(!config.isEnabled(.embeddings));
}
