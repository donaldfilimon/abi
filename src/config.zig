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
const shared_config = @import("shared/utils_combined.zig").config;
pub const ConfigLoader = shared_config.ConfigLoader;

// =============================================================================
// Domain Configuration Re-exports
// =============================================================================
// Domain-specific configs are defined in separate files for maintainability.
// This module re-exports them to maintain backward compatibility.

const gpu_mod = @import("config/gpu.zig");
const ai_mod = @import("config/ai.zig");
const database_mod = @import("config/database.zig");
const network_mod = @import("config/network.zig");
const observability_mod = @import("config/observability.zig");
const web_mod = @import("config/web.zig");
const plugins_mod = @import("config/plugins.zig");

// GPU configuration types
pub const GpuConfig = gpu_mod.GpuConfig;

// AI configuration types
pub const AiConfig = ai_mod.AiConfig;
pub const LlmConfig = ai_mod.LlmConfig;
pub const EmbeddingsConfig = ai_mod.EmbeddingsConfig;
pub const AgentsConfig = ai_mod.AgentsConfig;
pub const TrainingConfig = ai_mod.TrainingConfig;
pub const PersonasConfig = ai_mod.PersonasConfig;

// Database configuration types
pub const DatabaseConfig = database_mod.DatabaseConfig;

// Network configuration types
pub const NetworkConfig = network_mod.NetworkConfig;
pub const UnifiedMemoryConfig = network_mod.UnifiedMemoryConfig;
pub const LinkingConfig = network_mod.LinkingConfig;

// Observability configuration types
pub const ObservabilityConfig = observability_mod.ObservabilityConfig;

// Web configuration types
pub const WebConfig = web_mod.WebConfig;

// Plugin configuration types
pub const PluginConfig = plugins_mod.PluginConfig;

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
            .personas => if (self.ai) |ai| ai.personas != null else false,
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
    personas,
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
            .personas => "Multi-persona assistant",
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
            .ai, .llm, .embeddings, .agents, .training, .personas => build_options.enable_ai,
            .database => build_options.enable_database,
            .network => build_options.enable_network,
            .observability => build_options.enable_profiling,
            .web => build_options.enable_web,
        };
    }
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

    /// Enable personas with specified configuration.
    pub fn withPersonas(self: *Builder, personas_config: PersonasConfig) *Builder {
        if (self.config.ai == null) {
            self.config.ai = .{};
        }
        self.config.ai.?.personas = personas_config;
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
