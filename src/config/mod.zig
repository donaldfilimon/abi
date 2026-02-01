//! Configuration Module
//!
//! Re-exports all configuration types from domain-specific files.
//! Import this module for access to all configuration types.

const std = @import("std");
const build_options = @import("build_options");

// Domain-specific config imports
pub const gpu_config = @import("gpu.zig");
pub const ai_config = @import("ai.zig");
pub const database_config = @import("database.zig");
pub const network_config = @import("network.zig");
pub const observability_config = @import("observability.zig");
pub const web_config = @import("web.zig");
pub const plugin_config = @import("plugin.zig");
pub const loader = @import("loader.zig");

// Re-export loader types
pub const ConfigLoader = loader.ConfigLoader;
pub const LoadError = loader.LoadError;

// Re-export all config types for convenience
pub const GpuConfig = gpu_config.GpuConfig;
pub const RecoveryConfig = gpu_config.RecoveryConfig;

pub const AiConfig = ai_config.AiConfig;
pub const LlmConfig = ai_config.LlmConfig;
pub const EmbeddingsConfig = ai_config.EmbeddingsConfig;
pub const AgentsConfig = ai_config.AgentsConfig;
pub const TrainingConfig = ai_config.TrainingConfig;

pub const DatabaseConfig = database_config.DatabaseConfig;

pub const NetworkConfig = network_config.NetworkConfig;
pub const UnifiedMemoryConfig = network_config.UnifiedMemoryConfig;
pub const LinkingConfig = network_config.LinkingConfig;

pub const ObservabilityConfig = observability_config.ObservabilityConfig;

pub const WebConfig = web_config.WebConfig;

pub const PluginConfig = plugin_config.PluginConfig;

// ============================================================================
// Feature Enum
// ============================================================================

/// Available features in the framework.
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

    /// Number of features in the enum
    pub const feature_count = @typeInfo(Feature).@"enum".fields.len;

    /// Comptime-generated description lookup table for O(1) access.
    const DESCRIPTIONS: [feature_count][]const u8 = blk: {
        var descs: [feature_count][]const u8 = undefined;
        descs[@intFromEnum(Feature.gpu)] = "GPU acceleration and compute";
        descs[@intFromEnum(Feature.ai)] = "AI core functionality";
        descs[@intFromEnum(Feature.llm)] = "Local LLM inference";
        descs[@intFromEnum(Feature.embeddings)] = "Vector embeddings generation";
        descs[@intFromEnum(Feature.agents)] = "AI agent runtime";
        descs[@intFromEnum(Feature.training)] = "Model training pipelines";
        descs[@intFromEnum(Feature.database)] = "Vector database (WDBX)";
        descs[@intFromEnum(Feature.network)] = "Distributed compute network";
        descs[@intFromEnum(Feature.observability)] = "Metrics, tracing, profiling";
        descs[@intFromEnum(Feature.web)] = "Web/HTTP utilities";
        descs[@intFromEnum(Feature.personas)] = "Multi-persona AI assistant";
        descs[@intFromEnum(Feature.cloud)] = "Cloud provider integration";
        break :blk descs;
    };

    /// Comptime-generated compile-time enabled flags lookup table.
    const COMPILE_TIME_ENABLED: [feature_count]bool = blk: {
        var enabled: [feature_count]bool = undefined;
        enabled[@intFromEnum(Feature.gpu)] = build_options.enable_gpu;
        enabled[@intFromEnum(Feature.ai)] = build_options.enable_ai;
        enabled[@intFromEnum(Feature.llm)] = build_options.enable_ai;
        enabled[@intFromEnum(Feature.embeddings)] = build_options.enable_ai;
        enabled[@intFromEnum(Feature.agents)] = build_options.enable_ai;
        enabled[@intFromEnum(Feature.training)] = build_options.enable_ai;
        enabled[@intFromEnum(Feature.personas)] = build_options.enable_ai;
        enabled[@intFromEnum(Feature.database)] = build_options.enable_database;
        enabled[@intFromEnum(Feature.network)] = build_options.enable_network;
        enabled[@intFromEnum(Feature.observability)] = build_options.enable_profiling;
        enabled[@intFromEnum(Feature.web)] = build_options.enable_web;
        enabled[@intFromEnum(Feature.cloud)] = build_options.enable_web;
        break :blk enabled;
    };

    pub fn name(self: Feature) []const u8 {
        return @tagName(self);
    }

    /// Get feature description using O(1) comptime lookup table.
    pub fn description(self: Feature) []const u8 {
        return DESCRIPTIONS[@intFromEnum(self)];
    }

    /// Check if feature is compile-time enabled using O(1) comptime lookup table.
    pub fn isCompileTimeEnabled(self: Feature) bool {
        return COMPILE_TIME_ENABLED[@intFromEnum(self)];
    }
};

// ============================================================================
// Unified Config
// ============================================================================

/// Unified configuration for the ABI framework.
/// All feature configs are optional - null means the feature is disabled.
pub const Config = struct {
    gpu: ?GpuConfig = null,
    ai: ?AiConfig = null,
    database: ?DatabaseConfig = null,
    network: ?NetworkConfig = null,
    observability: ?ObservabilityConfig = null,
    web: ?WebConfig = null,
    plugins: PluginConfig = .{},

    /// Create a config with all compile-time enabled features using defaults.
    pub fn defaults() Config {
        return .{
            .gpu = if (build_options.enable_gpu) GpuConfig.defaults() else null,
            .ai = if (build_options.enable_ai) AiConfig.defaults() else null,
            .database = if (build_options.enable_database) DatabaseConfig.defaults() else null,
            .network = if (build_options.enable_network) NetworkConfig.defaults() else null,
            .observability = if (build_options.enable_profiling) ObservabilityConfig.defaults() else null,
            .web = if (build_options.enable_web) WebConfig.defaults() else null,
        };
    }

    /// Create a minimal config with no features enabled.
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
            .personas => if (self.ai) |ai| ai.personas != null else false,
            .cloud => self.web != null, // Cloud feature falls under web config
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

// ============================================================================
// Builder Pattern
// ============================================================================

/// Fluent builder for constructing Config.
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
        self.config = Config.defaults();
        return self;
    }

    pub fn withGpu(self: *Builder, cfg: GpuConfig) *Builder {
        self.config.gpu = cfg;
        return self;
    }

    pub fn withGpuDefaults(self: *Builder) *Builder {
        self.config.gpu = GpuConfig.defaults();
        return self;
    }

    pub fn withAi(self: *Builder, cfg: AiConfig) *Builder {
        self.config.ai = cfg;
        return self;
    }

    pub fn withAiDefaults(self: *Builder) *Builder {
        if (build_options.enable_ai) {
            self.config.ai = AiConfig.defaults();
        }
        return self;
    }

    pub fn withLlm(self: *Builder, cfg: LlmConfig) *Builder {
        if (self.config.ai == null) {
            self.config.ai = .{};
        }
        self.config.ai.?.llm = cfg;
        return self;
    }

    pub fn withDatabase(self: *Builder, cfg: DatabaseConfig) *Builder {
        self.config.database = cfg;
        return self;
    }

    pub fn withDatabaseDefaults(self: *Builder) *Builder {
        self.config.database = DatabaseConfig.defaults();
        return self;
    }

    pub fn withNetwork(self: *Builder, cfg: NetworkConfig) *Builder {
        self.config.network = cfg;
        return self;
    }

    pub fn withNetworkDefaults(self: *Builder) *Builder {
        self.config.network = NetworkConfig.defaults();
        return self;
    }

    pub fn withObservability(self: *Builder, cfg: ObservabilityConfig) *Builder {
        self.config.observability = cfg;
        return self;
    }

    pub fn withObservabilityDefaults(self: *Builder) *Builder {
        self.config.observability = ObservabilityConfig.defaults();
        return self;
    }

    pub fn withWeb(self: *Builder, cfg: WebConfig) *Builder {
        self.config.web = cfg;
        return self;
    }

    pub fn withWebDefaults(self: *Builder) *Builder {
        self.config.web = WebConfig.defaults();
        return self;
    }

    pub fn withPlugins(self: *Builder, cfg: PluginConfig) *Builder {
        self.config.plugins = cfg;
        return self;
    }

    pub fn build(self: *Builder) Config {
        return self.config;
    }
};

// ============================================================================
// Validation
// ============================================================================

pub const ConfigError = error{
    FeatureDisabled,
    InvalidConfig,
    MissingRequired,
    ConflictingConfig,
};

/// Validate configuration against compile-time constraints.
pub fn validate(config: Config) ConfigError!void {
    // Check GPU config against compile-time flag
    if (config.gpu != null and !build_options.enable_gpu) {
        return ConfigError.FeatureDisabled;
    }

    // Check AI config against compile-time flag
    if (config.ai != null and !build_options.enable_ai) {
        return ConfigError.FeatureDisabled;
    }

    // Check LLM config
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

test "Config.defaults respects build options" {
    const config = Config.defaults();
    if (build_options.enable_gpu) {
        try std.testing.expect(config.gpu != null);
    } else {
        try std.testing.expect(config.gpu == null);
    }
}

test "Config.minimal has no features" {
    const config = Config.minimal();
    try std.testing.expect(config.gpu == null);
    try std.testing.expect(config.ai == null);
    try std.testing.expect(config.database == null);
}

test "Builder creates valid config" {
    var builder = Builder.init(std.testing.allocator);
    const config = builder.withDefaults().build();
    try validate(config);
}

test "Feature.isCompileTimeEnabled" {
    // At least some feature should match build_options
    const gpu_enabled = Feature.gpu.isCompileTimeEnabled();
    try std.testing.expectEqual(build_options.enable_gpu, gpu_enabled);
}
