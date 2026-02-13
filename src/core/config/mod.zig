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
pub const cloud_config = @import("cloud.zig");
pub const analytics_config = @import("analytics.zig");
pub const auth_config = @import("auth.zig");
pub const messaging_config = @import("messaging.zig");
pub const cache_config = @import("cache.zig");
pub const storage_config = @import("storage.zig");
pub const search_config = @import("search.zig");
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

pub const CloudConfig = cloud_config.CloudConfig;

pub const AnalyticsConfig = analytics_config.AnalyticsConfig;

pub const AuthConfig = auth_config.AuthConfig;
pub const MessagingConfig = messaging_config.MessagingConfig;
pub const CacheConfig = cache_config.CacheConfig;
pub const StorageConfig = storage_config.StorageConfig;
pub const SearchConfig = search_config.SearchConfig;

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
    analytics,
    auth,
    messaging,
    cache,
    storage,
    search,

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
        descs[@intFromEnum(Feature.analytics)] = "Analytics event tracking";
        descs[@intFromEnum(Feature.auth)] = "Authentication and security";
        descs[@intFromEnum(Feature.messaging)] = "Event bus and messaging";
        descs[@intFromEnum(Feature.cache)] = "In-memory caching";
        descs[@intFromEnum(Feature.storage)] = "Unified file/object storage";
        descs[@intFromEnum(Feature.search)] = "Full-text search";
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
        enabled[@intFromEnum(Feature.cloud)] = build_options.enable_cloud;
        enabled[@intFromEnum(Feature.analytics)] = build_options.enable_analytics;
        enabled[@intFromEnum(Feature.auth)] = build_options.enable_auth;
        enabled[@intFromEnum(Feature.messaging)] = build_options.enable_messaging;
        enabled[@intFromEnum(Feature.cache)] = build_options.enable_cache;
        enabled[@intFromEnum(Feature.storage)] = build_options.enable_storage;
        enabled[@intFromEnum(Feature.search)] = build_options.enable_search;
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
    cloud: ?CloudConfig = null,
    analytics: ?AnalyticsConfig = null,
    auth: ?AuthConfig = null,
    messaging: ?MessagingConfig = null,
    cache: ?CacheConfig = null,
    storage: ?StorageConfig = null,
    search: ?SearchConfig = null,
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
            .cloud = if (build_options.enable_cloud) CloudConfig.defaults() else null,
            .analytics = if (build_options.enable_analytics) AnalyticsConfig.defaults() else null,
            .auth = if (build_options.enable_auth) AuthConfig.defaults() else null,
            .messaging = if (build_options.enable_messaging) MessagingConfig.defaults() else null,
            .cache = if (build_options.enable_cache) CacheConfig.defaults() else null,
            .storage = if (build_options.enable_storage) StorageConfig.defaults() else null,
            .search = if (build_options.enable_search) SearchConfig.defaults() else null,
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
            .cloud => self.cloud != null,
            .analytics => self.analytics != null,
            .auth => self.auth != null,
            .messaging => self.messaging != null,
            .cache => self.cache != null,
            .storage => self.storage != null,
            .search => self.search != null,
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

    pub fn withCloud(self: *Builder, cfg: CloudConfig) *Builder {
        self.config.cloud = cfg;
        return self;
    }

    pub fn withCloudDefaults(self: *Builder) *Builder {
        self.config.cloud = CloudConfig.defaults();
        return self;
    }

    pub fn withAnalytics(self: *Builder, cfg: AnalyticsConfig) *Builder {
        self.config.analytics = cfg;
        return self;
    }

    pub fn withAnalyticsDefaults(self: *Builder) *Builder {
        self.config.analytics = AnalyticsConfig.defaults();
        return self;
    }

    pub fn withAuth(self: *Builder, cfg: AuthConfig) *Builder {
        self.config.auth = cfg;
        return self;
    }

    pub fn withAuthDefaults(self: *Builder) *Builder {
        self.config.auth = AuthConfig.defaults();
        return self;
    }

    pub fn withMessaging(self: *Builder, cfg: MessagingConfig) *Builder {
        self.config.messaging = cfg;
        return self;
    }

    pub fn withMessagingDefaults(self: *Builder) *Builder {
        self.config.messaging = MessagingConfig.defaults();
        return self;
    }

    pub fn withCache(self: *Builder, cfg: CacheConfig) *Builder {
        self.config.cache = cfg;
        return self;
    }

    pub fn withCacheDefaults(self: *Builder) *Builder {
        self.config.cache = CacheConfig.defaults();
        return self;
    }

    pub fn withStorage(self: *Builder, cfg: StorageConfig) *Builder {
        self.config.storage = cfg;
        return self;
    }

    pub fn withStorageDefaults(self: *Builder) *Builder {
        self.config.storage = StorageConfig.defaults();
        return self;
    }

    pub fn withSearch(self: *Builder, cfg: SearchConfig) *Builder {
        self.config.search = cfg;
        return self;
    }

    pub fn withSearchDefaults(self: *Builder) *Builder {
        self.config.search = SearchConfig.defaults();
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

const FeatureValidation = struct {
    is_enabled_in_config: bool,
    is_enabled_at_build: bool,
};

/// Validate configuration against compile-time constraints.
pub fn validate(cfg: Config) ConfigError!void {
    const validations = [_]FeatureValidation{
        .{ .is_enabled_in_config = cfg.gpu != null, .is_enabled_at_build = build_options.enable_gpu },
        .{ .is_enabled_in_config = cfg.ai != null, .is_enabled_at_build = build_options.enable_ai },
        .{ .is_enabled_in_config = cfg.database != null, .is_enabled_at_build = build_options.enable_database },
        .{ .is_enabled_in_config = cfg.network != null, .is_enabled_at_build = build_options.enable_network },
        .{ .is_enabled_in_config = cfg.web != null, .is_enabled_at_build = build_options.enable_web },
        .{ .is_enabled_in_config = cfg.cloud != null, .is_enabled_at_build = build_options.enable_cloud },
        .{ .is_enabled_in_config = cfg.analytics != null, .is_enabled_at_build = build_options.enable_analytics },
        .{ .is_enabled_in_config = cfg.observability != null, .is_enabled_at_build = build_options.enable_profiling },
        .{ .is_enabled_in_config = cfg.auth != null, .is_enabled_at_build = build_options.enable_auth },
        .{ .is_enabled_in_config = cfg.messaging != null, .is_enabled_at_build = build_options.enable_messaging },
        .{ .is_enabled_in_config = cfg.cache != null, .is_enabled_at_build = build_options.enable_cache },
        .{ .is_enabled_in_config = cfg.storage != null, .is_enabled_at_build = build_options.enable_storage },
        .{ .is_enabled_in_config = cfg.search != null, .is_enabled_at_build = build_options.enable_search },
    };
    inline for (validations) |entry| {
        if (entry.is_enabled_in_config and !entry.is_enabled_at_build) {
            return ConfigError.FeatureDisabled;
        }
    }

    // LLM is nested under AI and has its own compile-time flag.
    if (cfg.ai) |ai| {
        if (ai.llm != null and !build_options.enable_llm) {
            return ConfigError.FeatureDisabled;
        }
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
    try std.testing.expect(config.cloud == null);
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

test "validate returns FeatureDisabled for compile-time disabled features" {
    if (!build_options.enable_gpu) {
        var config = Config.minimal();
        config.gpu = GpuConfig.defaults();
        try std.testing.expectError(ConfigError.FeatureDisabled, validate(config));
    }

    if (!build_options.enable_ai) {
        var config = Config.minimal();
        config.ai = AiConfig.defaults();
        try std.testing.expectError(ConfigError.FeatureDisabled, validate(config));
    }

    if (!build_options.enable_database) {
        var config = Config.minimal();
        config.database = DatabaseConfig.defaults();
        try std.testing.expectError(ConfigError.FeatureDisabled, validate(config));
    }

    if (!build_options.enable_network) {
        var config = Config.minimal();
        config.network = NetworkConfig.defaults();
        try std.testing.expectError(ConfigError.FeatureDisabled, validate(config));
    }

    if (!build_options.enable_web) {
        var web_cfg = Config.minimal();
        web_cfg.web = WebConfig.defaults();
        try std.testing.expectError(ConfigError.FeatureDisabled, validate(web_cfg));
    }

    if (!build_options.enable_cloud) {
        var cloud_cfg = Config.minimal();
        cloud_cfg.cloud = CloudConfig.defaults();
        try std.testing.expectError(ConfigError.FeatureDisabled, validate(cloud_cfg));
    }

    if (!build_options.enable_analytics) {
        var config = Config.minimal();
        config.analytics = AnalyticsConfig.defaults();
        try std.testing.expectError(ConfigError.FeatureDisabled, validate(config));
    }

    if (!build_options.enable_profiling) {
        var config = Config.minimal();
        config.observability = ObservabilityConfig.defaults();
        try std.testing.expectError(ConfigError.FeatureDisabled, validate(config));
    }
}

test "validate returns FeatureDisabled for llm when llm build flag is disabled" {
    if (build_options.enable_ai and !build_options.enable_llm) {
        var config = Config.minimal();
        var ai = AiConfig.defaults();
        ai.llm = LlmConfig.defaults();
        config.ai = ai;
        try std.testing.expectError(ConfigError.FeatureDisabled, validate(config));
    }
}
