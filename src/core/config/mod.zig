//! Configuration Module
//!
//! Re-exports all configuration types from domain-specific files.
//! Import this module for access to all configuration types.
//!
//! Use `ConfigLoader` (see `loader.zig`) to load config from environment variables
//! (e.g. `ABI_GPU_BACKEND`, `ABI_LLM_MODEL_PATH`). Use `Config.Builder` for fluent construction.

const std = @import("std");
const build_options = @import("build_options");
const feature_catalog = @import("../feature_catalog.zig");

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
pub const mobile_config = @import("mobile.zig");
pub const gateway_config = @import("gateway.zig");
pub const pages_config = @import("pages.zig");
pub const benchmarks_config = @import("benchmarks.zig");
pub const plugin_config = @import("plugin.zig");
pub const loader = @import("loader.zig");

// Re-export loader types
pub const ConfigLoader = loader.ConfigLoader;
pub const LoadError = loader.LoadError;

// Re-export config types by domain (convenience; use *_config for defaults/helpers)
// Compute
pub const GpuConfig = gpu_config.GpuConfig;
pub const RecoveryConfig = gpu_config.RecoveryConfig;
pub const AiConfig = ai_config.AiConfig;
pub const LlmConfig = ai_config.LlmConfig;
pub const EmbeddingsConfig = ai_config.EmbeddingsConfig;
pub const AgentsConfig = ai_config.AgentsConfig;
pub const TrainingConfig = ai_config.TrainingConfig;
pub const ContentKind = ai_config.ContentKind;
pub const DatabaseConfig = database_config.DatabaseConfig;
// Network & platform
pub const NetworkConfig = network_config.NetworkConfig;
pub const UnifiedMemoryConfig = network_config.UnifiedMemoryConfig;
pub const LinkingConfig = network_config.LinkingConfig;
pub const ObservabilityConfig = observability_config.ObservabilityConfig;
pub const WebConfig = web_config.WebConfig;
pub const CloudConfig = cloud_config.CloudConfig;
pub const AnalyticsConfig = analytics_config.AnalyticsConfig;
pub const AuthConfig = auth_config.AuthConfig;
// Data & infra
pub const MessagingConfig = messaging_config.MessagingConfig;
pub const CacheConfig = cache_config.CacheConfig;
pub const StorageConfig = storage_config.StorageConfig;
pub const SearchConfig = search_config.SearchConfig;
pub const GatewayConfig = gateway_config.GatewayConfig;
pub const PagesConfig = pages_config.PagesConfig;
pub const BenchmarksConfig = benchmarks_config.BenchmarksConfig;
pub const MobileConfig = mobile_config.MobileConfig;
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
    mobile,
    gateway,
    pages,
    benchmarks,
    reasoning,

    /// Number of features in the enum
    pub const feature_count = @typeInfo(Feature).@"enum".fields.len;

    pub fn name(self: Feature) []const u8 {
        return @tagName(self);
    }

    /// Get feature description from the canonical feature catalog.
    pub fn description(self: Feature) []const u8 {
        return feature_catalog.descriptionFromEnum(self);
    }

    /// Check if feature is compile-time enabled via catalog flag mapping.
    pub fn isCompileTimeEnabled(self: Feature) bool {
        inline for (feature_catalog.all) |entry| {
            if (self == comptime feature_catalog.toEnum(Feature, entry.feature)) {
                return @field(build_options, entry.compile_flag_field);
            }
        }
        return false;
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
    mobile: ?MobileConfig = null,
    gateway: ?GatewayConfig = null,
    pages: ?PagesConfig = null,
    benchmarks: ?BenchmarksConfig = null,
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
            .mobile = if (build_options.enable_mobile) MobileConfig.defaults() else null,
            .gateway = if (build_options.enable_gateway) GatewayConfig.defaults() else null,
            .pages = if (build_options.enable_pages) PagesConfig.defaults() else null,
            .benchmarks = if (build_options.enable_benchmarks) BenchmarksConfig.defaults() else null,
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
            .mobile => self.mobile != null,
            .gateway => self.gateway != null,
            .pages => self.pages != null,
            .benchmarks => self.benchmarks != null,
            .reasoning => self.ai != null and build_options.enable_reasoning,
        };
    }

    /// Get list of enabled features.
    pub fn enabledFeatures(self: Config, allocator: std.mem.Allocator) ![]Feature {
        var list = std.ArrayListUnmanaged(Feature).empty;
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

    pub fn withMobile(self: *Builder, cfg: MobileConfig) *Builder {
        self.config.mobile = cfg;
        return self;
    }

    pub fn withMobileDefaults(self: *Builder) *Builder {
        self.config.mobile = MobileConfig.defaults();
        return self;
    }

    pub fn withGateway(self: *Builder, cfg: GatewayConfig) *Builder {
        self.config.gateway = cfg;
        return self;
    }

    pub fn withGatewayDefaults(self: *Builder) *Builder {
        self.config.gateway = GatewayConfig.defaults();
        return self;
    }

    pub fn withPages(self: *Builder, cfg: PagesConfig) *Builder {
        self.config.pages = cfg;
        return self;
    }

    pub fn withPagesDefaults(self: *Builder) *Builder {
        self.config.pages = PagesConfig.defaults();
        return self;
    }

    pub fn withBenchmarks(self: *Builder, cfg: BenchmarksConfig) *Builder {
        self.config.benchmarks = cfg;
        return self;
    }

    pub fn withBenchmarksDefaults(self: *Builder) *Builder {
        self.config.benchmarks = BenchmarksConfig.defaults();
        return self;
    }

    pub fn withPlugins(self: *Builder, cfg: PluginConfig) *Builder {
        self.config.plugins = cfg;
        return self;
    }

    /// Finalize and return the built config; no allocation.
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
        .{ .is_enabled_in_config = cfg.mobile != null, .is_enabled_at_build = build_options.enable_mobile },
        .{ .is_enabled_in_config = cfg.gateway != null, .is_enabled_at_build = build_options.enable_gateway },
        .{ .is_enabled_in_config = cfg.pages != null, .is_enabled_at_build = build_options.enable_pages },
        .{ .is_enabled_in_config = cfg.benchmarks != null, .is_enabled_at_build = build_options.enable_benchmarks },
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
