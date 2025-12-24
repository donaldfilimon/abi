const std = @import("std");
const features = @import("../features/mod.zig");
const runtime = @import("runtime.zig");
const core = @import("../core/mod.zig");

/// Enumerates the coarse feature families that can be toggled at runtime.
pub const Feature = features.FeatureTag;

pub const feature_count = features.feature_count;
const FeatureMask = std.bit_set.IntegerBitSet(feature_count);

/// Bit-set backed feature selection utility used by the framework runtime.
pub const FeatureToggles = struct {
    mask: FeatureMask = FeatureMask.initEmpty(),

    pub fn enable(self: *FeatureToggles, feature: Feature) void {
        self.mask.set(@intFromEnum(feature));
    }

    pub fn disable(self: *FeatureToggles, feature: Feature) void {
        self.mask.unset(@intFromEnum(feature));
    }

    pub fn set(self: *FeatureToggles, feature: Feature, value: bool) void {
        if (value) {
            self.enable(feature);
        } else {
            self.disable(feature);
        }
    }

    pub fn enableMany(self: *FeatureToggles, feature_list: []const Feature) void {
        for (feature_list) |feature| {
            self.enable(feature);
        }
    }

    pub fn disableMany(self: *FeatureToggles, feature_list: []const Feature) void {
        for (feature_list) |feature| {
            self.disable(feature);
        }
    }

    pub fn isEnabled(self: FeatureToggles, feature: Feature) bool {
        return self.mask.isSet(@intFromEnum(feature));
    }

    pub fn count(self: FeatureToggles) usize {
        return self.mask.count();
    }

    pub fn clear(self: *FeatureToggles) void {
        self.mask = FeatureMask.initEmpty();
    }

    pub fn iterator(self: FeatureToggles) FeatureIterator {
        return .{ .mask = self.mask, .index = 0 };
    }

    pub fn toOwnedSlice(self: FeatureToggles, allocator: std.mem.Allocator) ![]Feature {
        var list = try allocator.alloc(Feature, self.count());
        var iter = self.iterator();
        var idx: usize = 0;
        while (iter.next()) |feature| : (idx += 1) {
            list[idx] = feature;
        }
        return list;
    }
};

/// Iterator used to traverse enabled features.
pub const FeatureIterator = struct {
    mask: FeatureMask,
    index: usize,

    pub fn next(self: *FeatureIterator) ?Feature {
        while (self.index < feature_count) : (self.index += 1) {
            if (self.mask.isSet(self.index)) {
                const feature = @as(Feature, @enumFromInt(self.index));
                self.index += 1;
                return feature;
            }
        }
        return null;
    }
};

/// Human readable name for a feature.
pub fn featureLabel(feature: Feature) []const u8 {
    return switch (feature) {
        .ai => "AI/Agents",
        .database => "Vector Database",
        .web => "Web Services",
        .monitoring => "Monitoring",
        .gpu => "GPU Acceleration",
        .connectors => "External Connectors",
        .simd => "SIMD Runtime",
    };
}

/// Short description describing the role of each feature for summary output.
pub fn featureDescription(feature: Feature) []const u8 {
    return switch (feature) {
        .ai => "Conversation agents, training loops, and inference helpers",
        .database => "High-performance embedding and vector persistence layer",
        .web => "HTTP servers, clients, and gateway orchestration",
        .monitoring => "Instrumentation, telemetry, and health checks",
        .gpu => "GPU kernel dispatch and compute pipelines",
        .connectors => "Third-party integrations and adapters",
        .simd => "Runtime SIMD utilities and fast math operations",
    };
}

/// Unified framework configuration that consolidates all configuration types
pub const FrameworkConfiguration = struct {
    // === FEATURE MANAGEMENT ===
    /// Explicit feature set. When provided, overrides boolean toggles.
    enabled_features: ?[]const Feature = null,
    /// Features that should be disabled even if present in `enabled_features`
    disabled_features: []const Feature = &.{},

    // Convenience booleans matching the public quick-start documentation.
    enable_ai: bool = true,
    enable_database: bool = true,
    enable_web: bool = true,
    enable_monitoring: bool = true,
    enable_gpu: bool = false,
    enable_connectors: bool = false,
    enable_simd: bool = true,

    // === RUNTIME SETTINGS ===
    max_plugins: u32 = 128,
    enable_hot_reload: bool = false,
    enable_profiling: bool = false,
    memory_limit_mb: ?u32 = null,
    log_level: runtime.RuntimeConfig.LogLevel = .info,

    /// Plugin loader configuration.
    plugin_paths: []const []const u8 = &.{},
    auto_discover_plugins: bool = false,
    auto_register_plugins: bool = false,
    auto_start_plugins: bool = false,

    // === OPERATIONAL SETTINGS ===
    // Core features
    enable_memory_tracking: bool = true,
    enable_performance_profiling: bool = true,

    // Concurrency settings
    max_concurrent_agents: u32 = 10,
    max_concurrent_requests: u32 = 1000,
    thread_pool_size: u32 = 4,

    // Plugin system (detailed)
    plugin_directory: []const u8 = "plugins/",
    enable_plugin_hot_reload: bool = true,

    // Logging (detailed)
    log_file: ?[]const u8 = null,
    enable_structured_logging: bool = true,
    profile: core.profile.ProfileKind = .dev,
    persona_manifest_path: []const u8 = "config/personas/default.json",

    // Performance
    enable_compression: bool = true,
    enable_caching: bool = true,
    cache_size_mb: u32 = 256,

    // Security
    enable_authentication: bool = false,
    enable_encryption: bool = false,
    max_request_size_mb: u32 = 10,

    // Database
    database_path: []const u8 = "data/",
    enable_database_compression: bool = true,
    max_database_size_gb: u32 = 10,

    // Web server
    web_server_port: u16 = 8080,
    web_server_host: []const u8 = "0.0.0.0",
    enable_websocket: bool = true,
    enable_cors: bool = true,

    // Monitoring
    enable_metrics: bool = true,
    metrics_port: u16 = 9090,
    enable_health_checks: bool = true,

    /// Validate the unified configuration
    pub fn validate(self: FrameworkConfiguration) !void {
        // Validate concurrency settings
        if (self.max_concurrent_agents == 0) {
            return error.InvalidConfiguration;
        }
        if (self.max_concurrent_requests == 0) {
            return error.InvalidConfiguration;
        }
        if (self.thread_pool_size == 0) {
            return error.InvalidConfiguration;
        }

        // Validate plugin settings
        if (self.max_plugins == 0) {
            return error.InvalidConfiguration;
        }

        // Validate performance settings
        if (self.cache_size_mb == 0) {
            return error.InvalidConfiguration;
        }

        // Validate security settings
        if (self.max_request_size_mb == 0) {
            return error.InvalidConfiguration;
        }

        // Validate database settings
        if (self.max_database_size_gb == 0) {
            return error.InvalidConfiguration;
        }

        // Validate web server settings
        if (self.web_server_port == 0) {
            return error.InvalidConfiguration;
        }

        // Validate monitoring settings
        if (self.enable_metrics and self.metrics_port == 0) {
            return error.InvalidConfiguration;
        }
    }

    /// Create a default configuration
    pub fn default() FrameworkConfiguration {
        return FrameworkConfiguration{};
    }

    /// Create a minimal configuration for testing
    pub fn minimal() FrameworkConfiguration {
        return FrameworkConfiguration{
            .enable_gpu = false,
            .enable_simd = false,
            .enable_memory_tracking = false,
            .enable_performance_profiling = false,
            .max_concurrent_agents = 1,
            .max_concurrent_requests = 10,
            .thread_pool_size = 1,
            .plugin_directory = "test_plugins/",
            .enable_plugin_hot_reload = false,
            .max_plugins = 5,
            .log_level = .debug,
            .enable_structured_logging = false,
            .profile = .testing,
            .persona_manifest_path = "config/personas/default.json",
            .enable_compression = false,
            .enable_caching = false,
            .cache_size_mb = 1,
            .enable_authentication = false,
            .enable_encryption = false,
            .max_request_size_mb = 1,
            .database_path = "test_data/",
            .enable_database_compression = false,
            .max_database_size_gb = 1,
            .web_server_port = 8081,
            .web_server_host = "127.0.0.1",
            .enable_websocket = false,
            .enable_cors = false,
            .enable_metrics = false,
            .metrics_port = 9091,
            .enable_health_checks = false,
        };
    }

    /// Create a production configuration
    pub fn production() FrameworkConfiguration {
        return FrameworkConfiguration{
            .enable_gpu = true,
            .enable_simd = true,
            .enable_memory_tracking = true,
            .enable_performance_profiling = true,
            .max_concurrent_agents = 100,
            .max_concurrent_requests = 10000,
            .thread_pool_size = 16,
            .plugin_directory = "/opt/abi/plugins/",
            .enable_plugin_hot_reload = false,
            .max_plugins = 100,
            .log_level = .info,
            .log_file = "/var/log/abi/framework.log",
            .enable_structured_logging = true,
            .profile = .prod,
            .persona_manifest_path = "/etc/abi/personas.json",
            .enable_compression = true,
            .enable_caching = true,
            .cache_size_mb = 1024,
            .enable_authentication = true,
            .enable_encryption = true,
            .max_request_size_mb = 100,
            .database_path = "/opt/abi/data/",
            .enable_database_compression = true,
            .max_database_size_gb = 100,
            .web_server_port = 8080,
            .web_server_host = "0.0.0.0",
            .enable_websocket = true,
            .enable_cors = true,
            .enable_metrics = true,
            .metrics_port = 9090,
            .enable_health_checks = true,
        };
    }

    /// Convert to RuntimeConfig for compatibility
    pub fn toRuntimeConfig(self: FrameworkConfiguration, allocator: std.mem.Allocator) !runtime.RuntimeConfig {
        const feature_capacity = feature_count;
        var enabled = std.ArrayList(Feature).initCapacity(allocator, feature_capacity) catch unreachable;
        defer enabled.deinit(allocator);
        var toggles = deriveFeatureTogglesFromConfig(self);
        var iterator = toggles.iterator();
        while (iterator.next()) |feature| {
            enabled.appendAssumeCapacity(feature);
        }

        var disabled = std.ArrayList(Feature).initCapacity(allocator, feature_capacity) catch unreachable;
        defer disabled.deinit(allocator);
        for (self.disabled_features) |feature| {
            disabled.appendAssumeCapacity(feature);
        }

        var config = runtime.RuntimeConfig{
            .max_plugins = self.max_plugins,
            .enable_hot_reload = self.enable_hot_reload,
            .enable_profiling = self.enable_profiling,
            .memory_limit_mb = self.memory_limit_mb,
            .log_level = self.log_level,
            .plugin_paths = try allocator.dupe([]const u8, self.plugin_paths),
            .auto_discover_plugins = self.auto_discover_plugins,
            .auto_register_plugins = self.auto_register_plugins,
            .auto_start_plugins = self.auto_start_plugins,
        };
        config.feature_storage.setEnabled(enabled.items);
        config.feature_storage.setDisabled(disabled.items);
        config.enabled_features = config.feature_storage.enabledSlice();
        config.disabled_features = config.feature_storage.disabledSlice();
        return config;
    }

    /// Convert to legacy FrameworkConfig for compatibility
    pub fn toFrameworkConfig(self: FrameworkConfiguration) core.config.FrameworkConfig {
        return core.config.FrameworkConfig{
            .enable_gpu = self.enable_gpu,
            .enable_simd = self.enable_simd,
            .enable_memory_tracking = self.enable_memory_tracking,
            .enable_performance_profiling = self.enable_performance_profiling,
            .max_concurrent_agents = self.max_concurrent_agents,
            .max_concurrent_requests = self.max_concurrent_requests,
            .thread_pool_size = self.thread_pool_size,
            .plugin_directory = self.plugin_directory,
            .enable_plugin_hot_reload = self.enable_plugin_hot_reload,
            .max_plugins = self.max_plugins,
            .log_level = @enumFromInt(@intFromEnum(self.log_level)),
            .log_file = self.log_file,
            .enable_structured_logging = self.enable_structured_logging,
            .profile = self.profile,
            .persona_manifest_path = self.persona_manifest_path,
            .enable_compression = self.enable_compression,
            .enable_caching = self.enable_caching,
            .cache_size_mb = self.cache_size_mb,
            .enable_authentication = self.enable_authentication,
            .enable_encryption = self.enable_encryption,
            .max_request_size_mb = self.max_request_size_mb,
            .database_path = self.database_path,
            .enable_database_compression = self.enable_database_compression,
            .max_database_size_gb = self.max_database_size_gb,
            .web_server_port = self.web_server_port,
            .web_server_host = self.web_server_host,
            .enable_websocket = self.enable_websocket,
            .enable_cors = self.enable_cors,
            .enable_metrics = self.enable_metrics,
            .metrics_port = self.metrics_port,
            .enable_health_checks = self.enable_health_checks,
        };
    }

    /// Create from legacy FrameworkConfig
    pub fn fromFrameworkConfig(legacy: core.config.FrameworkConfig) FrameworkConfiguration {
        return FrameworkConfiguration{
            .enable_gpu = legacy.enable_gpu,
            .enable_simd = legacy.enable_simd,
            .enable_memory_tracking = legacy.enable_memory_tracking,
            .enable_performance_profiling = legacy.enable_performance_profiling,
            .max_concurrent_agents = legacy.max_concurrent_agents,
            .max_concurrent_requests = legacy.max_concurrent_requests,
            .thread_pool_size = legacy.thread_pool_size,
            .plugin_directory = legacy.plugin_directory,
            .enable_plugin_hot_reload = legacy.enable_plugin_hot_reload,
            .max_plugins = legacy.max_plugins,
            .log_level = @enumFromInt(@intFromEnum(legacy.log_level)),
            .log_file = legacy.log_file,
            .enable_structured_logging = legacy.enable_structured_logging,
            .profile = legacy.profile,
            .persona_manifest_path = legacy.persona_manifest_path,
            .enable_compression = legacy.enable_compression,
            .enable_caching = legacy.enable_caching,
            .cache_size_mb = legacy.cache_size_mb,
            .enable_authentication = legacy.enable_authentication,
            .enable_encryption = legacy.enable_encryption,
            .max_request_size_mb = legacy.max_request_size_mb,
            .database_path = legacy.database_path,
            .enable_database_compression = legacy.enable_database_compression,
            .max_database_size_gb = legacy.max_database_size_gb,
            .web_server_port = legacy.web_server_port,
            .web_server_host = legacy.web_server_host,
            .enable_websocket = legacy.enable_websocket,
            .enable_cors = legacy.enable_cors,
            .enable_metrics = legacy.enable_metrics,
            .metrics_port = legacy.metrics_port,
            .enable_health_checks = legacy.enable_health_checks,
        };
    }

    /// Create from RuntimeConfig
    pub fn fromRuntimeConfig(legacy: runtime.RuntimeConfig) FrameworkConfiguration {
        return FrameworkConfiguration{
            .max_plugins = legacy.max_plugins,
            .enable_hot_reload = legacy.enable_hot_reload,
            .enable_profiling = legacy.enable_profiling,
            .memory_limit_mb = legacy.memory_limit_mb,
            .log_level = legacy.log_level,
            .enabled_features = legacy.enabled_features,
            .disabled_features = legacy.disabled_features,
            .plugin_paths = legacy.plugin_paths,
            .auto_discover_plugins = legacy.auto_discover_plugins,
            .auto_register_plugins = legacy.auto_register_plugins,
            .auto_start_plugins = legacy.auto_start_plugins,
        };
    }
};

/// Configuration supplied when bootstrapping the framework.
/// @deprecated Use FrameworkConfiguration for new code
pub const FrameworkOptions = struct {
    /// Optional explicit feature set. When provided all boolean toggles are
    /// ignored in favour of this list.
    enabled_features: ?[]const Feature = null,
    /// Features that should be disabled even if present in `enabled_features`
    /// or enabled through the boolean convenience flags.
    disabled_features: []const Feature = &.{},

    // Convenience booleans matching the public quick-start documentation.
    enable_ai: bool = true,
    enable_database: bool = true,
    enable_web: bool = true,
    enable_monitoring: bool = true,
    enable_gpu: bool = false,
    enable_connectors: bool = false,
    enable_simd: bool = true,

    /// Plugin loader configuration.
    plugin_paths: []const []const u8 = &.{},
    auto_discover_plugins: bool = false,
    auto_register_plugins: bool = false,
    auto_start_plugins: bool = false,

    /// Convert to unified FrameworkConfiguration
    pub fn toUnifiedConfig(self: FrameworkOptions) FrameworkConfiguration {
        return FrameworkConfiguration{
            .enabled_features = self.enabled_features,
            .disabled_features = self.disabled_features,
            .enable_ai = self.enable_ai,
            .enable_database = self.enable_database,
            .enable_web = self.enable_web,
            .enable_monitoring = self.enable_monitoring,
            .enable_gpu = self.enable_gpu,
            .enable_connectors = self.enable_connectors,
            .enable_simd = self.enable_simd,
            .plugin_paths = self.plugin_paths,
            .auto_discover_plugins = self.auto_discover_plugins,
            .auto_register_plugins = self.auto_register_plugins,
            .auto_start_plugins = self.auto_start_plugins,
        };
    }
};

/// Compute the feature toggles implied by the provided options.
pub fn deriveFeatureToggles(options: FrameworkOptions) FeatureToggles {
    var toggles = FeatureToggles{};

    if (options.enabled_features) |enabled_features| {
        toggles.enableMany(enabled_features);
    } else {
        toggles.set(.ai, options.enable_ai);
        toggles.set(.database, options.enable_database);
        toggles.set(.web, options.enable_web);
        toggles.set(.monitoring, options.enable_monitoring);
        toggles.set(.gpu, options.enable_gpu);
        toggles.set(.connectors, options.enable_connectors);
        toggles.set(.simd, options.enable_simd);
    }

    toggles.disableMany(options.disabled_features);
    return toggles;
}

/// Derive feature toggles from unified configuration
pub fn deriveFeatureTogglesFromConfig(config: FrameworkConfiguration) FeatureToggles {
    var toggles = FeatureToggles{};

    if (config.enabled_features) |enabled_features| {
        toggles.enableMany(enabled_features);
    } else {
        toggles.set(.ai, config.enable_ai);
        toggles.set(.database, config.enable_database);
        toggles.set(.web, config.enable_web);
        toggles.set(.monitoring, config.enable_monitoring);
        toggles.set(.gpu, config.enable_gpu);
        toggles.set(.connectors, config.enable_connectors);
        toggles.set(.simd, config.enable_simd);
    }

    toggles.disableMany(config.disabled_features);
    return toggles;
}

/// Convert framework options into a runtime configuration without heap allocations.
/// @deprecated Use FrameworkConfiguration.toRuntimeConfig() for new code
pub fn runtimeConfigFromOptions(
    allocator: std.mem.Allocator,
    options: FrameworkOptions,
) !runtime.RuntimeConfig {
    const feature_capacity = feature_count;

    var enabled = std.ArrayList(Feature).initCapacity(allocator, feature_capacity) catch unreachable;
    defer enabled.deinit(allocator);
    var toggles = deriveFeatureToggles(options);
    var iterator = toggles.iterator();
    while (iterator.next()) |feature| {
        enabled.appendAssumeCapacity(feature);
    }

    var disabled = std.ArrayList(Feature).initCapacity(allocator, feature_capacity) catch unreachable;
    defer disabled.deinit(allocator);
    for (options.disabled_features) |feature| {
        disabled.appendAssumeCapacity(feature);
    }

    var config = runtime.RuntimeConfig{
        .plugin_paths = options.plugin_paths,
        .auto_discover_plugins = options.auto_discover_plugins,
        .auto_register_plugins = options.auto_register_plugins,
        .auto_start_plugins = options.auto_start_plugins,
    };
    config.feature_storage.setEnabled(enabled.items);
    config.feature_storage.setDisabled(disabled.items);
    config.enabled_features = config.feature_storage.enabledSlice();
    config.disabled_features = config.feature_storage.disabledSlice();
    return config;
}

test "feature toggles enable and disable entries" {
    var toggles = FeatureToggles{};
    try std.testing.expectEqual(@as(usize, 0), toggles.count());

    toggles.enableMany(&.{ .ai, .database, .web });
    try std.testing.expect(toggles.isEnabled(.ai));
    try std.testing.expect(toggles.isEnabled(.database));
    try std.testing.expect(toggles.isEnabled(.web));
    try std.testing.expectEqual(@as(usize, 3), toggles.count());

    toggles.disable(.database);
    try std.testing.expect(!toggles.isEnabled(.database));
    try std.testing.expectEqual(@as(usize, 2), toggles.count());
}

test "deriveFeatureToggles respects overrides" {
    const overrides = FrameworkOptions{
        .enabled_features = &.{ .ai, .gpu },
        .disabled_features = &.{.gpu},
    };
    const toggles = deriveFeatureToggles(overrides);
    try std.testing.expect(toggles.isEnabled(.ai));
    try std.testing.expect(!toggles.isEnabled(.gpu));
    try std.testing.expect(!toggles.isEnabled(.database));
}

test "deriveFeatureToggles maps booleans and applies disabled list" {
    const options = FrameworkOptions{
        .enable_ai = false,
        .enable_database = true,
        .enable_web = true,
        .enable_monitoring = false,
        .enable_gpu = true,
        .enable_connectors = true,
        .enable_simd = false,
        .disabled_features = &.{ .web, .gpu },
    };
    const toggles = deriveFeatureToggles(options);

    try std.testing.expect(!toggles.isEnabled(.ai));
    try std.testing.expect(toggles.isEnabled(.database));
    try std.testing.expect(!toggles.isEnabled(.web));
    try std.testing.expect(!toggles.isEnabled(.gpu));
    try std.testing.expect(!toggles.isEnabled(.monitoring));
    try std.testing.expect(toggles.isEnabled(.connectors));
    try std.testing.expect(!toggles.isEnabled(.simd));
    try std.testing.expectEqual(@as(usize, 2), toggles.count());
}
