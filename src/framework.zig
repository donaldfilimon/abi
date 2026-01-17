//! Framework Orchestration Layer
//!
//! Manages the lifecycle of the ABI framework, coordinating feature
//! initialization, configuration, and runtime state.
//!
//! ## Usage
//!
//! ```zig
//! const abi = @import("abi");
//!
//! // Using init with defaults
//! var fw = try abi.init(allocator);
//! defer fw.deinit();
//!
//! // Using builder pattern
//! var fw = try abi.Framework.builder(allocator)
//!     .withGpu(.{ .backend = .vulkan })
//!     .withAi(.{ .llm = .{} })
//!     .build();
//! defer fw.deinit();
//!
//! // Check feature status
//! if (fw.isEnabled(.gpu)) {
//!     // Use GPU features
//! }
//! ```

const std = @import("std");
const build_options = @import("build_options");
const config_module = @import("config.zig");
const registry_mod = @import("registry/mod.zig");

pub const Config = config_module.Config;
pub const Feature = config_module.Feature;
pub const ConfigError = config_module.ConfigError;
pub const Registry = registry_mod.Registry;

// Feature modules - imported based on build configuration
const gpu_mod = if (build_options.enable_gpu) @import("gpu/mod.zig") else @import("gpu/stub.zig");
const ai_mod = if (build_options.enable_ai) @import("ai/mod.zig") else @import("ai/stub.zig");
const database_mod = if (build_options.enable_database) @import("database/mod.zig") else @import("database/stub.zig");
const network_mod = if (build_options.enable_network) @import("network/mod.zig") else @import("network/stub.zig");
const observability_mod = if (build_options.enable_profiling) @import("observability/mod.zig") else @import("observability/stub.zig");
const web_mod = if (build_options.enable_web) @import("web/mod.zig") else @import("web/stub.zig");
const runtime_mod = @import("runtime/mod.zig");

/// Framework orchestration handle.
/// Manages lifecycle of all enabled features.
pub const Framework = struct {
    allocator: std.mem.Allocator,
    config: Config,
    state: State,
    registry: Registry,

    // Feature handles (null if disabled)
    gpu: ?*gpu_mod.Context = null,
    ai: ?*ai_mod.Context = null,
    database: ?*database_mod.Context = null,
    network: ?*network_mod.Context = null,
    observability: ?*observability_mod.Context = null,
    web: ?*web_mod.Context = null,
    runtime: *runtime_mod.Context,

    pub const State = enum {
        uninitialized,
        initializing,
        running,
        stopping,
        stopped,
        failed,
    };

    pub const Error = error{
        AlreadyInitialized,
        NotInitialized,
        InitializationFailed,
        FeatureInitFailed,
        FeatureDisabled,
        InvalidState,
        EngineCreationFailed,
        // GPU errors
        GpuDisabled,
        NoDeviceAvailable,
        InvalidConfig,
        KernelCompilationFailed,
        KernelExecutionFailed,
        // AI errors
        AiDisabled,
        LlmDisabled,
        EmbeddingsDisabled,
        AgentsDisabled,
        TrainingDisabled,
        ModelNotFound,
        InferenceFailed,
        // Database errors
        DatabaseDisabled,
        ConnectionFailed,
        QueryFailed,
        IndexError,
        StorageError,
        // Network errors
        NetworkDisabled,
        NodeNotFound,
        ConsensusFailed,
        Timeout,
        // Observability errors
        ObservabilityDisabled,
        MetricsError,
        TracingError,
        ExportFailed,
        // Web errors
        WebDisabled,
        RequestFailed,
        InvalidUrl,
    } || std.mem.Allocator.Error || ConfigError || Registry.Error;

    /// Initialize the framework with the given configuration.
    pub fn init(allocator: std.mem.Allocator, cfg: Config) Error!Framework {
        // Validate configuration against compile-time constraints
        try config_module.validate(cfg);

        var fw = Framework{
            .allocator = allocator,
            .config = cfg,
            .state = .initializing,
            .registry = Registry.init(allocator),
            .runtime = undefined,
        };
        errdefer fw.registry.deinit();

        // Initialize runtime (always available)
        fw.runtime = try runtime_mod.Context.init(allocator);
        errdefer fw.runtime.deinit();

        // Initialize enabled features and register them
        errdefer fw.deinitFeatures();

        if (cfg.gpu) |gpu_cfg| {
            fw.gpu = try gpu_mod.Context.init(allocator, gpu_cfg);
            if (comptime build_options.enable_gpu) {
                try fw.registry.registerComptime(.gpu);
            }
        }

        if (cfg.ai) |ai_cfg| {
            fw.ai = try ai_mod.Context.init(allocator, ai_cfg);
            if (comptime build_options.enable_ai) {
                try fw.registry.registerComptime(.ai);
            }
        }

        if (cfg.database) |db_cfg| {
            fw.database = try database_mod.Context.init(allocator, db_cfg);
            if (comptime build_options.enable_database) {
                try fw.registry.registerComptime(.database);
            }
        }

        if (cfg.network) |net_cfg| {
            fw.network = try network_mod.Context.init(allocator, net_cfg);
            if (comptime build_options.enable_network) {
                try fw.registry.registerComptime(.network);
            }
        }

        if (cfg.observability) |obs_cfg| {
            fw.observability = try observability_mod.Context.init(allocator, obs_cfg);
            if (comptime build_options.enable_profiling) {
                try fw.registry.registerComptime(.observability);
            }
        }

        if (cfg.web) |web_cfg| {
            fw.web = try web_mod.Context.init(allocator, web_cfg);
            if (comptime build_options.enable_web) {
                try fw.registry.registerComptime(.web);
            }
        }

        fw.state = .running;
        return fw;
    }

    /// Create a framework with default configuration.
    pub fn initDefault(allocator: std.mem.Allocator) Error!Framework {
        return init(allocator, Config.defaults());
    }

    /// Create a framework with minimal configuration (no features enabled).
    pub fn initMinimal(allocator: std.mem.Allocator) Error!Framework {
        return init(allocator, Config.minimal());
    }

    /// Start building a framework configuration.
    pub fn builder(allocator: std.mem.Allocator) FrameworkBuilder {
        return FrameworkBuilder.init(allocator);
    }

    /// Shutdown and cleanup the framework.
    pub fn deinit(self: *Framework) void {
        if (self.state == .stopped) return;

        self.state = .stopping;
        self.deinitFeatures();
        self.registry.deinit();
        self.runtime.deinit();
        self.state = .stopped;
    }

    fn deinitFeatures(self: *Framework) void {
        if (self.web) |w| {
            w.deinit();
            self.web = null;
        }
        if (self.observability) |o| {
            o.deinit();
            self.observability = null;
        }
        if (self.network) |n| {
            n.deinit();
            self.network = null;
        }
        if (self.database) |d| {
            d.deinit();
            self.database = null;
        }
        if (self.ai) |a| {
            a.deinit();
            self.ai = null;
        }
        if (self.gpu) |g| {
            g.deinit();
            self.gpu = null;
        }
    }

    /// Check if the framework is running.
    pub fn isRunning(self: *const Framework) bool {
        return self.state == .running;
    }

    /// Check if a feature is enabled.
    pub fn isEnabled(self: *const Framework, feature: Feature) bool {
        return self.config.isEnabled(feature);
    }

    /// Get the current framework state.
    pub fn getState(self: *const Framework) State {
        return self.state;
    }

    /// Get GPU context (returns error if not enabled).
    pub fn getGpu(self: *Framework) Error!*gpu_mod.Context {
        return self.gpu orelse error.FeatureDisabled;
    }

    /// Get AI context (returns error if not enabled).
    pub fn getAi(self: *Framework) Error!*ai_mod.Context {
        return self.ai orelse error.FeatureDisabled;
    }

    /// Get database context (returns error if not enabled).
    pub fn getDatabase(self: *Framework) Error!*database_mod.Context {
        return self.database orelse error.FeatureDisabled;
    }

    /// Get network context (returns error if not enabled).
    pub fn getNetwork(self: *Framework) Error!*network_mod.Context {
        return self.network orelse error.FeatureDisabled;
    }

    /// Get observability context (returns error if not enabled).
    pub fn getObservability(self: *Framework) Error!*observability_mod.Context {
        return self.observability orelse error.FeatureDisabled;
    }

    /// Get web context (returns error if not enabled).
    pub fn getWeb(self: *Framework) Error!*web_mod.Context {
        return self.web orelse error.FeatureDisabled;
    }

    /// Get runtime context (always available).
    pub fn getRuntime(self: *Framework) *runtime_mod.Context {
        return self.runtime;
    }

    /// Get the feature registry for runtime feature management.
    pub fn getRegistry(self: *Framework) *Registry {
        return &self.registry;
    }

    /// Check if a feature is registered in the registry.
    pub fn isFeatureRegistered(self: *const Framework, feature: Feature) bool {
        return self.registry.isRegistered(feature);
    }

    /// List all registered features.
    pub fn listRegisteredFeatures(self: *const Framework, allocator: std.mem.Allocator) Registry.Error![]Feature {
        return self.registry.listFeatures(allocator);
    }
};

/// Fluent builder for Framework initialization.
pub const FrameworkBuilder = struct {
    allocator: std.mem.Allocator,
    config_builder: config_module.Builder,

    pub fn init(allocator: std.mem.Allocator) FrameworkBuilder {
        return .{
            .allocator = allocator,
            .config_builder = config_module.Builder.init(allocator),
        };
    }

    /// Start with default configuration.
    pub fn withDefaults(self: *FrameworkBuilder) *FrameworkBuilder {
        _ = self.config_builder.withDefaults();
        return self;
    }

    /// Enable GPU with configuration.
    pub fn withGpu(self: *FrameworkBuilder, gpu_config: config_module.GpuConfig) *FrameworkBuilder {
        _ = self.config_builder.withGpu(gpu_config);
        return self;
    }

    /// Enable GPU with defaults.
    pub fn withGpuDefaults(self: *FrameworkBuilder) *FrameworkBuilder {
        _ = self.config_builder.withGpuDefaults();
        return self;
    }

    /// Enable AI with configuration.
    pub fn withAi(self: *FrameworkBuilder, ai_config: config_module.AiConfig) *FrameworkBuilder {
        _ = self.config_builder.withAi(ai_config);
        return self;
    }

    /// Enable AI with defaults.
    pub fn withAiDefaults(self: *FrameworkBuilder) *FrameworkBuilder {
        _ = self.config_builder.withAiDefaults();
        return self;
    }

    /// Enable LLM only.
    pub fn withLlm(self: *FrameworkBuilder, llm_config: config_module.LlmConfig) *FrameworkBuilder {
        _ = self.config_builder.withLlm(llm_config);
        return self;
    }

    /// Enable database with configuration.
    pub fn withDatabase(self: *FrameworkBuilder, db_config: config_module.DatabaseConfig) *FrameworkBuilder {
        _ = self.config_builder.withDatabase(db_config);
        return self;
    }

    /// Enable database with defaults.
    pub fn withDatabaseDefaults(self: *FrameworkBuilder) *FrameworkBuilder {
        _ = self.config_builder.withDatabaseDefaults();
        return self;
    }

    /// Enable network with configuration.
    pub fn withNetwork(self: *FrameworkBuilder, net_config: config_module.NetworkConfig) *FrameworkBuilder {
        _ = self.config_builder.withNetwork(net_config);
        return self;
    }

    /// Enable network with defaults.
    pub fn withNetworkDefaults(self: *FrameworkBuilder) *FrameworkBuilder {
        _ = self.config_builder.withNetworkDefaults();
        return self;
    }

    /// Enable observability with configuration.
    pub fn withObservability(self: *FrameworkBuilder, obs_config: config_module.ObservabilityConfig) *FrameworkBuilder {
        _ = self.config_builder.withObservability(obs_config);
        return self;
    }

    /// Enable observability with defaults.
    pub fn withObservabilityDefaults(self: *FrameworkBuilder) *FrameworkBuilder {
        _ = self.config_builder.withObservabilityDefaults();
        return self;
    }

    /// Enable web with configuration.
    pub fn withWeb(self: *FrameworkBuilder, web_config: config_module.WebConfig) *FrameworkBuilder {
        _ = self.config_builder.withWeb(web_config);
        return self;
    }

    /// Enable web with defaults.
    pub fn withWebDefaults(self: *FrameworkBuilder) *FrameworkBuilder {
        _ = self.config_builder.withWebDefaults();
        return self;
    }

    /// Configure plugins.
    pub fn withPlugins(self: *FrameworkBuilder, plugin_config: config_module.PluginConfig) *FrameworkBuilder {
        _ = self.config_builder.withPlugins(plugin_config);
        return self;
    }

    /// Build and initialize the framework.
    pub fn build(self: *FrameworkBuilder) Framework.Error!Framework {
        const config = self.config_builder.build();
        return Framework.init(self.allocator, config);
    }
};

// ============================================================================
// Legacy compatibility layer
// ============================================================================

/// Legacy FrameworkOptions for backward compatibility.
/// @deprecated Use Config directly.
pub const FrameworkOptions = struct {
    enable_ai: bool = build_options.enable_ai,
    enable_gpu: bool = build_options.enable_gpu,
    enable_web: bool = build_options.enable_web,
    enable_database: bool = build_options.enable_database,
    enable_network: bool = build_options.enable_network,
    enable_profiling: bool = build_options.enable_profiling,
    disabled_features: []const Feature = &.{},
    plugin_paths: []const []const u8 = &.{},
    auto_discover_plugins: bool = false,

    /// Convert to new Config format.
    pub fn toConfig(self: FrameworkOptions) Config {
        return .{
            .gpu = if (self.enable_gpu) config_module.GpuConfig.defaults() else null,
            .ai = if (self.enable_ai) config_module.AiConfig.defaults() else null,
            .database = if (self.enable_database) config_module.DatabaseConfig.defaults() else null,
            .network = if (self.enable_network) config_module.NetworkConfig.defaults() else null,
            .observability = if (self.enable_profiling) config_module.ObservabilityConfig.defaults() else null,
            .web = if (self.enable_web) config_module.WebConfig.defaults() else null,
            .plugins = .{
                .paths = self.plugin_paths,
                .auto_discover = self.auto_discover_plugins,
            },
        };
    }
};

/// Legacy FrameworkConfiguration for backward compatibility.
/// @deprecated Use Config directly.
pub const FrameworkConfiguration = FrameworkOptions;

/// Legacy RuntimeConfig for backward compatibility.
/// @deprecated Use Config directly.
pub const RuntimeConfig = Config;

/// Convert legacy options to new config.
/// @deprecated Use Config directly.
pub fn runtimeConfigFromOptions(allocator: std.mem.Allocator, options: FrameworkOptions) !Config {
    _ = allocator;
    return options.toConfig();
}

// ============================================================================
// Tests
// ============================================================================

test "Framework.builder creates valid framework" {
    var builder_inst = Framework.builder(std.testing.allocator);
    _ = builder_inst.withDefaults();

    // Just test that builder compiles and creates config
    const config = builder_inst.config_builder.build();
    try config_module.validate(config);
}

test "FrameworkOptions.toConfig converts correctly" {
    const options = FrameworkOptions{
        .enable_gpu = true,
        .enable_ai = false,
    };
    const config = options.toConfig();

    try std.testing.expect(config.gpu != null);
    try std.testing.expect(config.ai == null);
}

test "Framework initializes registry with enabled features" {
    // Test minimal framework - no features enabled
    var fw = try Framework.initMinimal(std.testing.allocator);
    defer fw.deinit();

    // Registry should be initialized
    try std.testing.expectEqual(@as(usize, 0), fw.registry.count());
    try std.testing.expect(fw.isRunning());
}

test "Framework.getRegistry returns mutable registry" {
    var fw = try Framework.initMinimal(std.testing.allocator);
    defer fw.deinit();

    const reg = fw.getRegistry();
    try std.testing.expectEqual(@as(usize, 0), reg.count());
}

test "Framework.listRegisteredFeatures returns empty for minimal" {
    var fw = try Framework.initMinimal(std.testing.allocator);
    defer fw.deinit();

    const features = try fw.listRegisteredFeatures(std.testing.allocator);
    defer std.testing.allocator.free(features);

    try std.testing.expectEqual(@as(usize, 0), features.len);
}
