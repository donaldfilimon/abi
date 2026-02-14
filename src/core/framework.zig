//! Framework Orchestration Layer
//!
//! This module provides the central orchestration for the ABI framework, managing
//! the lifecycle of all feature modules, coordinating initialization and shutdown,
//! and maintaining runtime state.
//!
//! ## Overview
//!
//! The `Framework` struct is the primary entry point for using ABI. It:
//!
//! - Initializes and manages feature contexts (GPU, AI, Database, etc.)
//! - Maintains a feature registry for runtime feature management
//! - Provides typed access to enabled features
//! - Handles graceful shutdown and resource cleanup
//!
//! ## Initialization Patterns
//!
//! ### Default Initialization
//!
//! ```zig
//! const abi = @import("abi");
//!
//! var fw = try abi.Framework.initDefault(allocator);
//! defer fw.deinit();
//!
//! // All compile-time enabled features are now available
//! ```
//!
//! ### Custom Configuration
//!
//! ```zig
//! var fw = try abi.Framework.init(allocator, .{
//!     .gpu = .{ .backend = .vulkan },
//!     .ai = .{ .llm = .{ .model_path = "./model.gguf" } },
//!     .database = .{ .path = "./data" },
//! });
//! defer fw.deinit();
//! ```
//!
//! ### Builder Pattern
//!
//! ```zig
//! var fw = try abi.Framework.builder(allocator)
//!     .withGpu(.{ .backend = .vulkan })
//!     .withAiDefaults()
//!     .withDatabaseDefaults()
//!     .build();
//! defer fw.deinit();
//! ```
//!
//! ## Feature Access
//!
//! ```zig
//! // Check if a feature is enabled
//! if (fw.isEnabled(.gpu)) {
//!     // Get the feature context
//!     const gpu_ctx = try fw.getGpu();
//!     // Use GPU features...
//! }
//!
//! // Runtime context is always available
//! const runtime = fw.getRuntime();
//! ```
//!
//! ## State Management
//!
//! The framework transitions through the following states:
//! - `uninitialized`: Initial state before `init()`
//! - `initializing`: During feature initialization
//! - `running`: Normal operation state
//! - `stopping`: During shutdown
//! - `stopped`: After `deinit()` completes
//! - `failed`: If initialization fails

const std = @import("std");
const build_options = @import("build_options");
const config_module = @import("config/mod.zig");
const registry_mod = @import("registry/mod.zig");

pub const Config = config_module.Config;
pub const Feature = config_module.Feature;
pub const ConfigError = config_module.ConfigError;
pub const Registry = registry_mod.Registry;
pub const RegistryError = registry_mod.types.Error;

// Feature modules - imported based on build configuration
const gpu_mod = if (build_options.enable_gpu) @import("../features/gpu/mod.zig") else @import("../features/gpu/stub.zig");
const ai_mod = if (build_options.enable_ai) @import("../features/ai/mod.zig") else @import("../features/ai/stub.zig");
const database_mod = if (build_options.enable_database) @import("../features/database/mod.zig") else @import("../features/database/stub.zig");
const network_mod = if (build_options.enable_network) @import("../features/network/mod.zig") else @import("../features/network/stub.zig");
const observability_mod = if (build_options.enable_profiling) @import("../features/observability/mod.zig") else @import("../features/observability/stub.zig");
const web_mod = if (build_options.enable_web) @import("../features/web/mod.zig") else @import("../features/web/stub.zig");
const cloud_mod = if (build_options.enable_cloud) @import("../features/cloud/mod.zig") else @import("../features/cloud/stub.zig");
const analytics_mod = if (build_options.enable_analytics) @import("../features/analytics/mod.zig") else @import("../features/analytics/stub.zig");
const auth_mod = if (build_options.enable_auth) @import("../features/auth/mod.zig") else @import("../features/auth/stub.zig");
const messaging_mod = if (build_options.enable_messaging) @import("../features/messaging/mod.zig") else @import("../features/messaging/stub.zig");
const cache_mod = if (build_options.enable_cache) @import("../features/cache/mod.zig") else @import("../features/cache/stub.zig");
const storage_mod = if (build_options.enable_storage) @import("../features/storage/mod.zig") else @import("../features/storage/stub.zig");
const search_mod = if (build_options.enable_search) @import("../features/search/mod.zig") else @import("../features/search/stub.zig");
const mobile_mod = if (build_options.enable_mobile) @import("../features/mobile/mod.zig") else @import("../features/mobile/stub.zig");
const ai_core_mod = if (build_options.enable_ai) @import("../features/ai_core/mod.zig") else @import("../features/ai_core/stub.zig");
const ai_inference_mod = if (build_options.enable_llm) @import("../features/ai_inference/mod.zig") else @import("../features/ai_inference/stub.zig");
const ai_training_mod = if (build_options.enable_training) @import("../features/ai_training/mod.zig") else @import("../features/ai_training/stub.zig");
const ai_reasoning_mod = if (build_options.enable_reasoning) @import("../features/ai_reasoning/mod.zig") else @import("../features/ai_reasoning/stub.zig");
const ha_mod = @import("../services/ha/mod.zig");
const runtime_mod = @import("../services/runtime/mod.zig");

/// Framework orchestration handle.
///
/// The Framework struct is the central coordinator for the ABI framework. It manages
/// the lifecycle of all enabled feature modules, provides access to their contexts,
/// and maintains the framework's runtime state.
///
/// ## Thread Safety
///
/// The Framework itself is not thread-safe. If you need to access the framework from
/// multiple threads, you should use external synchronization or ensure each thread
/// has its own Framework instance.
///
/// ## Memory Management
///
/// The Framework allocates memory for feature contexts during initialization. All
/// allocated memory is released when `deinit()` is called. The caller must ensure
/// the provided allocator remains valid for the lifetime of the Framework.
///
/// ## Example
///
/// ```zig
/// var fw = try Framework.init(allocator, Config.defaults());
/// defer fw.deinit();
///
/// // Check state
/// if (fw.isRunning()) {
///     // Access features
///     if (fw.gpu) |gpu_ctx| {
///         // Use GPU...
///     }
/// }
/// ```
pub const Framework = struct {
    /// Memory allocator used for all framework allocations.
    allocator: std.mem.Allocator,

    /// Optional I/O backend shared across the framework.
    ///
    /// Sub-systems that need file or network access can retrieve it via `self.io`.
    /// It is set during initialization by the builder (see `FrameworkBuilder.withIo`).
    io: ?std.Io = null,

    /// The configuration used to initialize this framework instance.
    config: Config,

    /// Current lifecycle state of the framework.
    state: State,

    /// Feature registry for runtime feature management.
    registry: Registry,

    // Feature handles (null if disabled)
    /// GPU context, or null if GPU is not enabled.
    gpu: ?*gpu_mod.Context = null,
    /// AI context, or null if AI is not enabled.
    ai: ?*ai_mod.Context = null,
    /// Database context, or null if database is not enabled.
    database: ?*database_mod.Context = null,
    /// Network context, or null if network is not enabled.
    network: ?*network_mod.Context = null,
    /// Observability context, or null if observability is not enabled.
    observability: ?*observability_mod.Context = null,
    /// Web context, or null if web is not enabled.
    web: ?*web_mod.Context = null,
    /// Cloud context, or null if cloud/web is not enabled.
    cloud: ?*cloud_mod.Context = null,
    /// Analytics context, or null if analytics is not enabled.
    analytics: ?*analytics_mod.Context = null,
    /// Auth context, or null if auth is not enabled.
    auth: ?*auth_mod.Context = null,
    /// Messaging context, or null if messaging is not enabled.
    messaging: ?*messaging_mod.Context = null,
    /// Cache context, or null if cache is not enabled.
    cache: ?*cache_mod.Context = null,
    /// Storage context, or null if storage is not enabled.
    storage: ?*storage_mod.Context = null,
    /// Search context, or null if search is not enabled.
    search: ?*search_mod.Context = null,
    /// Mobile context, or null if mobile is not enabled.
    mobile: ?*mobile_mod.Context = null,
    /// AI Core context (agents, tools, prompts), or null if not enabled.
    ai_core: ?*ai_core_mod.Context = null,
    /// AI Inference context (LLM, embeddings, vision), or null if not enabled.
    ai_inference: ?*ai_inference_mod.Context = null,
    /// AI Training context (pipelines, federated), or null if not enabled.
    ai_training: ?*ai_training_mod.Context = null,
    /// AI Reasoning context (Abbey, RAG, eval), or null if not enabled.
    ai_reasoning: ?*ai_reasoning_mod.Context = null,
    /// High availability manager, or null if not initialized.
    ha: ?ha_mod.HaManager = null,
    /// Runtime context (always available).
    runtime: *runtime_mod.Context,

    /// Framework lifecycle states.
    pub const State = enum {
        uninitialized,
        initializing,
        running,
        stopping,
        stopped,
        failed,
    };

    /// Composable framework error set.
    /// See `core/errors.zig` for the full hierarchy.
    pub const Error = @import("errors.zig").FrameworkError;

    /// Initialize the framework with the given configuration.
    ///
    /// This is the primary initialization method for the Framework. It validates the
    /// configuration, initializes all enabled feature modules, and transitions the
    /// framework to the `running` state.
    ///
    /// ## Parameters
    ///
    /// - `allocator`: Memory allocator for framework resources. Must remain valid for
    ///   the lifetime of the Framework.
    /// - `cfg`: Configuration specifying which features to enable and their settings.
    ///
    /// ## Returns
    ///
    /// A fully initialized Framework instance in the `running` state.
    ///
    /// ## Errors
    ///
    /// - `ConfigError.FeatureDisabled`: A feature is enabled in config but disabled at compile time
    /// - `error.OutOfMemory`: Memory allocation failed
    /// - `error.FeatureInitFailed`: A feature module failed to initialize
    ///
    /// ## Example
    ///
    /// ```zig
    /// var fw = try Framework.init(allocator, .{
    ///     .gpu = .{ .backend = .vulkan },
    ///     .database = .{ .path = "./data" },
    /// });
    /// defer fw.deinit();
    /// ```
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

        if (cfg.cloud) |core_cloud| {
            // Map core/config CloudConfig fields to features/cloud runtime CloudConfig.
            const runtime_cloud = cloud_mod.CloudConfig{
                .memory_mb = core_cloud.memory_mb,
                .timeout_seconds = core_cloud.timeout_seconds,
                .tracing_enabled = core_cloud.tracing_enabled,
                .logging_enabled = core_cloud.logging_enabled,
                .log_level = @enumFromInt(@intFromEnum(core_cloud.log_level)),
            };
            fw.cloud = try cloud_mod.Context.init(allocator, runtime_cloud);
            if (comptime build_options.enable_cloud) {
                try fw.registry.registerComptime(.cloud);
            }
        }

        if (cfg.analytics) |analytics_cfg| {
            fw.analytics = try analytics_mod.Context.init(allocator, .{
                .buffer_capacity = analytics_cfg.buffer_capacity,
                .enable_timestamps = analytics_cfg.enable_timestamps,
                .app_id = analytics_cfg.app_id,
                .flush_interval_ms = analytics_cfg.flush_interval_ms,
            });
            if (comptime build_options.enable_analytics) {
                try fw.registry.registerComptime(.analytics);
            }
        }

        if (cfg.auth) |auth_cfg| {
            fw.auth = try auth_mod.Context.init(allocator, auth_cfg);
            if (comptime build_options.enable_auth) {
                try fw.registry.registerComptime(.auth);
            }
        }

        if (cfg.messaging) |msg_cfg| {
            fw.messaging = try messaging_mod.Context.init(allocator, msg_cfg);
            if (comptime build_options.enable_messaging) {
                try fw.registry.registerComptime(.messaging);
            }
        }

        if (cfg.cache) |cache_cfg| {
            fw.cache = try cache_mod.Context.init(allocator, cache_cfg);
            if (comptime build_options.enable_cache) {
                try fw.registry.registerComptime(.cache);
            }
        }

        if (cfg.storage) |storage_cfg| {
            fw.storage = try storage_mod.Context.init(allocator, storage_cfg);
            if (comptime build_options.enable_storage) {
                try fw.registry.registerComptime(.storage);
            }
        }

        if (cfg.search) |search_cfg| {
            fw.search = try search_mod.Context.init(allocator, search_cfg);
            if (comptime build_options.enable_search) {
                try fw.registry.registerComptime(.search);
            }
        }

        if (cfg.mobile) |mobile_cfg| {
            fw.mobile = try mobile_mod.Context.init(allocator, mobile_cfg);
            if (comptime build_options.enable_mobile) {
                try fw.registry.registerComptime(.mobile);
            }
        }

        // Initialize split AI modules (use shared AI config)
        if (cfg.ai) |ai_cfg| {
            if (comptime build_options.enable_ai) {
                fw.ai_core = ai_core_mod.Context.init(
                    allocator,
                    ai_cfg,
                ) catch null;
            }
            if (comptime build_options.enable_llm) {
                fw.ai_inference = ai_inference_mod.Context.init(
                    allocator,
                    ai_cfg,
                ) catch null;
            }
            if (comptime build_options.enable_training) {
                fw.ai_training = ai_training_mod.Context.init(
                    allocator,
                    ai_cfg,
                ) catch null;
            }
            if (comptime build_options.enable_reasoning) {
                fw.ai_reasoning = ai_reasoning_mod.Context.init(
                    allocator,
                    ai_cfg,
                ) catch null;
            }
        }

        // Initialize high availability if enabled (defaulting to primary)
        fw.ha = ha_mod.HaManager.init(allocator, .{});

        fw.state = .running;
        return fw;
    }

    /// Initialize the framework with the given configuration **and** an I/O backend.
    /// This method is used by the builder when `withIo` is supplied.
    pub fn initWithIo(allocator: std.mem.Allocator, cfg: Config, io: std.Io) Error!Framework {
        var fw = try Framework.init(allocator, cfg);
        fw.io = io;
        return fw;
    }

    /// Create a framework with default configuration.
    ///
    /// This is a convenience method that creates a framework with all compile-time
    /// enabled features also enabled at runtime with their default settings.
    ///
    /// ## Parameters
    ///
    /// - `allocator`: Memory allocator for framework resources
    ///
    /// ## Returns
    ///
    /// A Framework instance with default configuration.
    ///
    /// ## Example
    ///
    /// ```zig
    /// var fw = try Framework.initDefault(allocator);
    /// defer fw.deinit();
    /// ```
    pub fn initDefault(allocator: std.mem.Allocator) Error!Framework {
        return init(allocator, Config.defaults());
    }

    /// Create a framework with minimal configuration (no features enabled).
    ///
    /// This creates a framework with no optional features enabled. Only the
    /// runtime context is initialized. Useful for testing or when you want
    /// to explicitly enable specific features.
    ///
    /// ## Parameters
    ///
    /// - `allocator`: Memory allocator for framework resources
    ///
    /// ## Returns
    ///
    /// A Framework instance with minimal configuration.
    ///
    /// ## Example
    ///
    /// ```zig
    /// var fw = try Framework.initMinimal(allocator);
    /// defer fw.deinit();
    ///
    /// // Only runtime is available, no features enabled
    /// try std.testing.expect(fw.gpu == null);
    /// try std.testing.expect(fw.ai == null);
    /// ```
    pub fn initMinimal(allocator: std.mem.Allocator) Error!Framework {
        return init(allocator, Config.minimal());
    }

    /// Start building a framework configuration.
    ///
    /// Returns a FrameworkBuilder that provides a fluent API for configuring
    /// and initializing the framework.
    ///
    /// ## Parameters
    ///
    /// - `allocator`: Memory allocator for framework resources
    ///
    /// ## Returns
    ///
    /// A FrameworkBuilder instance for configuring the framework.
    ///
    /// ## Example
    ///
    /// ```zig
    /// var fw = try Framework.builder(allocator)
    ///     .withGpuDefaults()
    ///     .withAi(.{ .llm = .{} })
    ///     .build();
    /// defer fw.deinit();
    /// ```
    pub fn builder(allocator: std.mem.Allocator) FrameworkBuilder {
        return FrameworkBuilder.init(allocator);
    }

    /// Shutdown and cleanup the framework.
    ///
    /// This method transitions the framework to the `stopping` state, deinitializes
    /// all feature contexts in reverse order of initialization, cleans up the registry,
    /// and finally transitions to `stopped`.
    ///
    /// After calling `deinit()`, the framework instance should not be used. Any
    /// pointers to feature contexts become invalid.
    ///
    /// This method is idempotent - calling it multiple times is safe.
    ///
    /// ## Example
    ///
    /// ```zig
    /// var fw = try Framework.initDefault(allocator);
    /// // ... use framework ...
    /// fw.deinit();  // Clean up all resources
    /// ```
    pub fn deinit(self: *Framework) void {
        if (self.state == .stopped) return;

        self.state = .stopping;
        self.deinitFeatures();
        self.registry.deinit();
        self.runtime.deinit();
        self.state = .stopped;
    }

    /// Shutdown with timeout. Currently synchronous (timeout reserved for
    /// future async cleanup). Returns true if clean shutdown completed.
    pub fn shutdownWithTimeout(self: *Framework, _: u64) bool {
        self.deinit();
        return self.state == .stopped;
    }

    fn deinitOptionalContext(comptime Context: type, slot: *?*Context) void {
        if (slot.*) |ctx| {
            ctx.deinit();
            slot.* = null;
        }
    }

    fn deinitFeatures(self: *Framework) void {
        // Deinitialize in reverse order of initialization.
        // Split AI modules first (initialized last)
        deinitOptionalContext(ai_reasoning_mod.Context, &self.ai_reasoning);
        deinitOptionalContext(ai_training_mod.Context, &self.ai_training);
        deinitOptionalContext(ai_inference_mod.Context, &self.ai_inference);
        deinitOptionalContext(ai_core_mod.Context, &self.ai_core);
        // Then standard feature modules
        deinitOptionalContext(mobile_mod.Context, &self.mobile);
        deinitOptionalContext(search_mod.Context, &self.search);
        deinitOptionalContext(storage_mod.Context, &self.storage);
        deinitOptionalContext(cache_mod.Context, &self.cache);
        deinitOptionalContext(messaging_mod.Context, &self.messaging);
        deinitOptionalContext(auth_mod.Context, &self.auth);
        deinitOptionalContext(analytics_mod.Context, &self.analytics);
        deinitOptionalContext(cloud_mod.Context, &self.cloud);
        deinitOptionalContext(web_mod.Context, &self.web);
        deinitOptionalContext(observability_mod.Context, &self.observability);
        deinitOptionalContext(network_mod.Context, &self.network);
        deinitOptionalContext(database_mod.Context, &self.database);
        deinitOptionalContext(ai_mod.Context, &self.ai);
        deinitOptionalContext(gpu_mod.Context, &self.gpu);
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

    fn requireFeature(comptime Context: type, value: ?*Context) Error!*Context {
        return value orelse error.FeatureDisabled;
    }

    /// Get GPU context (returns error if not enabled).
    pub fn getGpu(self: *Framework) Error!*gpu_mod.Context {
        return requireFeature(gpu_mod.Context, self.gpu);
    }

    /// Get AI context (returns error if not enabled).
    pub fn getAi(self: *Framework) Error!*ai_mod.Context {
        return requireFeature(ai_mod.Context, self.ai);
    }

    /// Get database context (returns error if not enabled).
    pub fn getDatabase(self: *Framework) Error!*database_mod.Context {
        return requireFeature(database_mod.Context, self.database);
    }

    /// Get network context (returns error if not enabled).
    pub fn getNetwork(self: *Framework) Error!*network_mod.Context {
        return requireFeature(network_mod.Context, self.network);
    }

    /// Get observability context (returns error if not enabled).
    pub fn getObservability(self: *Framework) Error!*observability_mod.Context {
        return requireFeature(observability_mod.Context, self.observability);
    }

    /// Get web context (returns error if not enabled).
    pub fn getWeb(self: *Framework) Error!*web_mod.Context {
        return requireFeature(web_mod.Context, self.web);
    }

    /// Get cloud context (returns error if not enabled).
    pub fn getCloud(self: *Framework) Error!*cloud_mod.Context {
        return requireFeature(cloud_mod.Context, self.cloud);
    }

    /// Get analytics context (returns error if not enabled).
    pub fn getAnalytics(self: *Framework) Error!*analytics_mod.Context {
        return requireFeature(analytics_mod.Context, self.analytics);
    }

    /// Get auth context (returns error if not enabled).
    pub fn getAuth(self: *Framework) Error!*auth_mod.Context {
        return requireFeature(auth_mod.Context, self.auth);
    }

    /// Get messaging context (returns error if not enabled).
    pub fn getMessaging(self: *Framework) Error!*messaging_mod.Context {
        return requireFeature(messaging_mod.Context, self.messaging);
    }

    /// Get cache context (returns error if not enabled).
    pub fn getCache(self: *Framework) Error!*cache_mod.Context {
        return requireFeature(cache_mod.Context, self.cache);
    }

    /// Get storage context (returns error if not enabled).
    pub fn getStorage(self: *Framework) Error!*storage_mod.Context {
        return requireFeature(storage_mod.Context, self.storage);
    }

    /// Get search context (returns error if not enabled).
    pub fn getSearch(self: *Framework) Error!*search_mod.Context {
        return requireFeature(search_mod.Context, self.search);
    }

    /// Get mobile context (returns error if not enabled).
    pub fn getMobile(self: *Framework) Error!*mobile_mod.Context {
        return requireFeature(mobile_mod.Context, self.mobile);
    }

    /// Get AI core context (agents, tools, prompts).
    pub fn getAiCore(self: *Framework) Error!*ai_core_mod.Context {
        return requireFeature(ai_core_mod.Context, self.ai_core);
    }

    /// Get AI inference context (LLM, embeddings, vision).
    pub fn getAiInference(self: *Framework) Error!*ai_inference_mod.Context {
        return requireFeature(ai_inference_mod.Context, self.ai_inference);
    }

    /// Get AI training context (pipelines, federated).
    pub fn getAiTraining(self: *Framework) Error!*ai_training_mod.Context {
        return requireFeature(ai_training_mod.Context, self.ai_training);
    }

    /// Get AI reasoning context (Abbey, RAG, eval).
    pub fn getAiReasoning(self: *Framework) Error!*ai_reasoning_mod.Context {
        return requireFeature(ai_reasoning_mod.Context, self.ai_reasoning);
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
    pub fn listRegisteredFeatures(self: *const Framework, allocator: std.mem.Allocator) RegistryError![]Feature {
        return self.registry.listFeatures(allocator);
    }
};

/// Fluent builder for Framework initialization.
pub const FrameworkBuilder = struct {
    allocator: std.mem.Allocator,
    config_builder: config_module.Builder,
    // Optional shared I/O backend (set via `withIo`).  Sub‑systems that need
    // file or network access can retrieve it through `framework.io`.
    io: ?std.Io = null,

    pub fn init(allocator: std.mem.Allocator) FrameworkBuilder {
        return .{
            .allocator = allocator,
            .config_builder = config_module.Builder.init(allocator),
            .io = null,
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

    /// Provide a shared I/O backend for the framework.
    /// Pass the `std.Io` obtained from `IoBackend.init`.
    pub fn withIo(self: *FrameworkBuilder, io: std.Io) *FrameworkBuilder {
        self.io = io;
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

    /// Enable analytics with configuration.
    pub fn withAnalytics(self: *FrameworkBuilder, analytics_cfg: config_module.AnalyticsConfig) *FrameworkBuilder {
        _ = self.config_builder.withAnalytics(analytics_cfg);
        return self;
    }

    /// Enable analytics with defaults.
    pub fn withAnalyticsDefaults(self: *FrameworkBuilder) *FrameworkBuilder {
        _ = self.config_builder.withAnalyticsDefaults();
        return self;
    }

    /// Enable cloud with configuration.
    pub fn withCloud(self: *FrameworkBuilder, cloud_config: config_module.CloudConfig) *FrameworkBuilder {
        _ = self.config_builder.withCloud(cloud_config);
        return self;
    }

    /// Enable cloud with defaults.
    pub fn withCloudDefaults(self: *FrameworkBuilder) *FrameworkBuilder {
        _ = self.config_builder.withCloudDefaults();
        return self;
    }

    /// Enable auth with configuration.
    pub fn withAuth(self: *FrameworkBuilder, auth_config: config_module.AuthConfig) *FrameworkBuilder {
        _ = self.config_builder.withAuth(auth_config);
        return self;
    }

    /// Enable auth with defaults.
    pub fn withAuthDefaults(self: *FrameworkBuilder) *FrameworkBuilder {
        _ = self.config_builder.withAuthDefaults();
        return self;
    }

    /// Enable messaging with configuration.
    pub fn withMessaging(self: *FrameworkBuilder, msg_config: config_module.MessagingConfig) *FrameworkBuilder {
        _ = self.config_builder.withMessaging(msg_config);
        return self;
    }

    /// Enable messaging with defaults.
    pub fn withMessagingDefaults(self: *FrameworkBuilder) *FrameworkBuilder {
        _ = self.config_builder.withMessagingDefaults();
        return self;
    }

    /// Enable cache with configuration.
    pub fn withCache(self: *FrameworkBuilder, cache_config: config_module.CacheConfig) *FrameworkBuilder {
        _ = self.config_builder.withCache(cache_config);
        return self;
    }

    /// Enable cache with defaults.
    pub fn withCacheDefaults(self: *FrameworkBuilder) *FrameworkBuilder {
        _ = self.config_builder.withCacheDefaults();
        return self;
    }

    /// Enable storage with configuration.
    pub fn withStorage(self: *FrameworkBuilder, storage_config: config_module.StorageConfig) *FrameworkBuilder {
        _ = self.config_builder.withStorage(storage_config);
        return self;
    }

    /// Enable storage with defaults.
    pub fn withStorageDefaults(self: *FrameworkBuilder) *FrameworkBuilder {
        _ = self.config_builder.withStorageDefaults();
        return self;
    }

    /// Enable search with configuration.
    pub fn withSearch(self: *FrameworkBuilder, search_config: config_module.SearchConfig) *FrameworkBuilder {
        _ = self.config_builder.withSearch(search_config);
        return self;
    }

    /// Enable search with defaults.
    pub fn withSearchDefaults(self: *FrameworkBuilder) *FrameworkBuilder {
        _ = self.config_builder.withSearchDefaults();
        return self;
    }

    /// Enable mobile with configuration.
    pub fn withMobile(self: *FrameworkBuilder, mobile_cfg: config_module.MobileConfig) *FrameworkBuilder {
        _ = self.config_builder.withMobile(mobile_cfg);
        return self;
    }

    /// Enable mobile with defaults.
    pub fn withMobileDefaults(self: *FrameworkBuilder) *FrameworkBuilder {
        _ = self.config_builder.withMobileDefaults();
        return self;
    }

    /// Configure plugins.
    pub fn withPlugins(self: *FrameworkBuilder, plugin_config: config_module.PluginConfig) *FrameworkBuilder {
        _ = self.config_builder.withPlugins(plugin_config);
        return self;
    }

    /// Build and initialize the framework.
    /// If an I/O backend was supplied via `withIo`, it will be stored in the
    /// resulting `Framework` instance.
    pub fn build(self: *FrameworkBuilder) Framework.Error!Framework {
        const config = self.config_builder.build();
        if (self.io) |io| {
            return Framework.initWithIo(self.allocator, config, io);
        } else {
            return Framework.init(self.allocator, config);
        }
    }
};

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
