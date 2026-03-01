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
//!     .with(.gpu, abi.config.GpuConfig{ .backend = .vulkan })
//!     .withDefault(.ai)
//!     .withDefault(.database)
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
//!     const gpu_ctx = try fw.get(.gpu);
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
const framework_state = @import("framework/state.zig");
const framework_builder = @import("framework/builder.zig");
const lifecycle = @import("framework/lifecycle.zig");
const feature_catalog = @import("feature_catalog.zig");

pub const Config = config_module.Config;
pub const Feature = config_module.Feature;
pub const ConfigError = config_module.ConfigError;
pub const Registry = registry_mod.Registry;
pub const RegistryError = registry_mod.types.Error;

// Shared comptime-gated feature imports (DRY: single source of truth).
const fi = @import("framework/feature_imports.zig");
const gpu_mod = fi.gpu_mod;
const ai_mod = fi.ai_mod;
const database_mod = fi.database_mod;
const network_mod = fi.network_mod;
const observability_mod = fi.observability_mod;
const web_mod = fi.web_mod;
const cloud_mod = fi.cloud_mod;
const analytics_mod = fi.analytics_mod;
const auth_mod = fi.auth_mod;
const messaging_mod = fi.messaging_mod;
const cache_mod = fi.cache_mod;
const storage_mod = fi.storage_mod;
const search_mod = fi.search_mod;
const gateway_mod = fi.gateway_mod;
const pages_mod = fi.pages_mod;
const benchmarks_mod = fi.benchmarks_mod;
const mobile_mod = fi.mobile_mod;
const ai_core_mod = fi.ai_core_mod;
const ai_inference_mod = fi.ai_inference_mod;
const ai_training_mod = fi.ai_training_mod;
const ai_reasoning_mod = fi.ai_reasoning_mod;
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
    /// Gateway context, or null if gateway is not enabled.
    gateway: ?*gateway_mod.Context = null,
    /// Pages context, or null if pages is not enabled.
    pages: ?*pages_mod.Context = null,
    /// Benchmarks context, or null if benchmarks is not enabled.
    benchmarks: ?*benchmarks_mod.Context = null,
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
    pub const State = framework_state.State;

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
        return lifecycle.init(Framework, allocator, cfg);
    }

    /// Initialize the framework with the given configuration **and** an I/O backend.
    /// This method is used by the builder when `withIo` is supplied.
    pub fn initWithIo(allocator: std.mem.Allocator, cfg: Config, io: std.Io) Error!Framework {
        return lifecycle.initWithIo(Framework, allocator, cfg, io);
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
        return lifecycle.initDefault(Framework, allocator);
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
        return lifecycle.initMinimal(Framework, allocator);
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
    ///     .withDefault(.gpu)
    ///     .with(.ai, abi.config.AiConfig{ .llm = .{} })
    ///     .build();
    /// defer fw.deinit();
    /// ```
    pub fn builder(allocator: std.mem.Allocator) FrameworkBuilder {
        return framework_builder.init(FrameworkBuilder, allocator);
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
        lifecycle.deinit(self);
    }

    /// Shutdown with timeout. Currently synchronous (timeout reserved for
    /// future async cleanup). Returns true if clean shutdown completed.
    pub fn shutdownWithTimeout(self: *Framework, timeout_ms: u64) bool {
        return lifecycle.shutdownWithTimeout(self, timeout_ms);
    }

    fn deinitFeatures(self: *Framework) void {
        lifecycle.deinitFeatures(self);
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
        return framework_builder.init(FrameworkBuilder, allocator);
    }

    /// Start with default configuration.
    pub fn withDefaults(self: *FrameworkBuilder) *FrameworkBuilder {
        return framework_builder.withDefaults(FrameworkBuilder, self);
    }

    /// Provide a shared I/O backend for the framework.
    /// Pass the `std.Io` obtained from `IoBackend.init`.
    pub fn withIo(self: *FrameworkBuilder, io: std.Io) *FrameworkBuilder {
        return framework_builder.withIo(FrameworkBuilder, self, io);
    }

    // ── Feature builder methods (delegate to config builder) ──────────

    /// Enable a feature with explicit configuration.
    ///
    /// ## Example
    /// ```zig
    /// var fw = try Framework.builder(allocator)
    ///     .with(.gpu, abi.config.GpuConfig{ .backend = .vulkan })
    ///     .with(.database, abi.config.DatabaseConfig{ .path = "./data" })
    ///     .build();
    /// ```
    pub fn with(self: *FrameworkBuilder, comptime feature: Feature, cfg: anytype) *FrameworkBuilder {
        return framework_builder.with(FrameworkBuilder, self, feature, cfg);
    }

    /// Enable a feature with its default configuration.
    ///
    /// ## Example
    /// ```zig
    /// var fw = try Framework.builder(allocator)
    ///     .withDefault(.gpu)
    ///     .withDefault(.database)
    ///     .build();
    /// ```
    pub fn withDefault(self: *FrameworkBuilder, comptime feature: Feature) *FrameworkBuilder {
        return framework_builder.withDefault(FrameworkBuilder, self, feature);
    }

    /// Configure plugins.
    pub fn withPlugins(self: *FrameworkBuilder, plugin_config: config_module.PluginConfig) *FrameworkBuilder {
        return framework_builder.withPlugins(FrameworkBuilder, self, plugin_config);
    }

    /// Build and initialize the framework.
    /// If an I/O backend was supplied via `withIo`, it will be stored in the
    /// resulting `Framework` instance.
    pub fn build(self: *FrameworkBuilder) Framework.Error!Framework {
        return framework_builder.build(Framework, FrameworkBuilder, self);
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

test "Framework feature enum is catalog-backed" {
    try std.testing.expectEqual(feature_catalog.feature_count, @typeInfo(Feature).@"enum".fields.len);
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

test {
    std.testing.refAllDecls(@This());
}
