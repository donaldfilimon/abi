//! Unified Configuration System
//!
//! This module provides a single source of truth for all ABI framework configuration.
//! It supports both struct literal initialization and a fluent builder pattern API.
//!
//! ## Overview
//!
//! The configuration system is designed around the principle that each feature can be
//! independently enabled or disabled. A `null` value for any feature config means that
//! feature is disabled; a non-null value enables it with the specified settings.
//!
//! ## Configuration Styles
//!
//! ### 1. Minimal (Everything Auto-detected)
//!
//! ```zig
//! var fw = try abi.initDefault(allocator);
//! defer fw.deinit();
//! ```
//!
//! ### 2. Struct Literal Style
//!
//! ```zig
//! var fw = try abi.init(allocator, .{
//!     .gpu = .{ .backend = .vulkan },
//!     .ai = .{ .llm = .{} },
//!     .database = null,  // Explicitly disabled
//! });
//! defer fw.deinit();
//! ```
//!
//! ### 3. Builder Pattern Style
//!
//! ```zig
//! var builder = abi.config.Builder.init(allocator);
//! const config = builder
//!     .withGpu(.{ .backend = .cuda })
//!     .withAi(.{ .llm = .{ .model_path = "./models/llama.gguf" } })
//!     .withDatabaseDefaults()
//!     .build();
//!
//! var fw = try abi.Framework.init(allocator, config);
//! defer fw.deinit();
//! ```
//!
//! ## Feature Configuration Types
//!
//! - `GpuConfig` - GPU backend selection and memory limits
//! - `AiConfig` - AI sub-features (LLM, embeddings, agents, training, personas)
//! - `DatabaseConfig` - Vector database path and index settings
//! - `NetworkConfig` - Distributed compute and Raft settings
//! - `ObservabilityConfig` - Metrics, tracing, and profiling
//! - `WebConfig` - HTTP server and client settings
//! - `PluginConfig` - Plugin discovery and loading
//!
//! ## Compile-Time vs Runtime Configuration
//!
//! Some features can be disabled at compile time using build flags (e.g., `-Denable-gpu=false`).
//! If a feature is disabled at compile time, attempting to enable it at runtime via
//! configuration will result in a `ConfigError.FeatureDisabled` error.

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
const cloud_mod = @import("config/cloud.zig");
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

// Cloud configuration types
pub const CloudConfig = cloud_mod.CloudConfig;
pub const CloudProvider = cloud_mod.CloudProvider;

// Plugin configuration types
pub const PluginConfig = plugins_mod.PluginConfig;

/// Unified configuration for the ABI framework.
///
/// This struct holds configuration for all framework features. Each field being non-null
/// enables that feature with the specified settings; a null field means the feature is
/// disabled.
///
/// ## Thread Safety
///
/// Configuration is immutable once created. It can be safely shared across threads
/// after initialization.
///
/// ## Example
///
/// ```zig
/// // Create with struct literal
/// const config = Config{
///     .gpu = .{ .backend = .vulkan },
///     .ai = .{
///         .llm = .{ .model_path = "./model.gguf" },
///         .embeddings = .{ .dimension = 768 },
///     },
///     .database = .{ .path = "./vectors.db" },
/// };
///
/// // Or use defaults
/// const config = Config.defaults();
/// ```
pub const Config = struct {
    /// GPU acceleration settings.
    ///
    /// Set to a `GpuConfig` to enable GPU features, or leave as `null` to disable.
    /// When enabled, the framework will initialize the specified GPU backend.
    gpu: ?GpuConfig = null,

    /// AI settings with independent sub-features.
    ///
    /// The AI module has multiple sub-features (LLM, embeddings, agents, training, personas)
    /// that can be independently enabled within this config.
    ai: ?AiConfig = null,

    /// Vector database settings.
    ///
    /// Enables the WDBX vector database with HNSW/IVF-PQ indexing for similarity search.
    database: ?DatabaseConfig = null,

    /// Distributed network settings.
    ///
    /// Enables distributed compute capabilities including Raft consensus and task distribution.
    network: ?NetworkConfig = null,

    /// Observability and monitoring settings.
    ///
    /// Enables metrics collection, distributed tracing, and performance profiling.
    observability: ?ObservabilityConfig = null,

    /// Web/HTTP utilities settings.
    ///
    /// Enables HTTP server/client utilities and web-related functionality.
    web: ?WebConfig = null,

    /// Cloud function adapters settings.
    ///
    /// Enables cloud function adapters for AWS Lambda, GCP Functions, and Azure Functions.
    cloud: ?CloudConfig = null,

    /// Plugin configuration.
    ///
    /// Controls plugin discovery and loading. Unlike other features, this is always
    /// available (not optional) but can be configured to enable/disable plugin loading.
    plugins: PluginConfig = .{},

    /// Returns a config with all compile-time enabled features activated with defaults.
    ///
    /// This creates a configuration where every feature that was enabled at compile time
    /// is also enabled at runtime with default settings. Features disabled at compile time
    /// will have `null` configurations.
    ///
    /// ## Example
    ///
    /// ```zig
    /// const config = Config.defaults();
    /// // All compile-time enabled features are now runtime enabled
    /// ```
    pub fn defaults() Config {
        return .{
            .gpu = if (build_options.enable_gpu) GpuConfig.defaults() else null,
            .ai = if (build_options.enable_ai) AiConfig.defaults() else null,
            .database = if (build_options.enable_database) DatabaseConfig.defaults() else null,
            .network = if (build_options.enable_network) NetworkConfig.defaults() else null,
            .observability = if (build_options.enable_profiling) ObservabilityConfig.defaults() else null,
            .web = if (build_options.enable_web) WebConfig.defaults() else null,
            .cloud = if (build_options.enable_web) CloudConfig.defaults() else null,
            .plugins = .{},
        };
    }

    /// Returns a minimal config with no features enabled.
    ///
    /// This creates a configuration where all optional features are disabled (set to `null`).
    /// Useful for testing or when you want to enable features one at a time.
    ///
    /// ## Example
    ///
    /// ```zig
    /// var config = Config.minimal();
    /// config.gpu = .{ .backend = .vulkan };  // Enable only GPU
    /// ```
    pub fn minimal() Config {
        return .{};
    }

    /// Check if a feature is enabled in this configuration.
    ///
    /// ## Parameters
    ///
    /// - `feature`: The feature to check
    ///
    /// ## Returns
    ///
    /// `true` if the feature is enabled, `false` otherwise.
    ///
    /// ## Example
    ///
    /// ```zig
    /// const config = Config{
    ///     .gpu = .{ .backend = .vulkan },
    ///     .ai = null,
    /// };
    /// std.debug.print("GPU: {}, AI: {}\n", .{
    ///     config.isEnabled(.gpu),   // true
    ///     config.isEnabled(.ai),    // false
    /// });
    /// ```
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
            .cloud => self.cloud != null,
        };
    }

    /// Get a list of all enabled features.
    ///
    /// Allocates and returns a slice containing all features that are currently enabled
    /// in this configuration. The caller owns the returned slice and must free it.
    ///
    /// ## Parameters
    ///
    /// - `allocator`: Allocator for the returned slice
    ///
    /// ## Returns
    ///
    /// A slice of `Feature` values representing all enabled features.
    ///
    /// ## Example
    ///
    /// ```zig
    /// const config = Config.defaults();
    /// const features = try config.enabledFeatures(allocator);
    /// defer allocator.free(features);
    ///
    /// for (features) |feature| {
    ///     std.debug.print("Enabled: {s}\n", .{feature.name()});
    /// }
    /// ```
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
///
/// This enum represents all available features in the ABI framework. Features can be
/// checked at both compile time (via build options) and runtime (via configuration).
///
/// ## Hierarchy
///
/// Some features are sub-features of others:
/// - `ai` is the parent of `llm`, `embeddings`, `agents`, `training`, `personas`
///
/// Enabling a parent feature doesn't automatically enable sub-features; each must be
/// configured independently.
///
/// ## Example
///
/// ```zig
/// const feature = Feature.gpu;
/// std.debug.print("Feature: {s}\n", .{feature.name()});
/// std.debug.print("Description: {s}\n", .{feature.description()});
/// std.debug.print("Compile-time enabled: {}\n", .{feature.isCompileTimeEnabled()});
/// ```
pub const Feature = enum {
    /// GPU acceleration and compute
    gpu,
    /// AI core functionality (parent of llm, embeddings, agents, training, personas)
    ai,
    /// Local LLM inference
    llm,
    /// Vector embeddings generation
    embeddings,
    /// AI agent runtime
    agents,
    /// Model training pipelines
    training,
    /// Multi-persona AI assistant
    personas,
    /// Vector database (WDBX)
    database,
    /// Distributed compute network
    network,
    /// Metrics, tracing, and profiling
    observability,
    /// Web/HTTP utilities
    web,
    /// Cloud function adapters
    cloud,

    /// Get the feature name as a string.
    ///
    /// ## Returns
    ///
    /// The lowercase name of the feature (e.g., "gpu", "ai", "database").
    pub fn name(self: Feature) []const u8 {
        return @tagName(self);
    }

    /// Get a human-readable description of the feature.
    ///
    /// ## Returns
    ///
    /// A brief description of what the feature provides.
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
            .cloud => "Cloud function adapters",
        };
    }

    /// Check if this feature is available at compile time.
    ///
    /// This checks the build options to determine if the feature was enabled during
    /// compilation. Even if enabled at compile time, the feature may still be disabled
    /// at runtime via configuration.
    ///
    /// ## Returns
    ///
    /// `true` if the feature was enabled at compile time, `false` otherwise.
    ///
    /// ## Example
    ///
    /// ```zig
    /// if (Feature.gpu.isCompileTimeEnabled()) {
    ///     // GPU code is compiled in, safe to configure
    ///     config.gpu = .{ .backend = .vulkan };
    /// }
    /// ```
    pub fn isCompileTimeEnabled(self: Feature) bool {
        return switch (self) {
            .gpu => build_options.enable_gpu,
            .ai, .llm, .embeddings, .agents, .training, .personas => build_options.enable_ai,
            .database => build_options.enable_database,
            .network => build_options.enable_network,
            .observability => build_options.enable_profiling,
            .web, .cloud => build_options.enable_web,
        };
    }
};

// ============================================================================
// Builder API
// ============================================================================

/// Fluent builder for constructing Config instances.
///
/// The Builder provides a chainable API for constructing framework configuration.
/// Each method returns a pointer to the builder, allowing method chaining.
///
/// ## Example
///
/// ```zig
/// var builder = Builder.init(allocator);
/// const config = builder
///     .withGpu(.{ .backend = .cuda })
///     .withAi(.{ .llm = .{ .model_path = "./model.gguf" } })
///     .withDatabaseDefaults()
///     .build();
///
/// var fw = try Framework.init(allocator, config);
/// defer fw.deinit();
/// ```
///
/// ## Note
///
/// The builder stores a copy of the allocator but doesn't allocate memory itself.
/// All configuration is stored inline in the Config struct.
pub const Builder = struct {
    /// Allocator provided during initialization (stored for potential future use).
    allocator: std.mem.Allocator,
    /// The configuration being built.
    config: Config,

    /// Initialize a new builder with an empty configuration.
    ///
    /// ## Parameters
    ///
    /// - `allocator`: Memory allocator (stored for potential future use)
    ///
    /// ## Returns
    ///
    /// A new Builder instance with all features disabled.
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

    /// Enable cloud with specified configuration.
    pub fn withCloud(self: *Builder, cloud_config: CloudConfig) *Builder {
        self.config.cloud = cloud_config;
        return self;
    }

    /// Enable cloud with default configuration.
    pub fn withCloudDefaults(self: *Builder) *Builder {
        self.config.cloud = CloudConfig.defaults();
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

    // Check cloud config (depends on web being enabled)
    if (config.cloud != null and !build_options.enable_web) {
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
