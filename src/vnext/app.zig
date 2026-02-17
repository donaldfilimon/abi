//! vNext application handle.
//!
//! Wraps the existing Framework in a forward API that supports staged
//! compatibility while vNext surfaces are introduced.
//!
//! ## New API (vNext)
//!
//! - `App.start(alloc, cfg)` — create and start an App
//! - `app.stop()` / `app.stopWithOptions(.{})` — shutdown
//! - `app.feature(.gpu)` — typed optional context access
//! - `app.has(.gpu)` — boolean capability check
//! - `app.state()` — lifecycle state query
//! - `app.runtime()` / `app.registry()` — infrastructure access
//! - `App.configure(alloc)` — builder pattern entry point

const std = @import("std");
const core_config = @import("../core/config/mod.zig");
const framework_mod = @import("../core/framework.zig");
const capability_mod = @import("capability.zig");
const config_mod = @import("config.zig");
const runtime_mod = @import("../services/runtime/mod.zig");

const Capability = capability_mod.Capability;
const Framework = framework_mod.Framework;

// ============================================================================
// Comptime Helpers
// ============================================================================

/// Maps a Capability to its corresponding Framework struct field name.
fn frameworkFieldName(comptime cap: Capability) []const u8 {
    return switch (cap) {
        .gpu => "gpu",
        .ai => "ai",
        .database => "database",
        .network => "network",
        .observability => "observability",
        .web => "web",
        .cloud => "cloud",
        .analytics => "analytics",
        .auth => "auth",
        .messaging => "messaging",
        .cache => "cache",
        .storage => "storage",
        .search => "search",
        .mobile => "mobile",
        .gateway => "gateway",
        .pages => "pages",
        .benchmarks => "benchmarks",
        // Sub-features map to their parent module's context
        .llm, .embeddings => "ai_inference",
        .agents, .personas => "ai_core",
        .training => "ai_training",
        .reasoning => "ai_reasoning",
    };
}

/// Returns the type of the Framework field for a given Capability.
/// E.g., FeatureFieldType(.gpu) = ?*gpu_mod.Context
fn FeatureFieldType(comptime cap: Capability) type {
    const field_name = frameworkFieldName(cap);
    for (@typeInfo(Framework).@"struct".fields) |field| {
        if (std.mem.eql(u8, field.name, field_name)) {
            return field.type;
        }
    }
    @compileError("no Framework field for capability: " ++ @tagName(cap));
}

/// Returns the config type for a given Capability.
fn ConfigType(comptime cap: Capability) type {
    return switch (cap) {
        .gpu => core_config.GpuConfig,
        .ai => core_config.AiConfig,
        .database => core_config.DatabaseConfig,
        .network => core_config.NetworkConfig,
        .observability => core_config.ObservabilityConfig,
        .web => core_config.WebConfig,
        .cloud => core_config.CloudConfig,
        .analytics => core_config.AnalyticsConfig,
        .auth => core_config.AuthConfig,
        .messaging => core_config.MessagingConfig,
        .cache => core_config.CacheConfig,
        .storage => core_config.StorageConfig,
        .search => core_config.SearchConfig,
        .mobile => core_config.MobileConfig,
        .gateway => core_config.GatewayConfig,
        .pages => core_config.PagesConfig,
        .benchmarks => core_config.BenchmarksConfig,
        // Sub-features use their specific config type
        .llm => core_config.LlmConfig,
        .embeddings => core_config.EmbeddingsConfig,
        .agents => core_config.AgentsConfig,
        .training => core_config.TrainingConfig,
        .personas => core_config.ai_config.PersonasConfig,
        .reasoning => core_config.AiConfig,
    };
}

// ============================================================================
// Types
// ============================================================================

/// Options for controlled shutdown.
pub const StopOptions = struct {
    /// Timeout in milliseconds. Null means immediate (no timeout).
    timeout_ms: ?u64 = null,
};

// ============================================================================
// App
// ============================================================================

pub const App = struct {
    framework: framework_mod.Framework,

    /// Framework lifecycle states (re-exported for convenience).
    pub const State = framework_mod.Framework.State;

    pub const Error = framework_mod.Framework.Error || error{
        CapabilityUnavailable,
    };

    // ------------------------------------------------------------------
    // Lifecycle: new API
    // ------------------------------------------------------------------

    /// Create and start an App with the given configuration.
    ///
    /// This is the primary vNext entry point. Empty config (`start(alloc, .{})`)
    /// uses all compile-time defaults.
    pub fn start(allocator: std.mem.Allocator, cfg: config_mod.AppConfig) Error!App {
        return init(allocator, cfg);
    }

    /// Shutdown the application and release all resources.
    pub fn stop(self: *App) void {
        self.framework.deinit();
    }

    /// Shutdown with options (e.g., timeout).
    ///
    /// Returns `true` if clean shutdown completed within the timeout.
    pub fn stopWithOptions(self: *App, opts: StopOptions) bool {
        if (opts.timeout_ms) |ms| {
            return self.framework.shutdownWithTimeout(ms);
        }
        self.framework.deinit();
        return true;
    }

    // ------------------------------------------------------------------
    // Feature access: new API
    // ------------------------------------------------------------------

    /// Get a typed feature context by capability.
    ///
    /// Returns the optional context pointer directly from the Framework.
    /// Use `if (app.feature(.gpu)) |ctx| { ... }` for safe access.
    pub fn feature(self: *App, comptime cap: Capability) FeatureFieldType(cap) {
        return @field(self.framework, frameworkFieldName(cap));
    }

    /// Check whether a capability is enabled in this App's config.
    pub fn has(self: *const App, cap: Capability) bool {
        return self.framework.isEnabled(capability_mod.toFeature(core_config.Feature, cap));
    }

    /// Get the current lifecycle state.
    pub fn state(self: *const App) State {
        return self.framework.getState();
    }

    // ------------------------------------------------------------------
    // Infrastructure access: new API
    // ------------------------------------------------------------------

    /// Get the runtime context (always available).
    pub fn runtime(self: *App) *runtime_mod.Context {
        return self.framework.getRuntime();
    }

    /// Get the feature registry.
    pub fn registry(self: *App) *framework_mod.Registry {
        return self.framework.getRegistry();
    }

    // ------------------------------------------------------------------
    // Builder entry point
    // ------------------------------------------------------------------

    /// Start building an App configuration with the fluent builder API.
    ///
    /// ```zig
    /// var app = try App.configure(allocator)
    ///     .enable(.gpu, .{ .backend = .metal })
    ///     .enable(.database, .{})
    ///     .start();
    /// defer app.stop();
    /// ```
    pub fn configure(allocator: std.mem.Allocator) AppBuilder {
        return .{
            .allocator = allocator,
            .config = .{},
        };
    }

    // ------------------------------------------------------------------
    // Legacy API (preserved for backward compatibility)
    // ------------------------------------------------------------------

    pub fn init(allocator: std.mem.Allocator, cfg: config_mod.AppConfig) Error!App {
        var fw = try framework_mod.Framework.init(allocator, cfg.framework);
        errdefer fw.deinit();

        if (cfg.strict_capability_check) {
            for (cfg.required_capabilities) |capability| {
                const feat = capability_mod.toFeature(core_config.Feature, capability);
                if (!fw.isEnabled(feat)) {
                    return error.CapabilityUnavailable;
                }
            }
        }

        return .{ .framework = fw };
    }

    pub fn initDefault(allocator: std.mem.Allocator) Error!App {
        return init(allocator, config_mod.AppConfig.defaults());
    }

    pub fn deinit(self: *App) void {
        self.framework.deinit();
    }

    pub fn getFramework(self: *App) *framework_mod.Framework {
        return &self.framework;
    }

    pub fn getFrameworkConst(self: *const App) *const framework_mod.Framework {
        return &self.framework;
    }
};

// ============================================================================
// AppBuilder
// ============================================================================

/// Fluent builder for App configuration.
///
/// Provides a unified `.enable(cap, config)` method instead of per-feature
/// `withGpu()`, `withAi()`, etc.
pub const AppBuilder = struct {
    allocator: std.mem.Allocator,
    config: core_config.Config,
    io: ?std.Io = null,

    /// Enable a capability with its config.
    ///
    /// For sub-features (`.llm`, `.embeddings`, `.agents`, `.training`,
    /// `.personas`), the parent `.ai` config is auto-created if null.
    pub fn enable(self: *AppBuilder, comptime cap: Capability, cfg: ConfigType(cap)) *AppBuilder {
        switch (cap) {
            // Top-level features: set directly on Config
            .gpu => self.config.gpu = cfg,
            .ai => self.config.ai = cfg,
            .database => self.config.database = cfg,
            .network => self.config.network = cfg,
            .observability => self.config.observability = cfg,
            .web => self.config.web = cfg,
            .cloud => self.config.cloud = cfg,
            .analytics => self.config.analytics = cfg,
            .auth => self.config.auth = cfg,
            .messaging => self.config.messaging = cfg,
            .cache => self.config.cache = cfg,
            .storage => self.config.storage = cfg,
            .search => self.config.search = cfg,
            .mobile => self.config.mobile = cfg,
            .gateway => self.config.gateway = cfg,
            .pages => self.config.pages = cfg,
            .benchmarks => self.config.benchmarks = cfg,
            // Sub-features: auto-create parent .ai if needed
            .llm => {
                if (self.config.ai == null) self.config.ai = .{};
                self.config.ai.?.llm = cfg;
            },
            .embeddings => {
                if (self.config.ai == null) self.config.ai = .{};
                self.config.ai.?.embeddings = cfg;
            },
            .agents => {
                if (self.config.ai == null) self.config.ai = .{};
                self.config.ai.?.agents = cfg;
            },
            .training => {
                if (self.config.ai == null) self.config.ai = .{};
                self.config.ai.?.training = cfg;
            },
            .personas => {
                if (self.config.ai == null) self.config.ai = .{};
                self.config.ai.?.personas = cfg;
            },
            .reasoning => {
                // Reasoning uses the ai config (enable_reasoning build flag)
                self.config.ai = cfg;
            },
        }
        return self;
    }

    /// Set the I/O backend for the App.
    pub fn withIo(self: *AppBuilder, io_val: std.Io) *AppBuilder {
        self.io = io_val;
        return self;
    }

    /// Build and start the App.
    pub fn start(self: *AppBuilder) App.Error!App {
        const app_cfg = config_mod.AppConfig{
            .framework = self.config,
        };
        return App.init(self.allocator, app_cfg);
    }
};
