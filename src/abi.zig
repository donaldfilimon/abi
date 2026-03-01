//! ABI Framework - Main Library Interface
//!
//! A modern Zig 0.16 framework for modular AI services, vector search,
//! and high-performance compute. This is the primary entry point for all
//! ABI functionality.
//!
//! ## Quick Start
//!
//! ```zig
//! const std = @import("std");
//! const abi = @import("abi");
//!
//! pub fn main(init: std.process.Init) !void {
//!     const arena = init.arena.allocator();
//!
//!     var fw = try abi.initDefault(arena);
//!     defer fw.deinit();
//!
//!     std.debug.print("ABI v{s}\n", .{abi.version()});
//! }
//! ```
//!
//! ## Builder Pattern
//!
//! ```zig
//! var fw = try abi.Framework.builder(allocator)
//!     .withGpu(.{ .backend = .vulkan })
//!     .withAi(.{ .llm = .{ .model_path = "./models/llama.gguf" } })
//!     .withDatabase(.{ .path = "./data" })
//!     .build();
//! defer fw.deinit();
//! ```
//!
//! ## Feature Modules
//!
//! Access features through namespace exports:
//! - `abi.ai` (with submodules: `.core`, `.llm`, `.training`, `.streaming`, etc.)
//! - `abi.gpu`, `abi.database`, `abi.network`, `abi.web`, `abi.cloud`
//! - `abi.observability`, `abi.analytics`, `abi.auth`, `abi.messaging`
//! - `abi.cache`, `abi.storage`, `abi.search`, `abi.gateway`, `abi.pages`
//! - `abi.shared.simd`, `abi.connectors.discord`

const std = @import("std");
const build_options = @import("build_options");
const builtin = @import("builtin");

comptime {
    if (builtin.zig_version.major == 0 and builtin.zig_version.minor < 16) {
        @compileError("ABI requires Zig 0.16.0 or newer");
    }
}

// ============================================================================
// Core (always available)
// ============================================================================

/// Unified configuration system.
pub const config = @import("core/config/mod.zig");
pub const Config = config.Config;
pub const Feature = config.Feature;
pub const feature_catalog = @import("core/feature_catalog.zig");

/// Framework orchestration with builder pattern.
pub const framework = @import("core/framework.zig");
pub const Framework = framework.Framework;
pub const FrameworkBuilder = framework.FrameworkBuilder;

/// Composable error hierarchy for framework operations.
pub const errors = @import("core/errors.zig");
pub const FrameworkError = errors.FrameworkError;

/// Plugin registry for feature management.
pub const registry = @import("core/registry/mod.zig");
pub const Registry = registry.Registry;

// ============================================================================
// Services (always available)
// ============================================================================

/// Runtime infrastructure (thread pool, channels, scheduling).
pub const runtime = @import("services/runtime/mod.zig");

/// Platform detection and abstraction.
pub const platform = @import("services/platform/mod.zig");

/// Shared utilities (SIMD, time, sync, security, etc.).
pub const shared = @import("services/shared/mod.zig");

/// External service connectors (OpenAI, Anthropic, Ollama, etc.).
pub const connectors = @import("services/connectors/mod.zig");

/// High availability (replication, backup, PITR).
pub const ha = @import("services/ha/mod.zig");

/// Task management system.
pub const tasks = @import("services/tasks/mod.zig");

/// LSP (ZLS) client utilities.
pub const lsp = @import("services/lsp/mod.zig");

/// MCP (Model Context Protocol) server for WDBX database.
pub const mcp = @import("services/mcp/mod.zig");

/// ACP (Agent Communication Protocol) for agent-to-agent communication.
pub const acp = @import("services/acp/mod.zig");

/// SIMD operations (shorthand for `shared.simd`).
pub const simd = @import("services/shared/simd/mod.zig");

// ============================================================================
// Feature Modules (comptime-gated)
// ============================================================================

/// GPU acceleration.
pub const gpu = if (build_options.enable_gpu)
    @import("features/gpu/mod.zig")
else
    @import("features/gpu/stub.zig");

/// AI capabilities (modular sub-features).
pub const ai = if (build_options.enable_ai)
    @import("features/ai/mod.zig")
else
    @import("features/ai/stub.zig");

// NOTE(v0.4.0): Facade aliases `inference`, `training`, `reasoning`, `ai_core`
// have been removed. Use the canonical sub-module paths instead:
//   abi.ai.llm          (was abi.inference)
//   abi.ai.training      (was abi.training)
//   abi.ai.orchestration (was abi.reasoning)
//   abi.ai.core          (was abi.ai_core)

/// Vector database.
pub const database = if (build_options.enable_database)
    @import("features/database/mod.zig")
else
    @import("features/database/stub.zig");

/// Distributed network.
pub const network = if (build_options.enable_network)
    @import("features/network/mod.zig")
else
    @import("features/network/stub.zig");

/// Observability (metrics, tracing, profiling).
pub const observability = if (build_options.enable_profiling)
    @import("features/observability/mod.zig")
else
    @import("features/observability/stub.zig");

/// Web utilities.
pub const web = if (build_options.enable_web)
    @import("features/web/mod.zig")
else
    @import("features/web/stub.zig");

/// Analytics event tracking.
pub const analytics = if (build_options.enable_analytics)
    @import("features/analytics/mod.zig")
else
    @import("features/analytics/stub.zig");

/// Cloud function adapters.
pub const cloud = if (build_options.enable_cloud)
    @import("features/cloud/mod.zig")
else
    @import("features/cloud/stub.zig");

/// Authentication and security.
pub const auth = if (build_options.enable_auth)
    @import("features/auth/mod.zig")
else
    @import("features/auth/stub.zig");

/// Event bus and messaging.
pub const messaging = if (build_options.enable_messaging)
    @import("features/messaging/mod.zig")
else
    @import("features/messaging/stub.zig");

/// In-memory caching.
pub const cache = if (build_options.enable_cache)
    @import("features/cache/mod.zig")
else
    @import("features/cache/stub.zig");

/// Unified file/object storage.
pub const storage = if (build_options.enable_storage)
    @import("features/storage/mod.zig")
else
    @import("features/storage/stub.zig");

/// Mobile platform (lifecycle, sensors, notifications).
pub const mobile = if (build_options.enable_mobile)
    @import("features/mobile/mod.zig")
else
    @import("features/mobile/stub.zig");

/// API gateway (routing, rate limiting, circuit breaker).
pub const gateway = if (build_options.enable_gateway)
    @import("features/gateway/mod.zig")
else
    @import("features/gateway/stub.zig");

/// Full-text search.
pub const search = if (build_options.enable_search)
    @import("features/search/mod.zig")
else
    @import("features/search/stub.zig");

/// Dashboard/UI pages with URL path routing.
pub const pages = if (build_options.enable_pages)
    @import("features/observability/pages/mod.zig")
else
    @import("features/observability/pages/stub.zig");

/// Performance benchmarking and timing.
pub const benchmarks = if (build_options.enable_benchmarks)
    @import("features/benchmarks/mod.zig")
else
    @import("features/benchmarks/stub.zig");

// ============================================================================
// Canonical Namespaces (v2 API)
// ============================================================================

/// Canonical services namespace.
pub const services = struct {
    pub const runtime = @import("services/runtime/mod.zig");
    pub const platform = @import("services/platform/mod.zig");
    pub const shared = @import("services/shared/mod.zig");
    pub const connectors = @import("services/connectors/mod.zig");
    pub const ha = @import("services/ha/mod.zig");
    pub const tasks = @import("services/tasks/mod.zig");
    pub const lsp = @import("services/lsp/mod.zig");
    pub const mcp = @import("services/mcp/mod.zig");
    pub const acp = @import("services/acp/mod.zig");
    pub const simd = @import("services/shared/simd/mod.zig");
};

/// Canonical features namespace.
pub const features = struct {
    pub const gpu = if (build_options.enable_gpu)
        @import("features/gpu/mod.zig")
    else
        @import("features/gpu/stub.zig");
    pub const ai = if (build_options.enable_ai)
        @import("features/ai/mod.zig")
    else
        @import("features/ai/stub.zig");
    pub const database = if (build_options.enable_database)
        @import("features/database/mod.zig")
    else
        @import("features/database/stub.zig");
    pub const network = if (build_options.enable_network)
        @import("features/network/mod.zig")
    else
        @import("features/network/stub.zig");
    pub const observability = if (build_options.enable_profiling)
        @import("features/observability/mod.zig")
    else
        @import("features/observability/stub.zig");
    pub const web = if (build_options.enable_web)
        @import("features/web/mod.zig")
    else
        @import("features/web/stub.zig");
    pub const analytics = if (build_options.enable_analytics)
        @import("features/analytics/mod.zig")
    else
        @import("features/analytics/stub.zig");
    pub const cloud = if (build_options.enable_cloud)
        @import("features/cloud/mod.zig")
    else
        @import("features/cloud/stub.zig");
    pub const auth = if (build_options.enable_auth)
        @import("features/auth/mod.zig")
    else
        @import("features/auth/stub.zig");
    pub const messaging = if (build_options.enable_messaging)
        @import("features/messaging/mod.zig")
    else
        @import("features/messaging/stub.zig");
    pub const cache = if (build_options.enable_cache)
        @import("features/cache/mod.zig")
    else
        @import("features/cache/stub.zig");
    pub const storage = if (build_options.enable_storage)
        @import("features/storage/mod.zig")
    else
        @import("features/storage/stub.zig");
    pub const mobile = if (build_options.enable_mobile)
        @import("features/mobile/mod.zig")
    else
        @import("features/mobile/stub.zig");
    pub const gateway = if (build_options.enable_gateway)
        @import("features/gateway/mod.zig")
    else
        @import("features/gateway/stub.zig");
    pub const search = if (build_options.enable_search)
        @import("features/search/mod.zig")
    else
        @import("features/search/stub.zig");
    pub const pages = if (build_options.enable_pages)
        @import("features/observability/pages/mod.zig")
    else
        @import("features/observability/pages/stub.zig");
    pub const benchmarks = if (build_options.enable_benchmarks)
        @import("features/benchmarks/mod.zig")
    else
        @import("features/benchmarks/stub.zig");
};

/// Build and package metadata.
pub const meta = struct {
    pub const package_version = build_options.package_version;

    pub fn version() []const u8 {
        return package_version;
    }
};

/// Deprecated API compatibility namespace.
pub const compat = struct {
    pub const v1 = @import("compat/v1/mod.zig");
};

// ============================================================================
// Convenience Aliases (minimal)
// ============================================================================

/// Canonical application type alias (v2 API).
pub const App = Framework;
/// Canonical application builder type alias (v2 API).
pub const AppBuilder = FrameworkBuilder;

/// GPU handle and backend type; use `abi.gpu` for full API.
pub const Gpu = gpu.Gpu;
/// GPU backend enum (cuda, vulkan, metal, webgpu, tpu, etc.); use `abi.gpu` for full API.
pub const GpuBackend = gpu.Backend;

// ============================================================================
// Primary API
// ============================================================================

/// Initialize the ABI framework with custom configuration.
pub fn init(allocator: std.mem.Allocator, cfg: Config) !Framework {
    return Framework.init(allocator, cfg);
}

/// Initialize the ABI framework with default configuration.
pub fn initDefault(allocator: std.mem.Allocator) !Framework {
    return Framework.initDefault(allocator);
}

/// Canonical v2 initializer for application runtime.
pub fn initApp(allocator: std.mem.Allocator, cfg: Config) !App {
    return init(allocator, cfg);
}

/// Canonical v2 default initializer for application runtime.
pub fn initAppDefault(allocator: std.mem.Allocator) !App {
    return initDefault(allocator);
}

/// Canonical v2 builder entrypoint.
pub fn appBuilder(allocator: std.mem.Allocator) AppBuilder {
    return Framework.builder(allocator);
}

/// Get the ABI framework version string.
pub fn version() []const u8 {
    return meta.version();
}

// ============================================================================
// Tests
// ============================================================================

test {
    std.testing.refAllDecls(@This());
}

test "abi.version returns build package version" {
    try std.testing.expectEqualStrings("0.4.0", version());
}

test "v2 surface exports remain available" {
    _ = App;
    _ = AppBuilder;
    _ = features.ai;
    _ = services.runtime;
    _ = compat.v1.Framework;
    try std.testing.expectEqualStrings(meta.version(), version());
}

test "framework initialization with defaults" {
    const cfg = Config.defaults();
    try std.testing.expect(cfg.gpu != null or !build_options.enable_gpu);
}

test "config builder pattern" {
    var builder = config.Builder.init(std.testing.allocator);
    const cfg = builder
        .withGpuDefaults()
        .withAiDefaults()
        .build();

    if (build_options.enable_gpu) {
        try std.testing.expect(cfg.gpu != null);
    }
    if (build_options.enable_ai) {
        try std.testing.expect(cfg.ai != null);
    }
}
