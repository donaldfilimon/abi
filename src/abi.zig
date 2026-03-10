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
//!     var fw = try abi.App.initDefault(arena);
//!     defer fw.deinit();
//!
//!     std.debug.print("ABI v{s}\n", .{abi.version()});
//! }
//! ```
//!
//! ## Builder Pattern
//!
//! ```zig
//! var fw = try abi.App.builder(allocator)
//!     .with(.gpu, abi.config.GpuConfig{ .backend = .vulkan })
//!     .with(.ai, abi.config.AiConfig{ .llm = .{ .model_path = "./models/llama.gguf" } })
//!     .with(.database, abi.config.DatabaseConfig{ .path = "./data" })
//!     .build();
//! defer fw.deinit();
//! ```
//!
//! ## Feature Modules
//!
//! Access features through namespace exports:
//! - `abi.features.ai` (with submodules: `.core`, `.llm`, `.training`, `.streaming`, etc.)
//! - `abi.features.gpu`, `abi.features.database`, `abi.features.network`, `abi.features.web`, `abi.features.cloud`
//! - `abi.features.observability`, `abi.features.analytics`, `abi.features.auth`, `abi.features.messaging`
//! - `abi.features.cache`, `abi.features.storage`, `abi.features.search`, `abi.features.gateway`, `abi.features.pages`
//! - `abi.services.shared.simd`, `abi.services.connectors.discord`

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
pub const config = @import("core/config");
pub const Config = config.Config;
pub const Feature = config.Feature;
pub const feature_catalog = @import("core/feature_catalog");

/// Framework orchestration with builder pattern.
pub const framework = @import("core/framework");

/// Composable error hierarchy for framework operations.
pub const errors = @import("core/errors");
pub const FrameworkError = errors.FrameworkError;

/// Plugin registry for feature management.
pub const registry = @import("core/registry");
pub const Registry = registry.Registry;

// ============================================================================
// Canonical Namespaces (v2 API)
// ============================================================================

/// Canonical services namespace.
pub const services = struct {
    pub const runtime = @import("services/runtime");
    pub const platform = @import("services/platform");
    pub const shared = @import("shared_services");
    pub const connectors = @import("services/connectors");
    pub const ha = @import("services/ha");
    pub const tasks = @import("services/tasks");
    pub const lsp = @import("services/lsp");
    pub const mcp = @import("services/mcp");
    pub const acp = @import("services/acp");
    pub const simd = shared.simd;
};

/// Canonical features namespace.
pub const features = struct {
    pub const gpu = if (build_options.feat_gpu)
        @import("features/gpu")
    else
        @import("features/gpu/stub");
    pub const ai = if (build_options.feat_ai)
        @import("features/ai")
    else
        @import("features/ai/stub");
    pub const database = if (build_options.feat_database)
        @import("features/database")
    else
        @import("features/database/stub");

    pub const network = if (build_options.feat_network)
        @import("features/network")
    else
        @import("features/network/stub");
    pub const observability = if (build_options.feat_profiling)
        @import("features/observability")
    else
        @import("features/observability/stub");
    pub const web = if (build_options.feat_web)
        @import("features/web")
    else
        @import("features/web/stub");
    pub const analytics = if (build_options.feat_analytics)
        @import("features/analytics")
    else
        @import("features/analytics/stub");
    pub const cloud = if (build_options.feat_cloud)
        @import("features/cloud")
    else
        @import("features/cloud/stub");
    pub const auth = if (build_options.feat_auth)
        @import("features/auth")
    else
        @import("features/auth/stub");
    pub const messaging = if (build_options.feat_messaging)
        @import("features/messaging")
    else
        @import("features/messaging/stub");
    pub const cache = if (build_options.feat_cache)
        @import("features/cache")
    else
        @import("features/cache/stub");
    pub const storage = if (build_options.feat_storage)
        @import("features/storage")
    else
        @import("features/storage/stub");
    pub const mobile = if (build_options.feat_mobile)
        @import("features/mobile")
    else
        @import("features/mobile/stub");
    pub const gateway = if (build_options.feat_gateway)
        @import("features/gateway")
    else
        @import("features/gateway/stub");
    pub const search = if (build_options.feat_search)
        @import("features/search")
    else
        @import("features/search/stub");
    pub const pages = if (build_options.feat_pages)
        @import("features/observability/pages")
    else
        @import("features/observability/pages/stub");
    pub const benchmarks = if (build_options.feat_benchmarks)
        @import("features/benchmarks")
    else
        @import("features/benchmarks/stub");

    // Distributed compute mesh
    pub const compute = if (build_options.feat_compute)
        @import("features/compute")
    else
        @import("features/compute/stub");

    // Omni-modal document parsing
    pub const documents = if (build_options.feat_documents)
        @import("features/documents")
    else
        @import("features/documents/stub");

    // Native Desktop extensions
    pub const desktop = if (build_options.feat_desktop)
        @import("features/desktop")
    else
        @import("features/desktop/stub");
};

/// Build and package metadata.
pub const meta = struct {
    pub const package_version = build_options.package_version;

    pub fn version() []const u8 {
        return package_version;
    }
};

// ============================================================================
// Convenience Aliases (minimal)
// ============================================================================

/// Canonical application type alias (v2 API).
pub const App = framework.Framework;
/// Canonical application builder type alias (v2 API).
pub const AppBuilder = framework.FrameworkBuilder;

/// GPU handle and backend type; use `abi.features.gpu` for full API.
pub const Gpu = features.gpu.Gpu;
/// GPU backend enum (cuda, vulkan, metal, webgpu, tpu, etc.); use `abi.features.gpu` for full API.
pub const GpuBackend = features.gpu.Backend;

// ============================================================================
// Primary API
// ============================================================================

/// Canonical v2 builder entrypoint.
pub fn appBuilder(allocator: std.mem.Allocator) AppBuilder {
    return App.builder(allocator);
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
    _ = features.ai.profiles;
    _ = features.ai.coordination;
    _ = features.database.semantic_store;
    _ = services.runtime;
    try std.testing.expectEqualStrings(meta.version(), version());
}

test "framework initialization with defaults" {
    const cfg = Config.defaults();
    try std.testing.expect(cfg.gpu != null or !build_options.feat_gpu);
}

test "config builder pattern" {
    var builder = config.Builder.init(std.testing.allocator);
    const cfg = builder
        .withDefault(.gpu)
        .withDefault(.ai)
        .build();

    if (build_options.feat_gpu) {
        try std.testing.expect(cfg.gpu != null);
    }
    if (build_options.feat_ai) {
        try std.testing.expect(cfg.ai != null);
    }
}
