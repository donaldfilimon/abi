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
pub const config = @import("core/config/mod.zig");
pub const Config = config.Config;
pub const Feature = config.Feature;
pub const feature_catalog = @import("core/feature_catalog.zig");

/// Framework orchestration with builder pattern.
pub const framework = @import("core/framework.zig");

/// Composable error hierarchy for framework operations.
pub const errors = @import("core/errors.zig");
pub const FrameworkError = errors.FrameworkError;

/// Plugin registry for feature management.
pub const registry = @import("core/registry/mod.zig");
pub const Registry = registry.Registry;

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
        .withDefault(.gpu)
        .withDefault(.ai)
        .build();

    if (build_options.enable_gpu) {
        try std.testing.expect(cfg.gpu != null);
    }
    if (build_options.enable_ai) {
        try std.testing.expect(cfg.ai != null);
    }
}
