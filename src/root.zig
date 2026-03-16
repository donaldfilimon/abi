//! Public ABI package root.
//!
//! Canonical API: `abi.<domain>` (e.g. `abi.gpu`, `abi.ai`, `abi.connectors`).

const std = @import("std");
const build_options = @import("build_options");

// ── Core ─────────────────────────────────────────────────────────────────

pub const config = @import("core/config/mod.zig");
pub const Config = config.Config;
pub const Feature = config.Feature;

pub const errors = @import("core/errors.zig");
pub const FrameworkError = errors.FrameworkError;

pub const registry = @import("core/registry/mod.zig");
pub const Registry = registry.Registry;

pub const framework = @import("core/framework.zig");

// ── Services (non-feature-gated) ─────────────────────────────────────────

pub const foundation = @import("services/shared/mod.zig");
pub const runtime = @import("services/runtime/mod.zig");
pub const platform = @import("services/platform/mod.zig");
pub const connectors = @import("services/connectors/mod.zig");
pub const tasks = @import("services/tasks/mod.zig");
pub const mcp = @import("services/mcp/mod.zig");
pub const lsp = @import("services/lsp/mod.zig");
pub const acp = @import("services/acp/mod.zig");
pub const ha = @import("services/ha/mod.zig");
pub const inference = @import("inference/mod.zig");

// ── Features (comptime-gated mod/stub) ───────────────────────────────────

pub const gpu = if (build_options.feat_gpu) @import("features/gpu/mod.zig") else @import("features/gpu/stub.zig");
pub const ai = if (build_options.feat_ai) @import("features/ai/mod.zig") else @import("features/ai/stub.zig");
pub const database = if (build_options.feat_database) @import("features/database/mod.zig") else @import("features/database/stub.zig");
pub const network = if (build_options.feat_network) @import("features/network/mod.zig") else @import("features/network/stub.zig");
pub const observability = if (build_options.feat_profiling) @import("features/observability/mod.zig") else @import("features/observability/stub.zig");
pub const web = if (build_options.feat_web) @import("features/web/mod.zig") else @import("features/web/stub.zig");
pub const pages = if (build_options.feat_pages) @import("features/observability/pages/mod.zig") else @import("features/observability/pages/stub.zig");
pub const analytics = if (build_options.feat_analytics) @import("features/analytics/mod.zig") else @import("features/analytics/stub.zig");
pub const cloud = if (build_options.feat_cloud) @import("features/cloud/mod.zig") else @import("features/cloud/stub.zig");
pub const auth = if (build_options.feat_auth) @import("features/auth/mod.zig") else @import("features/auth/stub.zig");
pub const messaging = if (build_options.feat_messaging) @import("features/messaging/mod.zig") else @import("features/messaging/stub.zig");
pub const cache = if (build_options.feat_cache) @import("features/cache/mod.zig") else @import("features/cache/stub.zig");
pub const storage = if (build_options.feat_storage) @import("features/storage/mod.zig") else @import("features/storage/stub.zig");
pub const search = if (build_options.feat_search) @import("features/search/mod.zig") else @import("features/search/stub.zig");
pub const mobile = if (build_options.feat_mobile) @import("features/mobile/mod.zig") else @import("features/mobile/stub.zig");
pub const gateway = if (build_options.feat_gateway) @import("features/gateway/mod.zig") else @import("features/gateway/stub.zig");
pub const benchmarks = if (build_options.feat_benchmarks) @import("features/benchmarks/mod.zig") else @import("features/benchmarks/stub.zig");
pub const compute = if (build_options.feat_compute) @import("features/compute/mod.zig") else @import("features/compute/stub.zig");
pub const documents = if (build_options.feat_documents) @import("features/documents/mod.zig") else @import("features/documents/stub.zig");
pub const desktop = if (build_options.feat_desktop) @import("features/desktop/mod.zig") else @import("features/desktop/stub.zig");

// ── Convenience aliases ──────────────────────────────────────────────────

pub const meta = struct {
    pub const package_version = build_options.package_version;
    pub const features = @import("core/feature_catalog.zig");

    pub fn version() []const u8 {
        return package_version;
    }
};

const FrameworkApp = framework.Framework;
const FrameworkAppBuilder = framework.FrameworkBuilder;

pub const app = struct {
    pub const App = FrameworkApp;
    pub const AppBuilder = FrameworkAppBuilder;
    pub const Error = FrameworkApp.Error;

    pub fn builder(allocator: std.mem.Allocator) FrameworkAppBuilder {
        return FrameworkApp.builder(allocator);
    }

    pub fn version() []const u8 {
        return meta.version();
    }
};

pub const App = FrameworkApp;
pub const AppBuilder = FrameworkAppBuilder;
pub const Gpu = gpu.Gpu;
pub const GpuBackend = gpu.Backend;

pub fn appBuilder(allocator: std.mem.Allocator) AppBuilder {
    return App.builder(allocator);
}

pub fn version() []const u8 {
    return meta.version();
}

pub const feature_catalog = meta.features;

test {
    std.testing.refAllDecls(@This());
}
