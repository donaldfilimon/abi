//! Public ABI package root.
//!
//! The direct-domain surface is now the canonical package API. Legacy
//! `abi.features.*` and `abi.services.*` aliases remain as phase-1
//! compatibility bridges while in-repo callers migrate.

const std = @import("std");
const build_options = @import("build_options");

const framework_mod = @import("core/framework.zig");
const foundation_mod = @import("services/shared/mod.zig");
const runtime_mod = @import("services/runtime/mod.zig");
const platform_mod = @import("services/platform/mod.zig");
const connectors_mod = @import("services/connectors/mod.zig");
const tasks_mod = @import("services/tasks/mod.zig");
const mcp_mod = @import("services/mcp/mod.zig");
const lsp_mod = @import("services/lsp/mod.zig");
const acp_mod = @import("services/acp/mod.zig");
const ha_mod = @import("services/ha/mod.zig");
const gpu_mod = if (build_options.feat_gpu)
    @import("features/gpu/mod.zig")
else
    @import("features/gpu/stub.zig");
const ai_mod = if (build_options.feat_ai)
    @import("features/ai/mod.zig")
else
    @import("features/ai/stub.zig");
const database_mod = if (build_options.feat_database)
    @import("features/database/mod.zig")
else
    @import("features/database/stub.zig");
const network_mod = if (build_options.feat_network)
    @import("features/network/mod.zig")
else
    @import("features/network/stub.zig");
const observability_mod = if (build_options.feat_profiling)
    @import("features/observability/mod.zig")
else
    @import("features/observability/stub.zig");
const web_mod = if (build_options.feat_web)
    @import("features/web/mod.zig")
else
    @import("features/web/stub.zig");
const pages_mod = if (build_options.feat_pages)
    @import("features/observability/pages/mod.zig")
else
    @import("features/observability/pages/stub.zig");
const analytics_mod = if (build_options.feat_analytics)
    @import("features/analytics/mod.zig")
else
    @import("features/analytics/stub.zig");
const cloud_mod = if (build_options.feat_cloud)
    @import("features/cloud/mod.zig")
else
    @import("features/cloud/stub.zig");
const auth_mod = if (build_options.feat_auth)
    @import("features/auth/mod.zig")
else
    @import("features/auth/stub.zig");
const messaging_mod = if (build_options.feat_messaging)
    @import("features/messaging/mod.zig")
else
    @import("features/messaging/stub.zig");
const cache_mod = if (build_options.feat_cache)
    @import("features/cache/mod.zig")
else
    @import("features/cache/stub.zig");
const storage_mod = if (build_options.feat_storage)
    @import("features/storage/mod.zig")
else
    @import("features/storage/stub.zig");
const search_mod = if (build_options.feat_search)
    @import("features/search/mod.zig")
else
    @import("features/search/stub.zig");
const mobile_mod = if (build_options.feat_mobile)
    @import("features/mobile/mod.zig")
else
    @import("features/mobile/stub.zig");
const gateway_mod = if (build_options.feat_gateway)
    @import("features/gateway/mod.zig")
else
    @import("features/gateway/stub.zig");
const benchmarks_mod = if (build_options.feat_benchmarks)
    @import("features/benchmarks/mod.zig")
else
    @import("features/benchmarks/stub.zig");
const compute_mod = if (build_options.feat_compute)
    @import("features/compute/mod.zig")
else
    @import("features/compute/stub.zig");
const documents_mod = if (build_options.feat_documents)
    @import("features/documents/mod.zig")
else
    @import("features/documents/stub.zig");
const desktop_mod = if (build_options.feat_desktop)
    @import("features/desktop/mod.zig")
else
    @import("features/desktop/stub.zig");
const FrameworkApp = framework_mod.Framework;
const FrameworkAppBuilder = framework_mod.FrameworkBuilder;

pub const config = @import("core/config/mod.zig");
pub const Config = config.Config;
pub const Feature = config.Feature;

pub const errors = @import("core/errors.zig");
pub const FrameworkError = errors.FrameworkError;

pub const registry = @import("core/registry/mod.zig");
pub const Registry = registry.Registry;

pub const foundation = foundation_mod;
pub const runtime = runtime_mod;
pub const platform = platform_mod;
pub const connectors = connectors_mod;
pub const tasks = tasks_mod;
pub const mcp = mcp_mod;
pub const lsp = lsp_mod;
pub const acp = acp_mod;
pub const ha = ha_mod;

pub const gpu = gpu_mod;
pub const ai = ai_mod;
pub const database = database_mod;
pub const network = network_mod;
pub const observability = observability_mod;
pub const web = web_mod;
pub const pages = pages_mod;
pub const analytics = analytics_mod;
pub const cloud = cloud_mod;
pub const auth = auth_mod;
pub const messaging = messaging_mod;
pub const cache = cache_mod;
pub const storage = storage_mod;
pub const search = search_mod;
pub const mobile = mobile_mod;
pub const gateway = gateway_mod;
pub const benchmarks = benchmarks_mod;
pub const compute = compute_mod;
pub const documents = documents_mod;
pub const desktop = desktop_mod;

pub const meta = struct {
    pub const package_version = build_options.package_version;
    pub const features = @import("core/feature_catalog.zig");

    pub fn version() []const u8 {
        return package_version;
    }
};

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

// Compatibility bridges preserved during the logical-graph migration.
pub const feature_catalog = meta.features;
pub const framework = framework_mod;

pub const services = struct {
    pub const foundation = foundation_mod;
    pub const shared = foundation_mod;
    pub const runtime = runtime_mod;
    pub const platform = platform_mod;
    pub const connectors = connectors_mod;
    pub const tasks = tasks_mod;
    pub const lsp = lsp_mod;
    pub const mcp = mcp_mod;
    pub const acp = acp_mod;
    pub const ha = ha_mod;
    pub const simd = foundation_mod.simd;
};

pub const features = struct {
    pub const gpu = gpu_mod;
    pub const ai = ai_mod;
    pub const database = database_mod;
    pub const network = network_mod;
    pub const observability = observability_mod;
    pub const web = web_mod;
    pub const analytics = analytics_mod;
    pub const cloud = cloud_mod;
    pub const auth = auth_mod;
    pub const messaging = messaging_mod;
    pub const cache = cache_mod;
    pub const storage = storage_mod;
    pub const mobile = mobile_mod;
    pub const gateway = gateway_mod;
    pub const search = search_mod;
    pub const pages = pages_mod;
    pub const benchmarks = benchmarks_mod;
    pub const compute = compute_mod;
    pub const documents = documents_mod;
    pub const desktop = desktop_mod;
};

test {
    std.testing.refAllDecls(@This());
}
