//! ABI v1 compatibility surface.
//!
//! Temporary shim module that preserves high-usage legacy names while v2
//! namespaces (`abi.features`, `abi.services`, `abi.App*`) are adopted.

const std = @import("std");
const build_options = @import("build_options");

// Core
pub const config = @import("../../core/config/mod.zig");
pub const Config = config.Config;
pub const Feature = config.Feature;
pub const feature_catalog = @import("../../core/feature_catalog.zig");
pub const framework = @import("../../core/framework.zig");
pub const Framework = framework.Framework;
pub const FrameworkBuilder = framework.FrameworkBuilder;
pub const errors = @import("../../core/errors.zig");
pub const FrameworkError = errors.FrameworkError;
pub const registry = @import("../../core/registry/mod.zig");
pub const Registry = registry.Registry;

// Services
pub const runtime = @import("../../services/runtime/mod.zig");
pub const platform = @import("../../services/platform/mod.zig");
pub const shared = @import("../../services/shared/mod.zig");
pub const connectors = @import("../../services/connectors/mod.zig");
pub const ha = @import("../../services/ha/mod.zig");
pub const tasks = @import("../../services/tasks/mod.zig");
pub const lsp = @import("../../services/lsp/mod.zig");
pub const mcp = @import("../../services/mcp/mod.zig");
pub const acp = @import("../../services/acp/mod.zig");
pub const simd = @import("../../services/shared/simd/mod.zig");

// Features
pub const gpu = if (build_options.enable_gpu)
    @import("../../features/gpu/mod.zig")
else
    @import("../../features/gpu/stub.zig");
pub const ai = if (build_options.enable_ai)
    @import("../../features/ai/mod.zig")
else
    @import("../../features/ai/stub.zig");
pub const database = if (build_options.enable_database)
    @import("../../features/database/mod.zig")
else
    @import("../../features/database/stub.zig");
pub const network = if (build_options.enable_network)
    @import("../../features/network/mod.zig")
else
    @import("../../features/network/stub.zig");
pub const observability = if (build_options.enable_profiling)
    @import("../../features/observability/mod.zig")
else
    @import("../../features/observability/stub.zig");
pub const web = if (build_options.enable_web)
    @import("../../features/web/mod.zig")
else
    @import("../../features/web/stub.zig");
pub const analytics = if (build_options.enable_analytics)
    @import("../../features/analytics/mod.zig")
else
    @import("../../features/analytics/stub.zig");
pub const cloud = if (build_options.enable_cloud)
    @import("../../features/cloud/mod.zig")
else
    @import("../../features/cloud/stub.zig");
pub const auth = if (build_options.enable_auth)
    @import("../../features/auth/mod.zig")
else
    @import("../../features/auth/stub.zig");
pub const messaging = if (build_options.enable_messaging)
    @import("../../features/messaging/mod.zig")
else
    @import("../../features/messaging/stub.zig");
pub const cache = if (build_options.enable_cache)
    @import("../../features/cache/mod.zig")
else
    @import("../../features/cache/stub.zig");
pub const storage = if (build_options.enable_storage)
    @import("../../features/storage/mod.zig")
else
    @import("../../features/storage/stub.zig");
pub const mobile = if (build_options.enable_mobile)
    @import("../../features/mobile/mod.zig")
else
    @import("../../features/mobile/stub.zig");
pub const gateway = if (build_options.enable_gateway)
    @import("../../features/gateway/mod.zig")
else
    @import("../../features/gateway/stub.zig");
pub const search = if (build_options.enable_search)
    @import("../../features/search/mod.zig")
else
    @import("../../features/search/stub.zig");
pub const pages = if (build_options.enable_pages)
    @import("../../features/observability/pages/mod.zig")
else
    @import("../../features/observability/pages/stub.zig");
pub const benchmarks = if (build_options.enable_benchmarks)
    @import("../../features/benchmarks/mod.zig")
else
    @import("../../features/benchmarks/stub.zig");

// Aliases
pub const Gpu = gpu.Gpu;
pub const GpuBackend = gpu.Backend;

// Legacy entrypoints
pub fn init(allocator: std.mem.Allocator, cfg: Config) !Framework {
    return Framework.init(allocator, cfg);
}

pub fn initDefault(allocator: std.mem.Allocator) !Framework {
    return Framework.initDefault(allocator);
}

pub fn version() []const u8 {
    return build_options.package_version;
}
