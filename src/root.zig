//! Public ABI package root.
//!
//! Canonical API: `abi.<domain>` (e.g. `abi.gpu`, `abi.ai`, `abi.connectors`).
//! Compat bridges `abi.features.*` and `abi.services.*` re-export the same
//! modules for callers that have not yet migrated.

const std = @import("std");
const build_options = @import("build_options");
const internal = @import("internal/mod.zig");

// ── Core ─────────────────────────────────────────────────────────────────

pub const config = @import("core/config/mod.zig");
pub const Config = config.Config;
pub const Feature = config.Feature;

pub const errors = @import("core/errors.zig");
pub const FrameworkError = errors.FrameworkError;

pub const registry = @import("core/registry/mod.zig");
pub const Registry = registry.Registry;

pub const framework = internal.app;

// ── Services (non-feature-gated) ─────────────────────────────────────────

pub const foundation = internal.foundation;
pub const runtime = internal.runtime;
pub const platform = internal.platform;
pub const connectors = internal.integrations;
pub const tasks = internal.tooling;
pub const mcp = @import("services/mcp/mod.zig");
pub const lsp = @import("services/lsp/mod.zig");
pub const acp = @import("services/acp/mod.zig");
pub const ha = @import("services/ha/mod.zig");

// ── Features (comptime-gated mod/stub) ───────────────────────────────────

pub const gpu = if (build_options.feat_gpu) @import("features/gpu/mod.zig") else @import("features/gpu/stub.zig");
pub const ai = if (build_options.feat_ai) internal.ai.mod else internal.ai.stub;
pub const database = if (build_options.feat_database) internal.data.mod else internal.data.stub;
pub const network = if (build_options.feat_network) internal.network.mod else internal.network.stub;
pub const observability = if (build_options.feat_profiling) internal.observe.mod else internal.observe.stub;
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

// ── Compat bridges (`abi.services.*`, `abi.features.*`) ──────────────────
// These re-export the canonical top-level declarations so that existing
// callers using `abi.services.X` or `abi.features.X` continue to work.

const root = @This();

pub const services = struct {
    pub const foundation = root.foundation;
    pub const shared = root.foundation;
    pub const runtime = root.runtime;
    pub const platform = root.platform;
    pub const connectors = root.connectors;
    pub const tasks = root.tasks;
    pub const lsp = root.lsp;
    pub const mcp = root.mcp;
    pub const acp = root.acp;
    pub const ha = root.ha;
    pub const simd = root.foundation.simd;
};

pub const features = struct {
    pub const gpu = root.gpu;
    pub const ai = root.ai;
    pub const database = root.database;
    pub const network = root.network;
    pub const observability = root.observability;
    pub const web = root.web;
    pub const analytics = root.analytics;
    pub const cloud = root.cloud;
    pub const auth = root.auth;
    pub const messaging = root.messaging;
    pub const cache = root.cache;
    pub const storage = root.storage;
    pub const mobile = root.mobile;
    pub const gateway = root.gateway;
    pub const search = root.search;
    pub const pages = root.pages;
    pub const benchmarks = root.benchmarks;
    pub const compute = root.compute;
    pub const documents = root.documents;
    pub const desktop = root.desktop;
};

test {
    std.testing.refAllDecls(@This());
}
