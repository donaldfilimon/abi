//! Context initialization helpers for framework lifecycle.
//!
//! This module owns framework startup composition and feature registration so
//! `core/framework.zig` can remain a thin API façade.

const std = @import("std");
const build_options = @import("build_options");
const config_module = @import("../config/mod.zig");
const registry_mod = @import("../registry/mod.zig");
const state_machine = @import("state_machine.zig");
const shutdown = @import("shutdown.zig");

// Shared comptime-gated feature imports (DRY: single source of truth).
const fi = @import("feature_imports.zig");
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
const compute_mod = fi.compute_mod;
const documents_mod = fi.documents_mod;
const desktop_mod = fi.desktop_mod;
const ha_mod = @import("../../protocols/ha/mod.zig");
const runtime_mod = @import("../../runtime/mod.zig");

/// Initialize a framework with the provided configuration.
pub fn init(comptime Framework: type, allocator: std.mem.Allocator, cfg: config_module.Config) Framework.Error!Framework {
    var fw = Framework{
        .allocator = allocator,
        .config = cfg,
        .state = .initializing,
        .registry = registry_mod.Registry.init(allocator),
        .runtime = undefined,
    };
    errdefer fw.registry.deinit();

    // Configure feature graph from configuration and compile-time flags.
    try config_module.validate(cfg);

    // Initialize runtime (always available).
    fw.runtime = try runtime_mod.Context.init(allocator);
    errdefer fw.runtime.deinit();

    // Initialize enabled features and register them.
    errdefer shutdown.deinitFeatures(&fw);
    try initFeatureContexts(Framework, allocator, cfg, &fw);
    return fw;
}

/// Initialize with explicit I/O context.
pub fn initWithIo(comptime Framework: type, allocator: std.mem.Allocator, cfg: config_module.Config, io: std.Io) Framework.Error!Framework {
    var fw = try init(Framework, allocator, cfg);
    fw.io = io;
    return fw;
}

pub fn initDefault(comptime Framework: type, allocator: std.mem.Allocator) Framework.Error!Framework {
    return init(Framework, allocator, config_module.Config.defaults());
}

pub fn initMinimal(comptime Framework: type, allocator: std.mem.Allocator) Framework.Error!Framework {
    return init(Framework, allocator, config_module.Config.minimal());
}

/// Comptime descriptor for a standard feature: config field → framework field → build flag.
/// Used by initStandardFeatures to eliminate per-feature boilerplate.
const FeatureSpec = struct {
    cfg_field: []const u8,
    fw_field: []const u8,
    feat_flag: []const u8,
    registry_id: config_module.Feature,
};

/// Standard features that follow the pattern:
///   if (cfg.X) |c| { fw.X = try X_mod.Context.init(alloc, c); register(.X); }
const standard_features = [_]FeatureSpec{
    .{ .cfg_field = "gpu", .fw_field = "gpu", .feat_flag = "feat_gpu", .registry_id = .gpu },
    .{ .cfg_field = "ai", .fw_field = "ai", .feat_flag = "feat_ai", .registry_id = .ai },
    .{ .cfg_field = "database", .fw_field = "database", .feat_flag = "feat_database", .registry_id = .database },
    .{ .cfg_field = "network", .fw_field = "network", .feat_flag = "feat_network", .registry_id = .network },
    .{ .cfg_field = "observability", .fw_field = "observability", .feat_flag = "feat_profiling", .registry_id = .observability },
    .{ .cfg_field = "web", .fw_field = "web", .feat_flag = "feat_web", .registry_id = .web },
    .{ .cfg_field = "auth", .fw_field = "auth", .feat_flag = "feat_auth", .registry_id = .auth },
    .{ .cfg_field = "messaging", .fw_field = "messaging", .feat_flag = "feat_messaging", .registry_id = .messaging },
    .{ .cfg_field = "cache", .fw_field = "cache", .feat_flag = "feat_cache", .registry_id = .cache },
    .{ .cfg_field = "storage", .fw_field = "storage", .feat_flag = "feat_storage", .registry_id = .storage },
    .{ .cfg_field = "search", .fw_field = "search", .feat_flag = "feat_search", .registry_id = .search },
    .{ .cfg_field = "gateway", .fw_field = "gateway", .feat_flag = "feat_gateway", .registry_id = .gateway },
    .{ .cfg_field = "pages", .fw_field = "pages", .feat_flag = "feat_pages", .registry_id = .pages },
    .{ .cfg_field = "benchmarks", .fw_field = "benchmarks", .feat_flag = "feat_benchmarks", .registry_id = .benchmarks },
    .{ .cfg_field = "mobile", .fw_field = "mobile", .feat_flag = "feat_mobile", .registry_id = .mobile },
};

/// Initialize all standard features using the comptime feature spec table.
fn initStandardFeatures(comptime Framework: type, allocator: std.mem.Allocator, cfg: config_module.Config, fw: *Framework) Framework.Error!void {
    inline for (standard_features) |spec| {
        if (@field(cfg, spec.cfg_field)) |feature_cfg| {
            @field(fw, spec.fw_field) = try @field(fi, spec.fw_field ++ "_mod").Context.init(allocator, feature_cfg);
            if (comptime @field(build_options, spec.feat_flag)) {
                try fw.registry.registerComptime(spec.registry_id);
            }
        }
    }
}

/// Initialize features requiring config conversion (cloud, analytics).
fn initConvertedFeatures(comptime Framework: type, allocator: std.mem.Allocator, cfg: config_module.Config, fw: *Framework) Framework.Error!void {
    if (cfg.cloud) |core_cloud| {
        const runtime_cloud = cloud_mod.CloudConfig{
            .memory_mb = core_cloud.memory_mb,
            .timeout_seconds = core_cloud.timeout_seconds,
            .tracing_enabled = core_cloud.tracing_enabled,
            .logging_enabled = core_cloud.logging_enabled,
            .log_level = @enumFromInt(@intFromEnum(core_cloud.log_level)),
        };
        fw.cloud = try cloud_mod.Context.init(allocator, runtime_cloud);
        if (comptime build_options.feat_cloud) {
            try fw.registry.registerComptime(.cloud);
        }
    }

    if (cfg.analytics) |analytics_cfg| {
        fw.analytics = try analytics_mod.Context.init(allocator, .{
            .buffer_capacity = analytics_cfg.buffer_capacity,
            .enable_timestamps = analytics_cfg.enable_timestamps,
            .app_id = analytics_cfg.app_id,
            .flush_interval_ms = analytics_cfg.flush_interval_ms,
        });
        if (comptime build_options.feat_analytics) {
            try fw.registry.registerComptime(.analytics);
        }
    }
}

/// Initialize a configless feature: allocate a Context on the heap and init with just an allocator.
fn initConfigless(comptime Mod: type, allocator: std.mem.Allocator) !*Mod.Context {
    const ctx = try allocator.create(Mod.Context);
    ctx.* = Mod.Context.init(allocator);
    return ctx;
}

/// Initialize features that have no config struct (allocator-only init).
fn initConfiglessFeatures(comptime Framework: type, allocator: std.mem.Allocator, fw: *Framework) Framework.Error!void {
    if (comptime build_options.feat_compute) {
        fw.compute = initConfigless(compute_mod, allocator) catch return error.OutOfMemory;
        try fw.registry.registerComptime(.compute);
    }
    if (comptime build_options.feat_documents) {
        fw.documents = initConfigless(documents_mod, allocator) catch return error.OutOfMemory;
        try fw.registry.registerComptime(.documents);
    }
    if (comptime build_options.feat_desktop) {
        fw.desktop = initConfigless(desktop_mod, allocator) catch return error.OutOfMemory;
        try fw.registry.registerComptime(.desktop);
    }
}

fn initFeatureContexts(comptime Framework: type, allocator: std.mem.Allocator, cfg: config_module.Config, fw: *Framework) Framework.Error!void {
    try initStandardFeatures(Framework, allocator, cfg, fw);
    try initConvertedFeatures(Framework, allocator, cfg, fw);
    try initConfiglessFeatures(Framework, allocator, fw);

    // Always initialize HA manager using default provider when available.
    fw.ha = ha_mod.HaManager.init(allocator, .{});

    state_machine.markRunning(&fw.state);
}

test {
    std.testing.refAllDecls(@This());
}
