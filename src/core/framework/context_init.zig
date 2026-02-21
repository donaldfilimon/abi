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

const gpu_mod = if (build_options.enable_gpu) @import("../../features/gpu/mod.zig") else @import("../../features/gpu/stub.zig");
const ai_mod = if (build_options.enable_ai) @import("../../features/ai/mod.zig") else @import("../../features/ai/stub.zig");
const database_mod = if (build_options.enable_database) @import("../../features/database/mod.zig") else @import("../../features/database/stub.zig");
const network_mod = if (build_options.enable_network) @import("../../features/network/mod.zig") else @import("../../features/network/stub.zig");
const observability_mod = if (build_options.enable_profiling) @import("../../features/observability/mod.zig") else @import("../../features/observability/stub.zig");
const web_mod = if (build_options.enable_web) @import("../../features/web/mod.zig") else @import("../../features/web/stub.zig");
const cloud_mod = if (build_options.enable_cloud) @import("../../features/cloud/mod.zig") else @import("../../features/cloud/stub.zig");
const analytics_mod = if (build_options.enable_analytics) @import("../../features/analytics/mod.zig") else @import("../../features/analytics/stub.zig");
const auth_mod = if (build_options.enable_auth) @import("../../features/auth/mod.zig") else @import("../../features/auth/stub.zig");
const messaging_mod = if (build_options.enable_messaging) @import("../../features/messaging/mod.zig") else @import("../../features/messaging/stub.zig");
const cache_mod = if (build_options.enable_cache) @import("../../features/cache/mod.zig") else @import("../../features/cache/stub.zig");
const storage_mod = if (build_options.enable_storage) @import("../../features/storage/mod.zig") else @import("../../features/storage/stub.zig");
const search_mod = if (build_options.enable_search) @import("../../features/search/mod.zig") else @import("../../features/search/stub.zig");
const gateway_mod = if (build_options.enable_gateway) @import("../../features/gateway/mod.zig") else @import("../../features/gateway/stub.zig");
const pages_mod = if (build_options.enable_pages) @import("../../features/pages/mod.zig") else @import("../../features/pages/stub.zig");
const benchmarks_mod = if (build_options.enable_benchmarks) @import("../../features/benchmarks/mod.zig") else @import("../../features/benchmarks/stub.zig");
const mobile_mod = if (build_options.enable_mobile) @import("../../features/mobile/mod.zig") else @import("../../features/mobile/stub.zig");
const ai_core_mod = if (build_options.enable_ai) @import("../../features/ai/facades/core.zig") else @import("../../features/ai/facades/core_stub.zig");
const ai_inference_mod = if (build_options.enable_llm) @import("../../features/ai/facades/inference.zig") else @import("../../features/ai/facades/inference_stub.zig");
const ai_training_mod = if (build_options.enable_training) @import("../../features/ai/facades/training.zig") else @import("../../features/ai/facades/training_stub.zig");
const ai_reasoning_mod = if (build_options.enable_reasoning) @import("../../features/ai/facades/reasoning.zig") else @import("../../features/ai/facades/reasoning_stub.zig");
const ha_mod = @import("../../services/ha/mod.zig");
const runtime_mod = @import("../../services/runtime/mod.zig");

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

fn initFeatureContexts(comptime Framework: type, allocator: std.mem.Allocator, cfg: config_module.Config, fw: *Framework) Framework.Error!void {
    if (cfg.gpu) |gpu_cfg| {
        fw.gpu = try gpu_mod.Context.init(allocator, gpu_cfg);
        if (comptime build_options.enable_gpu) {
            try fw.registry.registerComptime(.gpu);
        }
    }

    if (cfg.ai) |ai_cfg| {
        fw.ai = try ai_mod.Context.init(allocator, ai_cfg);
        if (comptime build_options.enable_ai) {
            try fw.registry.registerComptime(.ai);
        }
    }

    if (cfg.database) |db_cfg| {
        fw.database = try database_mod.Context.init(allocator, db_cfg);
        if (comptime build_options.enable_database) {
            try fw.registry.registerComptime(.database);
        }
    }

    if (cfg.network) |net_cfg| {
        fw.network = try network_mod.Context.init(allocator, net_cfg);
        if (comptime build_options.enable_network) {
            try fw.registry.registerComptime(.network);
        }
    }

    if (cfg.observability) |obs_cfg| {
        fw.observability = try observability_mod.Context.init(allocator, obs_cfg);
        if (comptime build_options.enable_profiling) {
            try fw.registry.registerComptime(.observability);
        }
    }

    if (cfg.web) |web_cfg| {
        fw.web = try web_mod.Context.init(allocator, web_cfg);
        if (comptime build_options.enable_web) {
            try fw.registry.registerComptime(.web);
        }
    }

    if (cfg.cloud) |core_cloud| {
        const runtime_cloud = cloud_mod.CloudConfig{
            .memory_mb = core_cloud.memory_mb,
            .timeout_seconds = core_cloud.timeout_seconds,
            .tracing_enabled = core_cloud.tracing_enabled,
            .logging_enabled = core_cloud.logging_enabled,
            .log_level = @enumFromInt(@intFromEnum(core_cloud.log_level)),
        };
        fw.cloud = try cloud_mod.Context.init(allocator, runtime_cloud);
        if (comptime build_options.enable_cloud) {
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
        if (comptime build_options.enable_analytics) {
            try fw.registry.registerComptime(.analytics);
        }
    }

    if (cfg.auth) |auth_cfg| {
        fw.auth = try auth_mod.Context.init(allocator, auth_cfg);
        if (comptime build_options.enable_auth) {
            try fw.registry.registerComptime(.auth);
        }
    }

    if (cfg.messaging) |msg_cfg| {
        fw.messaging = try messaging_mod.Context.init(allocator, msg_cfg);
        if (comptime build_options.enable_messaging) {
            try fw.registry.registerComptime(.messaging);
        }
    }

    if (cfg.cache) |cache_cfg| {
        fw.cache = try cache_mod.Context.init(allocator, cache_cfg);
        if (comptime build_options.enable_cache) {
            try fw.registry.registerComptime(.cache);
        }
    }

    if (cfg.storage) |storage_cfg| {
        fw.storage = try storage_mod.Context.init(allocator, storage_cfg);
        if (comptime build_options.enable_storage) {
            try fw.registry.registerComptime(.storage);
        }
    }

    if (cfg.search) |search_cfg| {
        fw.search = try search_mod.Context.init(allocator, search_cfg);
        if (comptime build_options.enable_search) {
            try fw.registry.registerComptime(.search);
        }
    }

    if (cfg.gateway) |gateway_cfg| {
        fw.gateway = try gateway_mod.Context.init(allocator, gateway_cfg);
        if (comptime build_options.enable_gateway) {
            try fw.registry.registerComptime(.gateway);
        }
    }

    if (cfg.pages) |pages_cfg| {
        fw.pages = try pages_mod.Context.init(allocator, pages_cfg);
        if (comptime build_options.enable_pages) {
            try fw.registry.registerComptime(.pages);
        }
    }

    if (cfg.benchmarks) |benchmarks_cfg| {
        fw.benchmarks = try benchmarks_mod.Context.init(allocator, benchmarks_cfg);
        if (comptime build_options.enable_benchmarks) {
            try fw.registry.registerComptime(.benchmarks);
        }
    }

    if (cfg.mobile) |mobile_cfg| {
        fw.mobile = try mobile_mod.Context.init(allocator, mobile_cfg);
        if (comptime build_options.enable_mobile) {
            try fw.registry.registerComptime(.mobile);
        }
    }

    if (cfg.ai) |ai_cfg| {
        // AI split module initialization is non-fatal and logs only.
        if (comptime build_options.enable_ai) {
            fw.ai_core = ai_core_mod.Context.init(
                allocator,
                ai_cfg,
            ) catch |err| blk: {
                std.log.warn("ai_core init failed: {t}", .{err});
                break :blk null;
            };
        }
        if (comptime build_options.enable_llm) {
            fw.ai_inference = ai_inference_mod.Context.init(
                allocator,
                ai_cfg,
            ) catch |err| blk: {
                std.log.warn("ai_inference init failed: {t}", .{err});
                break :blk null;
            };
        }
        if (comptime build_options.enable_training) {
            fw.ai_training = ai_training_mod.Context.init(
                allocator,
                ai_cfg,
            ) catch |err| blk: {
                std.log.warn("ai_training init failed: {t}", .{err});
                break :blk null;
            };
        }
        if (comptime build_options.enable_reasoning) {
            fw.ai_reasoning = ai_reasoning_mod.Context.init(
                allocator,
                ai_cfg,
            ) catch |err| blk: {
                std.log.warn("ai_reasoning init failed: {t}", .{err});
                break :blk null;
            };
        }
    }

    // Always initialize HA manager using default provider when available.
    fw.ha = ha_mod.HaManager.init(allocator, .{});

    state_machine.markRunning(&fw.state);
}
