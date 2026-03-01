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
const ai_core_mod = fi.ai_core_mod;
const ai_inference_mod = fi.ai_inference_mod;
const ai_training_mod = fi.ai_training_mod;
const ai_reasoning_mod = fi.ai_reasoning_mod;
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
        // AI sub-modules fail non-fatally: the main `ai` module is available but
        // specialized sub-features (core, inference, training, reasoning) may be null.
        // Users can check via abi.features.ai.isLlmEnabled() or `abi system-info`.
        if (comptime build_options.enable_ai) {
            fw.ai_core = ai_core_mod.Context.init(
                allocator,
                ai_cfg,
            ) catch |err| blk: {
                std.log.warn("ai.core sub-module init failed (non-fatal): {t} — check `abi system-info`", .{err});
                break :blk null;
            };
        }
        if (comptime build_options.enable_llm) {
            fw.ai_inference = ai_inference_mod.Context.init(
                allocator,
                ai_cfg,
            ) catch |err| blk: {
                std.log.warn("ai.inference sub-module init failed (non-fatal): {t} — check `abi system-info`", .{err});
                break :blk null;
            };
        }
        if (comptime build_options.enable_training) {
            fw.ai_training = ai_training_mod.Context.init(
                allocator,
                ai_cfg,
            ) catch |err| blk: {
                std.log.warn("ai.training sub-module init failed (non-fatal): {t} — check `abi system-info`", .{err});
                break :blk null;
            };
        }
        if (comptime build_options.enable_reasoning) {
            fw.ai_reasoning = ai_reasoning_mod.Context.init(
                allocator,
                ai_cfg,
            ) catch |err| blk: {
                std.log.warn("ai.reasoning sub-module init failed (non-fatal): {t} — check `abi system-info`", .{err});
                break :blk null;
            };
        }
    }

    // Always initialize HA manager using default provider when available.
    fw.ha = ha_mod.HaManager.init(allocator, .{});

    state_machine.markRunning(&fw.state);
}

test {
    std.testing.refAllDecls(@This());
}
