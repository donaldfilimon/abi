//! Framework shutdown helpers for split lifecycle modules.

const std = @import("std");
const state_machine = @import("state_machine.zig");
const build_options = @import("build_options");
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

/// Composable error type for registry operations from framework shutdown helpers.
const registry_types = @import("../registry/types.zig");
pub const RegistryError = registry_types.Error;

/// Release feature contexts and mark the framework stopped.
pub fn deinit(self: anytype) void {
    if (self.state == .stopped or self.state == .uninitialized) return;

    state_machine.markStopping(&self.state);
    deinitFeatures(self);
    self.registry.deinit();
    self.runtime.deinit();
    state_machine.markStopped(&self.state);
}

/// Shutdown wrapper for timeout-based behavior (currently synchronous).
pub fn shutdownWithTimeout(self: anytype, _: u64) bool {
    deinit(self);
    return self.state == .stopped;
}

pub fn deinitFeatures(self: anytype) void {
    // Deinitialize in reverse order of initialization.
    // HA manager first (initialized last, after AI modules).
    if (self.ha) |*ha| {
        ha.deinit();
        self.ha = null;
    }

    // Split AI modules.
    deinitOptionalContext(ai_reasoning_mod.Context, &self.ai_reasoning);
    deinitOptionalContext(ai_training_mod.Context, &self.ai_training);
    deinitOptionalContext(ai_inference_mod.Context, &self.ai_inference);
    deinitOptionalContext(ai_core_mod.Context, &self.ai_core);

    // Standard feature modules (reverse order of initFeatureContexts).
    deinitOptionalContext(mobile_mod.Context, &self.mobile);
    deinitOptionalContext(benchmarks_mod.Context, &self.benchmarks);
    deinitOptionalContext(pages_mod.Context, &self.pages);
    deinitOptionalContext(gateway_mod.Context, &self.gateway);
    deinitOptionalContext(search_mod.Context, &self.search);
    deinitOptionalContext(storage_mod.Context, &self.storage);
    deinitOptionalContext(cache_mod.Context, &self.cache);
    deinitOptionalContext(messaging_mod.Context, &self.messaging);
    deinitOptionalContext(auth_mod.Context, &self.auth);
    deinitOptionalContext(analytics_mod.Context, &self.analytics);
    deinitOptionalContext(cloud_mod.Context, &self.cloud);
    deinitOptionalContext(web_mod.Context, &self.web);
    deinitOptionalContext(observability_mod.Context, &self.observability);
    deinitOptionalContext(network_mod.Context, &self.network);
    deinitOptionalContext(database_mod.Context, &self.database);
    deinitOptionalContext(ai_mod.Context, &self.ai);
    deinitOptionalContext(gpu_mod.Context, &self.gpu);
}

/// Safely deinitialize an optional feature context slot.
pub fn deinitOptionalContext(comptime Context: type, slot: *?*Context) void {
    if (slot.*) |ctx| {
        ctx.deinit();
        slot.* = null;
    }
}
