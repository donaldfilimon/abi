//! Framework shutdown helpers for split lifecycle modules.

const std = @import("std");
const state_machine = @import("state_machine.zig");

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
    self.config.plugins.deinit(self.allocator);
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

    // Configless features (reverse order: desktop, documents, compute).
    // These are heap-allocated by the framework (not self-owned), so we
    // must destroy the allocation after calling deinit.
    deinitConfiglessContext(desktop_mod.Context, &self.desktop, self.allocator);
    deinitConfiglessContext(documents_mod.Context, &self.documents, self.allocator);
    deinitConfiglessContext(compute_mod.Context, &self.compute, self.allocator);

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

/// Deinitialize a configless feature context and free the heap allocation.
/// Used for features whose Context.init returns a value (not a pointer),
/// so the framework owns the heap allocation.
fn deinitConfiglessContext(comptime Context: type, slot: *?*Context, allocator: std.mem.Allocator) void {
    if (slot.*) |ctx| {
        ctx.deinit();
        allocator.destroy(ctx);
        slot.* = null;
    }
}

test {
    std.testing.refAllDecls(@This());
}
