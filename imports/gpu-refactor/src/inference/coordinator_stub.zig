//! Inference workload manager stub.

const std = @import("std");
const engine = @import("engine.zig");
const scheduler = @import("scheduler.zig");
const kv_cache = @import("kv_cache.zig");

pub const CoordinatorConfig = struct {
    engine_cfg: engine.Config,
    kv_cache_cfg: kv_cache.Config,
};

pub const InferenceCoordinator = struct {
    pub fn init(_: std.mem.Allocator, _: CoordinatorConfig) !InferenceCoordinator {
        return .{};
    }

    pub fn deinit(_: *InferenceCoordinator) void {}
};
