//! Inference workload manager stub.

const std = @import("std");
const inference = @import("../inference/mod.zig");

pub const CoordinatorConfig = struct {
    engine_cfg: inference.EngineConfig,
    kv_cache_cfg: inference.PagedKVCacheConfig,
};

pub const InferenceCoordinator = struct {
    pub fn init(_: std.mem.Allocator, _: CoordinatorConfig) !InferenceCoordinator {
        return .{};
    }

    pub fn deinit(_: *InferenceCoordinator) void {}
};
