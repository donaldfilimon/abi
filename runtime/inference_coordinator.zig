//! Inference workload manager.

const std = @import("std");
const inference = @import("../inference/mod.zig");

pub const CoordinatorConfig = struct {
    engine_cfg: inference.EngineConfig,
    kv_cache_cfg: inference.PagedKVCacheConfig,
};

pub const InferenceCoordinator = struct {
    allocator: std.mem.Allocator,
    engine: inference.Engine,
    scheduler: inference.Scheduler,
    kv_cache: inference.PagedKVCache,

    pub fn init(allocator: std.mem.Allocator, cfg: CoordinatorConfig) !InferenceCoordinator {
        var eng = try inference.Engine.init(allocator, cfg.engine_cfg);
        errdefer eng.deinit();

        var sched = try inference.Scheduler.init(allocator);
        errdefer sched.deinit();

        var cache = try inference.PagedKVCache.init(allocator, cfg.kv_cache_cfg);
        errdefer cache.deinit();

        return .{
            .allocator = allocator,
            .engine = eng,
            .scheduler = sched,
            .kv_cache = cache,
        };
    }

    pub fn deinit(self: *InferenceCoordinator) void {
        self.kv_cache.deinit();
        self.scheduler.deinit();
        self.engine.deinit();
    }
};
