//! Inference workload manager.

const std = @import("std");
const engine = @import("engine.zig");
const scheduler = @import("scheduler.zig");
const kv_cache = @import("kv_cache.zig");

pub const CoordinatorConfig = struct {
    engine_cfg: engine.Config,
    kv_cache_cfg: kv_cache.Config,
};

pub const InferenceCoordinator = struct {
    allocator: std.mem.Allocator,
    engine: engine.Engine,
    scheduler: scheduler.Scheduler,
    kv_cache: kv_cache.PagedKVCache,

    pub fn init(allocator: std.mem.Allocator, cfg: CoordinatorConfig) !InferenceCoordinator {
        var eng = try engine.Engine.init(allocator, cfg.engine_cfg);
        errdefer eng.deinit();

        var sched = try scheduler.Scheduler.init(allocator);
        errdefer sched.deinit();

        var cache = try kv_cache.PagedKVCache.init(allocator, cfg.kv_cache_cfg);
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
