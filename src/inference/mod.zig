//! Top-level inference primitives.
//!
//! This module is the canonical public home for ABI inference runtime types.
//! It implements the engine backend interface, allowing for model integration
//! and inference orchestration within the ABI pipeline.

pub const engine = @import("engine.zig");
pub const scheduler = @import("scheduler.zig");
pub const sampler = @import("sampler.zig");
pub const kv_cache = @import("kv_cache.zig");

pub const Engine = engine.Engine;
pub const EngineConfig = engine.Config;
pub const EngineResult = engine.Result;
pub const EngineStats = engine.Stats;
pub const FinishReason = engine.FinishReason;
pub const Backend = engine.Backend;

pub const Scheduler = scheduler.Scheduler;
pub const Request = scheduler.Request;

pub const Sampler = sampler.Sampler;
pub const SamplingParams = sampler.SamplingParams;

pub const PagedKVCache = kv_cache.PagedKVCache;
pub const PagedKVCacheConfig = kv_cache.Config;

/// Engine entry point for framework integration
pub const Context = struct {
    engine: *Engine,

    pub fn init(allocator: std.mem.Allocator, config: EngineConfig) !Context {
        const eng = try allocator.create(Engine);
        eng.* = try Engine.init(allocator, config);
        return .{ .engine = eng };
    }

    pub fn deinit(self: *Context) void {
        self.engine.deinit();
        self.engine.allocator.destroy(self.engine);
    }
};

const std = @import("std");

test {
    std.testing.refAllDecls(@This());
}
