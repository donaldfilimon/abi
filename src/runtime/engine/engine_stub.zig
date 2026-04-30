//! Engine stub for platforms without thread support.

const std = @import("std");

pub const EngineConfig = struct {
    max_tasks: u32 = 1000,
    worker_threads: u32 = 4,
};

pub const Engine = struct {
    pub fn init(_: std.mem.Allocator, _: EngineConfig) !Engine {
        return error.ThreadsNotSupported;
    }
    pub fn deinit(_: *Engine) void {}
};
