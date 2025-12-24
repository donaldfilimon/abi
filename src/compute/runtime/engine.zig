//! Compute engine - core runtime
//!
//! Main execution engine with work-stealing scheduler,
//! result caching, and shutdown semantics.

const std = @import("std");
const config = @import("config.zig");

pub const Engine = struct {
    allocator: std.mem.Allocator,
    workers: []Worker,
    running: std.atomic.Value(bool),
    config: config.EngineConfig,

    const Worker = struct {
        id: u32,
    };

    pub fn init(allocator: std.mem.Allocator, cfg: config.EngineConfig) !*Engine {
        const self = try allocator.create(Engine);
        errdefer allocator.destroy(self);

        self.* = .{
            .allocator = allocator,
            .workers = &[_]Worker{},
            .running = std.atomic.Value(bool).init(false),
            .config = cfg,
        };

        self.* = .{
            .allocator = allocator,
            .workers = &[_]Worker{},
            .running = std.atomic.Value(bool).init(false),
            .config = cfg,
        };

        return self;
    }

    pub fn deinit(self: *Engine) void {
        self.running.store(false, .release);
        self.allocator.free(self.workers);
        self.allocator.destroy(self);
    }
};
