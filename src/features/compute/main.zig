const std = @import("std");

pub const core = @import("core.zig");
pub const lockfree = @import("lockfree.zig");
pub const workload = @import("workload.zig");
pub const compute_engine = @import("compute_engine.zig");
pub const allocator = @import("allocator.zig");
pub const gpu = @import("gpu.zig");
pub const network = @import("network.zig");
pub const profiler = @import("profiler.zig");

pub const FRAMEWORK_VERSION = "0.1.0";

pub const FrameworkConfig = struct {
    worker_count: ?u32 = null,
    enable_gpu: bool = false,
    enable_network: bool = false,
    enable_profiler: bool = false,
};

pub const Framework = struct {
    allocator: std.mem.Allocator,
    engine: *compute_engine.ComputeEngine,
    profiler_instance: ?profiler.Profiler = null,

    pub fn init(backing_allocator: std.mem.Allocator, config: FrameworkConfig) !Framework {
        const workers = config.worker_count orelse @as(u32, @intCast(std.Thread.getCpuCount() catch 1));
        const engine = try compute_engine.ComputeEngine.init(backing_allocator, workers);

        var profiler_instance: ?profiler.Profiler = null;
        if (config.enable_profiler) {
            profiler_instance = try profiler.Profiler.init(backing_allocator);
        }

        _ = config.enable_gpu;
        _ = config.enable_network;

        return .{
            .allocator = backing_allocator,
            .engine = engine,
            .profiler_instance = profiler_instance,
        };
    }

    pub fn deinit(self: *Framework) void {
        if (self.profiler_instance) |*prof| {
            prof.deinit();
            self.profiler_instance = null;
        }
        self.engine.deinit();
    }
};

test "framework init and version" {
    const testing = std.testing;

    var framework = try Framework.init(testing.allocator, .{ .worker_count = 1 });
    defer framework.deinit();

    try testing.expect(FRAMEWORK_VERSION.len > 0);
}
