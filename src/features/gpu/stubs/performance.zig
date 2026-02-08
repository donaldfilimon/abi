const std = @import("std");

pub const SyncEvent = struct {
    pub fn init() @This() {
        return .{};
    }
    pub fn signal(_: *@This()) void {}
    pub fn wait(_: *@This()) void {}
};

pub const KernelRing = struct {
    pub fn init(_: std.mem.Allocator, _: usize) !@This() {
        return error.GpuDisabled;
    }
    pub fn deinit(_: *@This()) void {}
};

pub const AdaptiveTiling = struct {
    pub const TileConfig = struct {
        tile_width: usize = 16,
        tile_height: usize = 16,
    };

    pub fn init(_: std.mem.Allocator) @This() {
        return .{};
    }
    pub fn deinit(_: *@This()) void {}
};
