//! Sync manager stub (cross-frontend skill sync).

const std = @import("std");
const types = @import("../protocol/types.zig");

pub const SyncManager = struct {
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) SyncManager {
        return .{ .allocator = allocator };
    }

    pub fn deinit(self: *SyncManager) void {
        _ = self;
    }

    pub fn emit(self: *SyncManager, event: types.SyncEvent) !void {
        _ = self;
        _ = event;
    }
};
