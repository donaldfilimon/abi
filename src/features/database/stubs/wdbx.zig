const std = @import("std");
const types = @import("types.zig");

pub const wdbx = struct {
    pub const DatabaseHandle = types.DatabaseHandle;
    pub const SearchResult = types.SearchResult;
    pub const VectorView = types.VectorView;
    pub const Stats = types.Stats;
    pub const BatchItem = types.BatchItem;
    pub const DatabaseConfig = types.DatabaseConfig;

    pub fn createDatabase(_: std.mem.Allocator, _: []const u8) !types.DatabaseHandle {
        return error.DatabaseDisabled;
    }
    pub fn createDatabaseWithConfig(_: std.mem.Allocator, _: []const u8, _: DatabaseConfig) !types.DatabaseHandle {
        return error.DatabaseDisabled;
    }
    pub fn connectDatabase(_: std.mem.Allocator, _: []const u8) !types.DatabaseHandle {
        return error.DatabaseDisabled;
    }
    pub fn closeDatabase(_: *types.DatabaseHandle) void {}
    pub fn insertVector(_: *types.DatabaseHandle, _: u64, _: []const f32, _: ?[]const u8) !void {
        return error.DatabaseDisabled;
    }
    pub fn insertBatch(_: *types.DatabaseHandle, _: []const types.BatchItem) !void {
        return error.DatabaseDisabled;
    }
    pub fn searchVectors(_: *types.DatabaseHandle, _: std.mem.Allocator, _: []const f32, _: usize) ![]types.SearchResult {
        return error.DatabaseDisabled;
    }
    pub fn searchVectorsInto(_: *types.DatabaseHandle, _: []const f32, _: usize, _: []types.SearchResult) usize {
        return 0;
    }
    pub fn deleteVector(_: *types.DatabaseHandle, _: u64) bool {
        return false;
    }
    pub fn updateVector(_: *types.DatabaseHandle, _: u64, _: []const f32) !bool {
        return error.DatabaseDisabled;
    }
    pub fn getVector(_: *types.DatabaseHandle, _: u64) ?types.VectorView {
        return null;
    }
    pub fn listVectors(_: *types.DatabaseHandle, _: std.mem.Allocator, _: usize) ![]types.VectorView {
        return error.DatabaseDisabled;
    }
    pub fn getStats(_: *types.DatabaseHandle) types.Stats {
        return .{};
    }
    pub fn optimize(_: *types.DatabaseHandle) !void {
        return error.DatabaseDisabled;
    }
    pub fn backup(_: *types.DatabaseHandle, _: []const u8) !void {
        return error.DatabaseDisabled;
    }
    pub fn restore(_: *types.DatabaseHandle, _: []const u8) !void {
        return error.DatabaseDisabled;
    }
};

test {
    std.testing.refAllDecls(@This());
}
