//! Stubbed storage namespace for disabled database builds.

const std = @import("std");

pub const format = struct {
    pub const MAGIC: [4]u8 = .{ 'W', 'D', 'B', 'X' };
    pub const FORMAT_VERSION: u16 = 3;
    pub const SectionType = enum(u16) {
        metadata = 1,
        vectors = 2,
        bloom_filter = 3,
        vector_index = 4,
        lineage = 5,
        distributed = 6,
    };
};

pub const MAGIC = format.MAGIC;
pub const FORMAT_VERSION = format.FORMAT_VERSION;
pub const SectionType = format.SectionType;

pub const StorageConfig = struct {
    verify_checksums: bool = true,
    include_index: bool = true,
};

pub const StorageV2Config = StorageConfig;
pub const HnswGraphData = struct {
    entry_point: u32 = 0,
    max_layer: u32 = 0,
    neighbors: []const []const u32 = &.{},
};

pub fn freeHnswGraphData(_: std.mem.Allocator, _: HnswGraphData) void {}

pub fn saveDatabase(_: std.mem.Allocator, _: anytype, _: []const u8) !void {
    return error.FeatureDisabled;
}
pub fn saveDatabaseWithConfig(_: std.mem.Allocator, _: anytype, _: []const u8, _: StorageConfig) !void {
    return error.FeatureDisabled;
}
pub fn loadDatabase(_: std.mem.Allocator, _: []const u8) !@import("../stubs/misc.zig").database.Database {
    return error.FeatureDisabled;
}
pub fn loadDatabaseWithConfig(_: std.mem.Allocator, _: []const u8, _: StorageConfig) !@import("../stubs/misc.zig").database.Database {
    return error.FeatureDisabled;
}
pub fn saveDatabaseV2(_: std.mem.Allocator, _: anytype, _: []const u8, _: StorageV2Config) !void {
    return error.FeatureDisabled;
}
pub fn loadDatabaseV2(_: std.mem.Allocator, _: []const u8, _: StorageV2Config) !@import("../stubs/misc.zig").database.Database {
    return error.FeatureDisabled;
}
pub fn readFileBytesForTest(_: std.mem.Allocator, _: []const u8) ![]u8 {
    return error.FeatureDisabled;
}
pub fn writeFileBytesForTest(_: std.mem.Allocator, _: []const u8, _: []const u8) !void {
    return error.FeatureDisabled;
}
