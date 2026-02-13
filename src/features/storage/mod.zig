//! Storage Module
//!
//! Unified file/object storage abstraction (local, S3, GCS).

const std = @import("std");
const core_config = @import("../../core/config/storage.zig");

pub const StorageConfig = core_config.StorageConfig;
pub const StorageBackend = core_config.StorageBackend;

pub const StorageError = error{
    FeatureDisabled,
    ObjectNotFound,
    BucketNotFound,
    PermissionDenied,
    StorageFull,
    OutOfMemory,
};

pub const StorageObject = struct {
    key: []const u8 = "",
    size: u64 = 0,
    content_type: []const u8 = "application/octet-stream",
    last_modified: u64 = 0,
};

pub const Context = struct {
    allocator: std.mem.Allocator,
    config: StorageConfig,

    pub fn init(allocator: std.mem.Allocator, config: StorageConfig) !*Context {
        const ctx = try allocator.create(Context);
        ctx.* = .{ .allocator = allocator, .config = config };
        return ctx;
    }

    pub fn deinit(self: *Context) void {
        self.allocator.destroy(self);
    }
};

pub fn init(_: std.mem.Allocator, _: StorageConfig) StorageError!void {}
pub fn deinit() void {}
pub fn isEnabled() bool {
    return true;
}
pub fn isInitialized() bool {
    return true;
}

pub fn putObject(_: std.mem.Allocator, _: []const u8, _: []const u8) StorageError!void {}
pub fn getObject(_: std.mem.Allocator, _: []const u8) StorageError![]const u8 {
    return "";
}
pub fn deleteObject(_: []const u8) StorageError!bool {
    return false;
}
pub fn listObjects(_: std.mem.Allocator, _: []const u8) StorageError![]StorageObject {
    return &.{};
}
