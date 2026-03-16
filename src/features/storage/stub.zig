//! Storage Stub Module
//!
//! API-compatible no-op implementations when storage is disabled.

const std = @import("std");
const core_config = @import("../../core/config/platform.zig");
const stub_context = @import("../../core/stub_context.zig");

pub const StorageConfig = core_config.StorageConfig;
pub const StorageBackend = core_config.StorageBackend;

pub const StorageError = error{
    FeatureDisabled,
    ObjectNotFound,
    BucketNotFound,
    PermissionDenied,
    StorageFull,
    OutOfMemory,
    InvalidKey,
    BackendNotAvailable,
};

pub const StorageObject = struct {
    key: []const u8 = "",
    size: u64 = 0,
    content_type: []const u8 = "application/octet-stream",
    last_modified: u64 = 0,
};

pub const ObjectMetadata = struct {
    content_type: []const u8 = "application/octet-stream",
    custom: [4]MetadataEntry = [_]MetadataEntry{.{}} ** 4,
    custom_count: u8 = 0,

    pub const MetadataEntry = struct {
        key: []const u8 = "",
        value: []const u8 = "",
    };
};

pub const StorageStats = struct {
    total_objects: u64 = 0,
    total_bytes: u64 = 0,
    backend: StorageBackend = .memory,
};

pub const Context = stub_context.StubContext(StorageConfig);

pub fn init(_: std.mem.Allocator, _: StorageConfig) StorageError!void {
    return error.FeatureDisabled;
}
pub fn deinit() void {}
pub fn isEnabled() bool {
    return false;
}
pub fn isInitialized() bool {
    return false;
}

pub fn putObject(_: std.mem.Allocator, _: []const u8, _: []const u8) StorageError!void {
    return error.FeatureDisabled;
}
pub fn putObjectWithMetadata(
    _: std.mem.Allocator,
    _: []const u8,
    _: []const u8,
    _: ObjectMetadata,
) StorageError!void {
    return error.FeatureDisabled;
}
pub fn getObject(_: std.mem.Allocator, _: []const u8) StorageError![]const u8 {
    return error.FeatureDisabled;
}
pub fn deleteObject(_: []const u8) StorageError!bool {
    return error.FeatureDisabled;
}
pub fn objectExists(_: []const u8) StorageError!bool {
    return error.FeatureDisabled;
}
pub fn listObjects(_: std.mem.Allocator, _: []const u8) StorageError![]StorageObject {
    return error.FeatureDisabled;
}
pub fn stats() StorageStats {
    return .{};
}

test {
    std.testing.refAllDecls(@This());
}
