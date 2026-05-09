//! Storage Stub Module
//!
//! API-compatible no-op implementations when storage is disabled.

const std = @import("std");
const stub_context = @import("../core/stub_helpers.zig");
pub const types = @import("types.zig");

pub const StorageConfig = types.StorageConfig;
pub const StorageBackend = types.StorageBackend;
pub const StorageError = types.StorageError;
pub const Error = StorageError;
pub const StorageObject = types.StorageObject;
pub const ObjectMetadata = types.ObjectMetadata;
pub const StorageStats = types.StorageStats;

const feature = stub_context.StubFeature(StorageConfig, StorageError);
pub const Context = feature.Context;
pub const init = feature.init;
pub const deinit = feature.deinit;
pub const isEnabled = feature.isEnabled;
pub const isInitialized = feature.isInitialized;

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

// ── Sub-module stubs (parity with mod.zig) ────────────────────────────
pub const backend_iface = struct {
    pub fn create(_: std.mem.Allocator) StorageError!void {
        return error.FeatureDisabled;
    }
};
pub const memory_backend = struct {
    pub fn create(_: std.mem.Allocator) StorageError!void {
        return error.FeatureDisabled;
    }
};
pub const local_backend = struct {
    pub fn create(_: std.mem.Allocator) StorageError!void {
        return error.FeatureDisabled;
    }
};
pub const validation = struct {
    pub fn validateKey(_: []const u8) bool {
        return false;
    }
};

test {
    std.testing.refAllDecls(@This());
}
